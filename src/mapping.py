#!/usr/bin/env python
""" Semantic mapping

Author: Henry Zhang
Date:February 24, 2020

"""
import cv2
import numpy as np
import os
import os.path as osp
import rospy
import sys

# Add src directory into the path
sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), "../")))

from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseStamped, Transform, Pose, Quaternion
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs import point_cloud2 as pc2
from tf import Transformer, TransformListener, TransformerROS
from tf.transformations import quaternion_matrix, euler_from_quaternion, euler_matrix

from src.camera import camera_setup_1, camera_setup_6
from src.config.mapping_cfg import get_cfg_defaults
from src.data.confusion_matrix import ConfusionMatrix
from src.homography import generate_homography
from src.rendering import color_map_local
from src.utils import homogenize, dehomogenize, get_rotation_from_angle_2d
from src.utils_ros import set_map_pose, get_transformation, get_transform_from_pose, create_point_cloud


class SemanticMapping:
    """
    Create a semantic bird's eye view map from the LiDAR sensor and 2D semantic segmentation image. The BEV map is
    represented by a grid.

    """

    def __init__(self, cfg):
        """

        Args:
            cfg: Configuration file - Please read config/mapping_cfg
        """
        # Sanity check
        assert len(cfg.LABELS) == len(cfg.LABELS_NAMES) == len(cfg.LABEL_COLORS)

        # Set up ros subscribers
        self.sub_pose = rospy.Subscriber("/current_pose", PoseStamped, self.pose_callback)
        self.image_sub_cam1 = rospy.Subscriber("/camera1/semantic", Image, self.image_callback)
        self.image_sub_cam6 = rospy.Subscriber("/camera6/semantic", Image, self.image_callback)
        if cfg.DEPTH_METHOD == 'points_map':
            self.sub_pcd = rospy.Subscriber("/reduced_map", PointCloud2, self.pcd_callback)
        elif cfg.DEPTH_METHOD == 'points_raw':
            self.sub_pcd = rospy.Subscriber("/points_raw", PointCloud2, self.pcd_callback)
        else:
            rospy.logwarn("Depth estimation method set to others, use planar assumption!")

        # Set up ros publishers
        self.pub_semantic_local_map = rospy.Publisher("/semantic_local_map", Image, queue_size=5)
        self.pub_pcd = rospy.Publisher("/semantic_point_cloud", PointCloud2, queue_size=5)

        self.tf_listener = TransformListener()
        self.tf_ros = TransformerROS()
        self.bridge = CvBridge()

        self.output_dir = cfg.OUTPUT_DIR
        self.depth_method = cfg.DEPTH_METHOD

        self.pose = None
        self.pose_queue = []
        self.pose_time = None
        self.cam1 = camera_setup_1()
        self.cam6 = camera_setup_6()
        rospy.logwarn("currently only for front view")

        self.pcd = None
        self.pcd_frame_id = None
        self.pcd_queue = []
        self.pcd_header_queue = []
        self.pcd_time = None
        self.pcd_intensity_threshold = cfg.PCD.INTENSITY_THLD
        self.use_pcd_intensity = cfg.PCD.USE_INTENSITY

        self.map = None
        self.map_pose = None
        self.map_boundary = cfg.BOUNDARY
        self.resolution = cfg.RESOLUTION
        self.save_map_to_file = False
        self.label_names = cfg.LABELS_NAMES
        self.label_colors = np.array(cfg.LABEL_COLORS)

        self.map_width = int((self.map_boundary[0][1] - self.map_boundary[0][0]) / self.resolution)
        self.map_height = int((self.map_boundary[1][1] - self.map_boundary[1][0]) / self.resolution)
        self.map_depth = len(self.label_names)

        self.position_rel = np.array([[0, 0, 0]]).T
        self.yaw_rel = 0

        self.preprocessing()

        # Instruction
        # 1. Please download the confusion matrix from the Google Drive
        # 2. Then create a directory called "external_data/confusion_matrix" and save the download folder into the
        # directory.
        # 3. Set the load_path to the path of the cfn_mtx.npy
        confusion_matrix = ConfusionMatrix(load_path=cfg.CONFUSION_MTX.LOAD_PATH)
        self.confusion_matrix = confusion_matrix.get_submatrix(cfg.LABELS, to_probability=True)

    def preprocessing(self):
        """ Setup constant matrices """
        self.T_velodyne_to_basklink = self.set_velodyne_to_baselink()
        self.T_cam1_to_base = np.matmul(self.T_velodyne_to_basklink, self.cam1.T)
        self.T_cam6_to_base = np.matmul(self.T_velodyne_to_basklink, self.cam6.T)

        self.discretize_matrix_inv = np.array([
            [self.resolution, 0, self.map_boundary[0][0]],
            [0, -self.resolution, self.map_boundary[1][1]],
            [0, 0, 1],
        ]).astype(np.float)
        self.discretize_matrix = np.linalg.inv(self.discretize_matrix_inv)

        self.anchor_points = np.array([
            [self.map_width, self.map_width / 3, self.map_width, self.map_width / 3],
            [self.map_height / 4, self.map_height / 4, self.map_height * 3 / 4, self.map_height * 3 / 4],
        ])

        self.anchor_points_2 = np.array([
            [self.map_width, self.map_width / 2, self.map_width / 2, self.map_width],
            [self.map_height / 4, self.map_height / 4, self.map_height * 3 / 4, self.map_height * 3 / 4],
        ])

    def set_velodyne_to_baselink(self):
        rospy.logwarn("velodyne to baselink from TF is tunned, current version fits best.")
        T = euler_matrix(0., 0.140, 0.)
        t = np.array([[2.64, 0, 1.98]]).T
        T[0:3, -1::] = t
        return T

    def pcd_callback(self, msg):
        """ Callback function for the point cloud dataset """
        rospy.logwarn("pcd data frame_id %s", msg.header.frame_id)
        rospy.logdebug("pcd data received")
        rospy.logdebug("pcd size: %d, %d", msg.height, msg.width)
        rospy.logwarn("pcd queue size: %d", len(self.pcd_queue))
        pcd = np.empty((4, msg.width))
        for i, el in enumerate(pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True)):
            pcd[:, i] = el
        self.pcd_queue.append(pcd)
        self.pcd_header_queue.append(msg.header)
        self.pcd_frame_id = msg.header.frame_id

    def update_pcd(self, target_stamp):
        """
        Find the closest point cloud wrt the target_stamp

        We first want to find the smallest stamp that is larger than the timestamp, then compare it with the
        largest stamp that is smaller than the timestamp. and then pick the smallest one as the result. If such
        condition does not exist, i.e. all the stamps are smaller than the time stamp, we just pick the latest one.

        Args:
            target_stamp:

        Returns:

        """
        for i in range(len(self.pcd_header_queue) - 1):
            if self.pcd_header_queue[i + 1].stamp > target_stamp:
                if self.pcd_header_queue[i].stamp < target_stamp:
                    diff_2 = self.pcd_header_queue[i + 1].stamp - target_stamp
                    diff_1 = target_stamp - self.pcd_header_queue[i].stamp
                    if diff_1 > diff_2:
                        header = self.pcd_header_queue[i + 1]
                        pcd = self.pcd_queue[i + 1]
                    else:
                        header = self.pcd_header_queue[i]
                        pcd = self.pcd_queue[i]
                    self.pcd_header_queue = self.pcd_header_queue[i::]
                    self.pcd_queue = self.pcd_queue[i::]
                    rospy.logdebug("Setting current pcd at: %d.%09ds", header.stamp.secs, header.stamp.nsecs)
                    return pcd, header.stamp
        header = self.pcd_header_queue[-1]
        pcd = self.pcd_queue[-1]
        self.pcd_header_queue = self.pcd_header_queue[-1::]
        self.pcd_queue = self.pcd_queue[-1::]
        rospy.logdebug("Setting current pcd at: %d.%09ds", header.stamp.secs, header.stamp.nsecs)
        return pcd, header.stamp

    def pose_callback(self, msg):
        rospy.logdebug("Getting pose at: %d.%09ds", msg.header.stamp.secs, msg.header.stamp.nsecs)
        self.pose_queue.append(msg)
        if msg.header.stamp.secs == 1581541270:  # 1581541437: TODO: explain the magic number here
            self.save_map_to_file = True
        rospy.logdebug("Pose queue length: %d", len(self.pose_queue))

    def set_global_map_pose(self):
        pose = Pose()
        pose.position.x = -1369.0496826171875  # TODO: These numbers need explanation
        pose.position.y = -562.84814453125
        pose.position.z = 0.0
        pose.orientation.w = 1.0
        set_map_pose(pose, '/world', 'global_map')

    def update_pose(self, target_stamp):
        """
        Find the closest pose wrt the target_stamp.

        This is the same implementation as the update_pcd().
        """
        for i in range(len(self.pose_queue) - 1):
            if self.pose_queue[i + 1].header.stamp > target_stamp:
                if self.pose_queue[i].header.stamp < target_stamp:
                    diff_2 = self.pose_queue[i + 1].header.stamp - target_stamp
                    diff_1 = target_stamp - self.pose_queue[i].header.stamp
                    if diff_1 > diff_2:
                        msg = self.pose_queue[i + 1]
                    else:
                        msg = self.pose_queue[i]
                    self.pose_queue = self.pose_queue[i::]
                    rospy.logdebug("Setting current pose at: %d.%09ds", msg.header.stamp.secs, msg.header.stamp.nsecs)
                    return msg.pose, msg.header.stamp
        msg = self.pose_queue[-1]
        self.pose_queue = self.pose_queue[-1::]
        rospy.logdebug("Setting current pose at: %d.%09ds", msg.header.stamp.secs, msg.header.stamp.nsecs)
        return msg.pose, msg.header.stamp

    def image_callback(self, msg):
        """
        The callback function for the camera image. When the semantic camera image is published, this function will be
        invoked and generate a BEV semantic map from the image.
        """
        rospy.logdebug("Mapping image at: %d.%09ds", msg.header.stamp.secs, msg.header.stamp.nsecs)
        try:
            image_in = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except CvBridgeError as e:
            print(e)
            return

        if msg.header.frame_id == "camera1":
            camera_calibration = self.cam1
        elif msg.header.frame_id == "camera6":
            camera_calibration = self.cam6
        else:
            rospy.logwarn("cannot find camera for frame_id %s", msg.header.frame_id)

        if self.depth_method in ['points_map', 'points_raw']:
            if len(self.pcd_header_queue) == 0: return
            self.pcd, self.pcd_time = self.update_pcd(msg.header.stamp)

        if len(self.pose_queue) == 0: return
        self.pose, self.pose_time = self.update_pose(msg.header.stamp)
        self.set_global_map_pose()

        self.mapping(image_in, self.pose, camera_calibration)

        rospy.logdebug("Finished Mapping image at: %d.%09ds", msg.header.stamp.secs, msg.header.stamp.nsecs)
        # rospy.signal_shutdown('Done with the mapping')

    def mapping(self, semantic_image, pose, camera_calibration):
        """
        Receives the semantic segmentation image, the pose of the vehicle, and the calibration of the camera,
        we will build a semantic point cloud, then project it into the 2D bird's eye view coordinates.

        Args:
            semantic_image: 2D semantic image
            pose: vehicle pose
            camera_calibration: The calibration information of the camera
        """
        # Initialize the map
        if self.map is None:
            num_class = len(self.label_names)
            self.map = np.ones((self.map_height, self.map_width, self.map_depth)) / num_class
            transform_matrix, trans, rot, euler = get_transformation(
                frame_from='/base_link', frame_to='/global_map',
                time_from=self.pose_time, time_to=rospy.Time(0),
                static_frame='/world', tf_listener=self.tf_listener, tf_ros=self.tf_ros,
            )
            self.position_rel = np.array([[trans[0], trans[1], trans[2]]]).T
            self.yaw_rel = euler[2]

        if self.depth_method in ['points_map', 'points_raw']:
            pcd_in_range, pcd_label = self.project_pcd(self.pcd, self.pcd_frame_id, semantic_image, pose,
                                                       camera_calibration)
            pcd_pub = create_point_cloud(pcd_in_range[0:3].T, pcd_label.T, frame_id=self.pcd_frame_id)
            self.pub_pcd.publish(pcd_pub)

            self.map = self.update_map(self.map, pcd_in_range, pcd_label)
        else:
            self.map = self.update_map_planar(self.map, semantic_image, camera_calibration)

        if self.save_map_to_file:
            color_map = color_map_local(self.map, self.label_names, self.label_colors)

            output_dir = osp.join(self.output_dir, "picture")
            os.makedirs(output_dir, exist_ok=True)

            output_file = osp.join(output_dir, "global_map.png")
            print("Saving image to", output_file)
            cv2.imwrite(output_file, color_map)

            # Publish the image
            try:
                image_pub = self.bridge.cv2_to_imgmsg(color_map, encoding="passthrough")
                self.pub_semantic_local_map.publish(image_pub)
            except CvBridgeError as e:
                print(e)

    def project_pcd(self, pcd, pcd_frame_id, image, pose, camera_calibration):
        """
        Extract labels of each point in the pcd from image

        Params:
            cam: camera calibration information, it includes the camera projection matrix.
        Return:
            point cloud that are visible in the image, and their associated labels
        """
        if pcd is None: return
        if pcd_frame_id != "velodyne":
            T_base_to_origin = get_transform_from_pose(pose)
            T_origin_to_velodyne = np.linalg.inv(np.matmul(T_base_to_origin, self.T_velodyne_to_basklink))

            pcd_velodyne = np.matmul(T_origin_to_velodyne, homogenize(pcd[0:3, :]))
        else:
            pcd_velodyne = homogenize(pcd[0:3, :])

        IXY = dehomogenize(np.matmul(camera_calibration.P, pcd_velodyne)).astype(np.int32)

        mask_positive = pcd_velodyne[0, :] > 0  # Only use the points in the front.
        mask = np.logical_and(np.logical_and(0 <= IXY[0, :], IXY[0, :] < image.shape[1]),
                              np.logical_and(0 <= IXY[1, :], IXY[1, :] < image.shape[0]))
        mask = np.logical_and(mask, mask_positive)

        # mask_intensity = pcd[3, :] > self.pcd_intensity_threshold
        # mask = np.logical_and(mask_intensity, mask)

        masked_pcd = pcd[:, mask]
        image_idx = IXY[:, mask]
        label = image[image_idx[1, :], image_idx[0, :]].T

        return masked_pcd, label

    def update_map(self, map, pcd, label):
        """
        Project the semantic point cloud on the BEV map

        TODO: This is the key part of our algorithm. Think about how to improve this.

        Args:
            map: np.ndarray with shape (H, W, C). H is the height, W is the width, and C is the semantic class.
            pcd: np.ndarray with shape (4, N). N is the number of points. The point cloud
            label: np.ndarray with shape (3, N). N is the number of points. The RGB label of each point cloud.

        Returns:
            Updated map
        """
        print("pcd limits", np.min(pcd[0]), np.max(pcd[0]), np.min(pcd[1]), np.max(pcd[1]))

        normal = np.array([[0.0, 0.0, 1.0]]).T  # The normal of the z axis
        T_pcd_to_local, _, _, _ = get_transformation(
            frame_from=self.pcd_frame_id, time_from=self.pcd_time,
            frame_to='/global_map', time_to=rospy.Time(0),
            static_frame='world', tf_listener=self.tf_listener, tf_ros=self.tf_ros,
        )
        pcd_local = np.matmul(T_pcd_to_local, homogenize(pcd[0:3]))[0:3, :]
        pcd_on_map = pcd_local - np.matmul(normal, np.matmul(normal.T, pcd_local))

        # Discretize point cloud into grid
        pcd_pixel = np.matmul(self.discretize_matrix, homogenize(pcd_on_map[0:2, :])).astype(np.int32)
        print("pcd_pixel limits", np.min(pcd_pixel[0]), np.max(pcd_pixel[0]),
              np.min(pcd_pixel[1]), np.max(pcd_pixel[1]))
        on_grid_mask = np.logical_and(np.logical_and(0 <= pcd_pixel[0, :], pcd_pixel[0, :] < self.map_width),
                                      np.logical_and(0 <= pcd_pixel[1, :], pcd_pixel[1, :] < self.map_height))

        # Update corresponding labels
        for i, label_name in enumerate(self.label_names):
            # Code explanation:
            # We first do an elementwise comparison
            # a = label == self.label_colors[i].reshape(3, 1)
            # Then we do a logical AND among the rows of a, represented by *a.
            idx = np.logical_and(*(label == self.label_colors[i].reshape(3, 1)))
            idx_mask = np.logical_and(idx, on_grid_mask)

            # Update the local map with Bayes update rule
            # y = pcd_pixel[1, idx_mask], x = pcd_pixel[0, idx_mask]
            map[pcd_pixel[1, idx_mask], pcd_pixel[0, idx_mask], :] *= self.confusion_matrix[i, :].reshape(1, -1)

            # # Intensity-aware
            # if self.use_pcd_intensity and label_name in ["crosswalk", "land"]:
            #     mask_intensity = pcd[3, :] > self.pcd_intensity_threshold  # mask of high intensity
            #     # map_local[pcd_pixel[1, mask_intensity], pcd_pixel[0, mask_intensity], i] += 1000 # only intensity
            #     mask_intensity_idx = np.logical_and(mask_intensity,
            #                                         idx_mask)  # mask of current label with high intensity
            #
            #     # TODO: tune this formula for extra odds
            #     # extra_odds added = (i-thred)
            #     extra_odds = pcd[3, mask_intensity_idx] - self.pcd_intensity_threshold
            #     # assign to current channel
            #     map[pcd_pixel[1, mask_intensity_idx], pcd_pixel[0, mask_intensity_idx], i] += 10

        return map

    def update_map_planar(self, map_local, image, cam):
        """ Project the semantic image onto the map plane and update it """

        points_map = homogenize(np.array(self.anchor_points_2))
        points_local = np.matmul(self.discretize_matrix_inv, points_map)
        points_local[2, :] = 0
        points_local = homogenize(points_local)

        T_local_to_base, _, _, _ = get_transformation(frame_from='/local_map', time_from=rospy.Time(0),
                                                      frame_to='/base_link', time_to=self.pose_time,
                                                      static_frame='world',
                                                      tf_listener=self.tf_listener, tf_ros=self.tf_ros)
        T_base_to_velodyne = np.linalg.inv(self.T_velodyne_to_basklink)
        T_local_to_velodyne = np.matmul(T_base_to_velodyne, T_local_to_base)

        # compute new points
        points_velodyne = np.matmul(T_local_to_velodyne, points_local)
        points_image = dehomogenize(np.matmul(cam.P, points_velodyne))

        # generate homography
        image_on_map = generate_homography(image, points_image.T, self.anchor_points_2.T, vis=False,
                                           out_size=[self.map_width, self.map_height])
        sep = int((8 - self.map_boundary[0][0]) / self.resolution)
        mask = np.ones(map_local.shape[0:2])
        mask[:, 0:sep] = 0
        idx_mask_3 = np.zeros([map_local.shape[0], map_local.shape[1], 3])

        for i in range(len(self.label_names)):
            idx = image_on_map[:, :, 0] == self.label_names[i]
            idx_mask = np.logical_and(idx, mask)
            map_local[idx_mask, i] += 1
            # idx_mask_3[idx_mask] = self.catogories_color[i]
        # cv2.imshow("mask", idx_mask_3.astype(np.uint8))
        # cv2.waitKey(1)

        map_local[map_local < 0] = 0

        # threshold and normalize
        # map_local[map_local > self.map_value_max] = self.map_value_max
        # print("max:", np.max(map_local))
        # normalized_map = self.normalize_map(map_local)

        return map_local

    def add_car_to_map(self, color_map):
        """ visualize ego car on the color map """
        # setting parameters
        length = 4.0
        width = 1.8
        mask_length = int(length / self.resolution)
        mask_width = int(width / self.resolution)
        car_center = np.array([[length / 4, width / 2]]).T / self.resolution
        discretize_matrix_inv = np.array([
            [self.resolution, 0, -length / 4],
            [0, -self.resolution, width / 2],
            [0, 0, 1]
        ])

        # pixels in ego frame
        Ix = np.tile(np.arange(0, mask_length), mask_width)
        Iy = np.repeat(np.arange(0, mask_width), mask_length)
        Ixy = np.vstack([Ix, Iy])

        # transform to map frame
        R = get_rotation_from_angle_2d(self.yaw_rel)
        Ixy_map = np.matmul(R, Ixy - car_center) + self.position_rel[0:2].reshape([2, 1]) / self.resolution + \
                  np.array([[-self.map_boundary[0][0] / self.resolution, self.map_height / 2]]).T
        Ixy_map = Ixy_map.astype(np.int)

        # setting color
        color_map[Ixy_map[1, :], Ixy_map[0, :], :] = [255, 0, 0]
        return color_map

    def get_extrinsics(self, pose, camera_id):
        T_base_to_origin = get_transform_from_pose(pose)

        # from camera to origin
        if camera_id == "camera1":
            T_cam_to_origin = np.matmul(T_base_to_origin, self.T_cam1_to_base)
        elif camera_id == "camera6":
            T_cam_to_origin = np.matmul(T_base_to_origin, self.T_cam6_to_base)
        else:
            rospy.logwarn("unable to find camera to base for camera_id %s", camera_id)
        T_origin_to_cam = np.linalg.inv(T_cam_to_origin)
        extrinsics = T_origin_to_cam[0:3]
        return extrinsics


# main
def main():
    rospy.init_node('semantic_mapping')

    cfg = get_cfg_defaults()
    sm = SemanticMapping(cfg)
    rospy.spin()


if __name__ == "__main__":
    main()
