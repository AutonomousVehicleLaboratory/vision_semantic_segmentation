#!/usr/bin/env python
""" Semantic mapping

Author: Henry Zhang
Date:February 24, 2020

"""

# module
import numpy as np
import cv2

import rospy
from tf import Transformer, TransformListener, TransformerROS
from tf.transformations import quaternion_matrix, euler_from_quaternion, euler_matrix

from geometry_msgs.msg import PoseStamped, Transform, Pose, Quaternion
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs import point_cloud2 as pc2


from camera import camera_setup_6
from utils import homogenize, dehomogenize, get_rotation_from_angle_2d
from utils_ros import set_map_pose, get_transformation, get_transform_from_pose, create_point_cloud
from homography import generate_homography
import time
# parameters


# classes
class SemanticMapping:
    def __init__(self, discretization = 0.1, boundary = [[-20, 50], [-10, 10]]):
        self.sub_pose = rospy.Subscriber("/current_pose", PoseStamped, self.pose_callback)
        self.sub_pcd = rospy.Subscriber("/reduced_map", PointCloud2, self.pcd_callback)
        self.image_sub_cam6 = rospy.Subscriber("/camera6/semantic", Image, self.image_callback)

        self.pub_semantic_local_map = rospy.Publisher("/semantic_local_map", Image, queue_size=5)
        self.pub_pcd = rospy.Publisher("/semantic_point_cloud", PointCloud2, queue_size=5)

        self.tf_listener = TransformListener()
        self.tfmr = Transformer()
        self.tf_ros = TransformerROS()
        self.bridge = CvBridge()
        
        self.pose = None
        self.pose_queue = []
        self.cam6 = camera_setup_6()
        rospy.logwarn("currently only setup for camera6")
        rospy.logwarn("currently only for front view")
        self.pcd = None
        
        self.map = None
        self.map_pose = None
        self.map_boundary = boundary
        self.d = discretization # discretization in meters
        self.map_value_max = 10 # prevent over confidence (deprecated)
        self.map_decay = 4 # prevent over confidence
        self.catogories = [128, 140, 255, 107, 244] # values of labels in the iuput images of mapping
        self.catogories_color = np.array([
            [128, 64, 128], # road
            [140, 140, 200], # crosswalk
            [255, 255, 255], # lane
            [107, 142, 35], # vegetation
            [244, 35, 232] # sizewalk
        ])
        self.map_width = int((boundary[0][1] - boundary[0][0]) / self.d)
        self.map_height = int((boundary[1][1] - boundary[1][0]) / self.d)
        self.map_depth = len(self.catogories)

        self.position_rel = np.array([[0,0,0]]).T
        self.yaw_rel = 0
        
        self.preprocessing()


    def preprocessing(self):
        """ setup constant matrices """
        self.T_velodyne_to_basklink = self.set_velodyne_to_baselink()
        self.T_cam_to_base = np.matmul(self.T_velodyne_to_basklink, self.cam6.T)
        
        self.discretize_matrix_inv = np.array([
            [self.d, 0, self.map_boundary[0][0]],
            [0, -self.d, self.map_boundary[1][1]],
            [0, 0, 1]
        ]).astype(np.float)
        self.discretize_matrix = np.linalg.inv(self.discretize_matrix_inv)
        
        self.anchor_points = np.array([
            [self.map_width, self.map_width / 3, self.map_width, self.map_width / 3],
            [self.map_height/4, self.map_height/4, self.map_height * 3 / 4, self.map_height * 3 / 4]
        ])


    def set_velodyne_to_baselink(self):
        rospy.logwarn("velodyne to baselink from TF is different from this, which is correct?")
        T = euler_matrix(0., 0.157, 0.)
        t = np.array([[2.64, 0, 1.98]]).T
        T[0:3,-1::] = t
        return T
    

    def pcd_callback(self, msg):
        rospy.logdebug("pcd data received")
        rospy.logdebug("pcd size: %d, %d", msg.height, msg.width)
        self.pcd = np.empty((3, msg.width))
        for i, el in enumerate( pc2.read_points(msg, field_names = ("x", "y", "z"), skip_nans=True)):
            self.pcd[:,i] = el
        self.pcd = homogenize(self.pcd)


    def pose_callback(self, msg):
        rospy.loginfo("Getting pose at: %d.%09ds", msg.header.stamp.secs, msg.header.stamp.nsecs)
        self.pose_queue.append(msg)
        rospy.loginfo("Pose queue length: %d", len(self.pose_queue))
        # T, trans, rot, euler = get_transformation(frame_from='/velodyne', frame_to='/base_link')
    
    def update_map_pose(self):
        if self.map_pose is None:
            self.map_pose = self.pose
            set_map_pose(self.pose, '/world', '/local_map')
        else:
            set_map_pose(self.map_pose, '/world', '/local_map')
            get_transformation(tf_listener=self.tf_listener, tf_ros=self.tf_ros)

    def update_pose(self, stamp):
        """ update self.pose with the closest one in the queue """
        for i in range(len(self.pose_queue)-1):
            if self.pose_queue[i+1].header.stamp > stamp:
                if self.pose_queue[i].header.stamp < stamp:
                    diff_2 = self.pose_queue[i+1].header.stamp - stamp
                    diff_1 = stamp - self.pose_queue[i].header.stamp
                    if diff_1 > diff_2:
                        msg = self.pose_queue[i+1]
                    else:
                        msg = self.pose_queue[i]
                    self.pose_queue = self.pose_queue[i::]
                    rospy.loginfo("Setting current pose at: %d.%09ds", msg.header.stamp.secs, msg.header.stamp.nsecs)
                    return msg.pose
        msg = self.pose_queue[-1]
        rospy.loginfo("Setting current pose at: %d.%09ds", msg.header.stamp.secs, msg.header.stamp.nsecs)
        return msg.pose

        
    def image_callback(self, msg):
        rospy.loginfo("Mapping image at: %d.%09ds", msg.header.stamp.secs, msg.header.stamp.nsecs)
        try:
            image_in = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except CvBridgeError as e:
            print(e)   
        
        ## =========== Mapping
        if self.pcd is None or len(self.pose_queue) == 0:
            return
        
        self.pose = self.update_pose(msg.header.stamp)
        self.update_map_pose()

        color_map = self.mapping(image_in, self.pose)

        try:
            image_pub = self.bridge.cv2_to_imgmsg(color_map, encoding="passthrough")
        except CvBridgeError as e:
            print(e)
        
        # map is independent of camera
        self.pub_semantic_local_map.publish(image_pub)
        

    def mapping(self, im_src, pose):
        # tic = time.time()
        if self.require_new_map(pose):
            pose_old = self.map_pose
            map_new = self.create_new_local_map(pose)
            self.map = self.transform_old_map(map_new, self.map, pose_old, pose)
            self.map_pose = pose
            set_map_pose(self.pose, '/world', '/local_map')
        
        pcd_in_range, pcd_label = self.project_pcd(self.pcd, im_src, pose)
        updated_map = self.update_map(self.map, pcd_in_range, pcd_label)

        # generate color map
        color_map = self.color_map(updated_map)
        color_map_with_car = self.add_car_to_map(color_map)

        self.map = updated_map
        # toc2 = time.time()
        # rospy.loginfo("time: %f s", toc2 - tic)

        pcd_pub = create_point_cloud(dehomogenize(pcd_in_range).T, pcd_label.T, frame_id='world')
        self.pub_pcd.publish(pcd_pub)

        return color_map
    

    def project_pcd(self, pcd, image, pose):
        """ extract labels of each point in the pcd from image 
        
        Params:
            P_norm: camera extrinsics
        Return:
            labels
        """
        if pcd is None:
            return
        if pcd.shape[0] == 3:
            pcd = homogenize(pcd)

        pcd = pcd[:,pcd[0,:]!=0]
        
        T_base_to_origin = get_transform_from_pose(pose)
        T_origin_to_velodyne = np.linalg.inv(np.matmul(T_base_to_origin, self.T_velodyne_to_basklink))

        pcd_velody = np.matmul(T_origin_to_velodyne, pcd)
        IXY = dehomogenize( np.matmul(self.cam6.P, pcd_velody)).astype(np.int32)

        mask_positive = pcd_velody[0, :] > 0
        mask = np.logical_and( np.logical_and( 0 <= IXY[0,:], IXY[0,:] < image.shape[1]), 
                               np.logical_and( 0 <= IXY[1,:], IXY[1,:] < image.shape[0]))
        mask = np.logical_and(mask, mask_positive) # enforce only use points in the front

        masked_pcd = pcd[:,mask]
        image_idx = IXY[:,mask]
        label = image[image_idx[1,:],image_idx[0,:]].T
        
        return masked_pcd, label


    def require_new_map(self, pose):
        transform_matrix, trans, rot, euler = get_transformation(tf_listener=self.tf_listener, tf_ros=self.tf_ros)
        self.position_rel = np.array([[trans[0], trans[1], trans[2]]]).T
        self.yaw_rel = euler[2]
        if trans is None or self.map is None or np.abs(trans[0]) > 10 or np.abs(trans[1]) > 2 or np.linalg.norm(euler) > 0.1:
            flag = True
        else:
            flag = False
        rospy.logdebug("%s", "True" if flag else "False")
        return flag


    def create_new_local_map(self, pose):
        map_new = np.zeros((self.map_height , self.map_width, self.map_depth))
        self.position_rel = np.array([[0, 0, 0]]).T
        self.yaw_rel = 0
        return map_new


    def transform_old_map(self, map_new, map_old, pose_old, pose_new):
        if map_old is None:
            return map_new
        else:
            points_old_map = homogenize(np.array(self.anchor_points))
            points_old_local = np.matmul(self.discretize_matrix_inv, points_old_map)
            points_old_local[2,:] = 0
            points_old_local = homogenize(points_old_local)

            # compute transformation (deprecated method, use tf istead for accuracy)
            T_old_map_to_world = get_transform_from_pose(pose_old)
            T_new_map_to_world = get_transform_from_pose(pose_new)
            T_old_map_to_new_map = np.matmul(T_old_map_to_world, np.linalg.inv(T_new_map_to_world))

            mat, _, _, _ = get_transformation(frame_from='/local_map', 
                                              frame_to='/base_link',
                                              tf_listener=self.tf_listener,
                                              tf_ros=self.tf_ros )

            if mat is not None:
                T_old_map_to_new_map = mat

            # compute new points
            points_new_local = np.matmul(T_old_map_to_new_map, points_old_local)
            points_new_local = points_new_local[0:2]
            points_new_local = homogenize(points_new_local)
            points_new_map = np.matmul(self.discretize_matrix, points_new_local)[0:2]

            # generate homography
            map_old_transformed = generate_homography(map_old, points_old_map[0:2].T, points_new_map.T, vis=False)
            
            # decay factor, make the map not as over confident
            map_old_transformed = map_old_transformed / self.map_decay
            return map_old_transformed


    def update_map(self, map_local, pcd, label):
        """ project the pcd on the map """

        normal = np.array([[0.0 ,0.0 ,1.0]]).T
        T_local_to_world = get_transform_from_pose(self.map_pose)
        T_world_to_local = np.linalg.inv(T_local_to_world)
        pcd_local = np.matmul(T_world_to_local, pcd)[0:3,:]
        pcd_on_map = pcd_local - np.matmul(normal, np.matmul(normal.T, pcd_local))

        # discretize
        pcd_pixel = np.matmul(self.discretize_matrix, homogenize(pcd_on_map[0:2,:])).astype(np.int32)
        mask = np.logical_and( np.logical_and(0 <= pcd_pixel[0,:], pcd_pixel[0,:] < self.map_width ),
                               np.logical_and(0 <= pcd_pixel[1,:], pcd_pixel[1,:] < self.map_height ))
        
        # update corresponding labels
        for i in range(len(self.catogories)):
            idx = label[0,:] == self.catogories[i]
            idx_mask = np.logical_and(idx, mask)
            map_local[pcd_pixel[1, idx_mask], pcd_pixel[0, idx_mask], i] += 2
            map_local[pcd_pixel[1, idx_mask], pcd_pixel[0, idx_mask], :] -= 1
        
        map_local[map_local < 0] = 0

        # threshold and normalize
        # map_local[map_local > self.map_value_max] = self.map_value_max
        # print("max:", np.max(map_local))
        # normalized_map = self.normalize_map(map_local)

        return map_local


    def normalize_map(self, map_local):
        normalized_map = np.array(map_local)
        normalized_map = normalized_map * 255 / self.map_value_max
        normalized_map = normalized_map.astype(np.uint8)
        return normalized_map


    def color_map(self, map_local):
        """ color the map by which label has max number of points """
        color_map = np.zeros((self.map_height, self.map_width, 3)).astype(np.uint8)
        
        map_sum = np.sum(map_local, axis=2) # get all zero mask
        map_argmax = np.argmax(map_local, axis=2)
        
        for i in range(len(self.catogories)):
            color_map[map_argmax == i] = self.catogories_color[i]
        
        color_map[map_sum == 0] = [0,0,0] # recover all zero positions
        
        return color_map
    
    
    def add_car_to_map(self, color_map):
        """ visualize ego car on the color map """
        # setting parameters
        length = 4.0
        width = 1.8
        mask_length = int(length / self.d)
        mask_width = int(width / self.d)
        discretize_matrix_inv = np.array([
            [self.d, 0, -length/4],
            [0, -self.d, width/2],
            [0, 0, 1]
        ])

        # pixels in ego frame
        Ix = np.tile(np.arange(0, mask_length), mask_width)
        Iy = np.repeat(np.arange(0, mask_width), mask_length)
        Ixy = np.vstack([Ix, Iy])
        
        # transform to map frame
        R = get_rotation_from_angle_2d(self.yaw_rel)
        Ixy_map = np.matmul(R, Ixy) + self.position_rel[0:2].reshape([2,1]) / self.d + \
                        np.array([[-self.map_boundary[0][0]/self.d, self.map_height / 2]]).T
        Ixy_map = Ixy_map.astype(np.int)
        
        # setting color
        color_map[Ixy_map[1,:], Ixy_map[0,:],:] = [255, 0, 0]
        return color_map


    def get_extrinsics(self, pose):
        T_base_to_origin = get_transform_from_pose(pose)

        # from camera to origin
        T_cam_to_origin = np.matmul(T_base_to_origin, self.T_cam_to_base)
        T_origin_to_cam = np.linalg.inv(T_cam_to_origin)
        extrinsics = T_origin_to_cam[0:3]
        return extrinsics


# main
def main():
    rospy.init_node('semantic_mapping')
    sm = SemanticMapping()
    rospy.spin()


if __name__ == "__main__":
    main()