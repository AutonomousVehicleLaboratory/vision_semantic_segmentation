#!/usr/bin/env python
""" Semantic Segmentation Ros Wrapper

Author: Hengyuan Zhang
Date:February 14, 2020
"""

# module
from __future__ import absolute_import, division, print_function, unicode_literals  # python2 compatibility

import cv2
import os.path as osp
import rospy
import numpy as np
import sys

# Add network directory into the path
sys.path.insert(0, osp.join(osp.dirname(__file__), "network"))

import network.deeplab_v3_plus.data.utils.mapillary_visualization as mapillary_visl

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from shape_msgs.msg import Plane
from visualization_msgs.msg import Marker, MarkerArray

from camera import camera_setup_6, camera_setup_1
from network.deeplab_v3_plus.config.demo import cfg
from plane_3d import Plane3D
from semantic_convex_hull import generate_convex_hull
from semantic_segmentation import SemanticSegmentation  # source code
from vis import visualize_marker


# parameters


# classes
class VisionSemanticSegmentationNode:

    def __init__(self):
        self.image_sub_cam1 = rospy.Subscriber("/camera1/image_raw", Image, self.image_callback)
        self.image_sub_cam6 = rospy.Subscriber("/camera6/image_raw", Image, self.image_callback)
        self.plane_sub = rospy.Subscriber("/estimated_plane", Plane, self.plane_callback)

        self.image_pub_cam1 = rospy.Publisher("/camera1/semantic", Image, queue_size=1)
        self.image_pub_cam6 = rospy.Publisher("/camera6/semantic", Image, queue_size=1)
        self.pub_crosswalk_markers = rospy.Publisher("/crosswalk_convex_hull_rviz", MarkerArray, queue_size=10)
        self.pub_road_markers = rospy.Publisher("/road_convex_hull_rviz", MarkerArray, queue_size=10)
        
        # Load the configuration
        # By default we are using the configuration config/avl.yaml
        config_file = osp.dirname(__file__) + '/../config/avl.yaml'
        cfg.merge_from_file(config_file)
        self.seg = SemanticSegmentation(cfg)
        self.seg_color_fn = mapillary_visl.apply_color_map
        self.seg_color_ref = mapillary_visl.get_labels(cfg.DATASET_CONFIG)
        
        self.plane = None
        self.plane_last_update_time = rospy.get_rostime()
        self.cam6 = camera_setup_6()
        self.cam1 = camera_setup_1()

        self.hull_id = 0
        self.bridge = CvBridge()


    def image_callback(self, msg):
        rospy.logdebug("Segmentating image at: %d.%09ds", msg.header.stamp.secs, msg.header.stamp.nsecs)
        try:
            image_in = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except CvBridgeError as e:
            print(e)
        
        ## ========== Image preprocessing
        image_in = cv2.cvtColor(image_in, cv2.COLOR_BGR2RGB)
        if msg.header.frame_id == "camera1":
            image_in = cv2.undistort(image_in, self.cam1.K, self.cam1.dist)
        elif msg.header.frame_id == "camera6":
            image_in = cv2.undistort(image_in, self.cam6.K, self.cam6.dist)
        else:
            rospy.logwarn("unseen camera frame id %s, no undistortion performed.", msg.header.frame_id)
        
        # resize image
        scale_percent = 50  # percent of original size
        width = int(image_in.shape[1] * scale_percent / 100)
        height = int(image_in.shape[0] * scale_percent / 100)
        dim = (width, height)
        
        image_in_resized = cv2.resize(image_in, dim, interpolation=cv2.INTER_AREA)
        # print(image_in.shape, "-->", image_in_resized.shape)
        
        ## ========== semantic segmentation
        image_out_resized = self.seg.segmentation(image_in_resized)

        # print(image_out_resized.shape, "-->", image_in.shape)
        image_out_resized = image_out_resized.astype(np.uint8)

        ## ========== semantic extraction
        # self.generate_and_publish_convex_hull(image_out_resized, msg.header.frame_id, index_care_about=2) # cross walk
        # self.generate_and_publish_convex_hull(image_out_resized, msg.header.frame_id, index_care_about=1) # road
        
        # NOTE: we use INTER_NEAREST because values are discrete labels
        image_out = cv2.resize(image_out_resized, (image_in.shape[1], image_in.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
        
        ## ========== Visualize semantic images
        # Convert network label to color
        colored_output = self.seg_color_fn(image_out, self.seg_color_ref)
        colored_output = np.squeeze(colored_output)
        colored_output = colored_output.astype(np.uint8)

        try:
            image_pub = self.bridge.cv2_to_imgmsg(colored_output, encoding="passthrough")
        except CvBridgeError as e:
            print(e)
        rospy.logdebug("Publish Segmentated image at: %d.%09ds", msg.header.stamp.secs, msg.header.stamp.nsecs)
        
        image_pub.header.stamp = msg.header.stamp
        image_pub.header.frame_id = msg.header.frame_id
        if msg.header.frame_id == "camera1":
            self.image_pub_cam1.publish(image_pub)
        elif msg.header.frame_id == "camera6":
            self.image_pub_cam6.publish(image_pub)
        else:
            rospy.logwarn("publisher not spepcify for this camera frame_id %s.", msg.header.frame_id)


    def generate_and_publish_convex_hull(self, image, cam_frame_id, index_care_about=1):
        if cam_frame_id == "camera1":
            cam = self.cam1
        elif cam_frame_id == "camera6":
            cam = self.cam6
        
        vertice_list = generate_convex_hull(image, index_care_about=index_care_about, vis=False)
        
        # scale vertices to true position in original image (network output is small)
        scale_x = float(cam.imSize[0]) / image.shape[1]
        scale_y = float(cam.imSize[1]) / image.shape[0]
        for i in range(len(vertice_list)):
            vertice_list[i] = vertice_list[i] * np.array([[scale_x, scale_y]]).T
        
        self.cam_back_project_convex_hull(cam, vertice_list, index_care_about=index_care_about)
        

    def cam_back_project_convex_hull(self, cam, vertice_list, index_care_about=1):
        if len(vertice_list) == 0:
            print("vertice_list empty!")
            return

        current_time = rospy.get_rostime()
        duration = current_time - self.plane_last_update_time
        rospy.logdebug("duration: %d.%09d s", duration.secs, duration.nsecs)

        if duration.secs != 0 or duration.nsecs > 1e8:
            rospy.logwarn('too long since last update of plane %d.%09d s, please use smaller image', duration.secs, duration.nsecs)
        
        print("vertice_list non empty!, length ", len(vertice_list))

        vertices_marker_array = MarkerArray()
        for vertices in vertice_list:
            # print(vertices)
            x = vertices
            d_vec, C_vec = cam.pixel_to_ray_vec(x)
            intersection_vec = self.plane.plane_ray_intersection_vec(d_vec, C_vec)

            self.hull_id += 1
            
            if index_care_about == 1:
                color = [0.8, 0., 0., 0.8] # crosswalk is red
                vis_time = 10.0 # convex_hull marker alive time 
            else:
                color = [0.0, 0, 0.8, 0.8] # road is blue
                vis_time = 3.0
            marker = visualize_marker([0, 0, 0], 
                                      mkr_id=self.hull_id, 
                                      frame_id="velodyne", 
                                      mkr_type="line_strip", 
                                      scale=0.1,
                                      points=intersection_vec.T, 
                                      lifetime=vis_time, 
                                      mkr_color=color)
            vertices_marker_array.markers.append(marker)

        if index_care_about == 1:
            self.pub_crosswalk_markers.publish(vertices_marker_array)
        else:
            self.pub_road_markers.publish(vertices_marker_array)


    def plane_callback(self, msg):
        self.plane = Plane3D(msg.coef[0], msg.coef[1], msg.coef[2], msg.coef[3])
        self.plane_last_update_time = rospy.get_rostime()
        # print("plane received: ", self.plane.param.T)


# main
def main(args):
    rospy.init_node('vision_semantic_segmentation')
    vss = VisionSemanticSegmentationNode()
    rate = rospy.Rate(15)  # ROS Rate at 15Hz, note that from rosbag, the image comes at 12Hz

    while not rospy.is_shutdown():
        rate.sleep()


if __name__ == '__main__':
    main(sys.argv)
