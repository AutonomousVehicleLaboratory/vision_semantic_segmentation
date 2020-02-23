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
from visualization_msgs.msg import Marker

from camera import camera_setup_6
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
        self.pub_convex_hull_markers = rospy.Publisher("/estimated_convex_hull_rviz", Marker, queue_size=10)
        
        # Load the configuration
        # By default we are using the configuration config/avl.yaml
        config_file = osp.dirname(__file__) + '/../config/avl.yaml'
        cfg.merge_from_file(config_file)
        self.seg = SemanticSegmentation(cfg)
        self.seg_color_fn = mapillary_visl.apply_color_map
        self.seg_color_ref = mapillary_visl.get_labels(cfg.DATASET_CONFIG)
        
        self.plane = None
        self.cam6 = camera_setup_6()
        self.cam1 = camera_setup_6()
        print("WARNING: using camera 6 data for camera 1 since it is not calibrated!")
        
        self.bridge = CvBridge()


    def image_callback(self, msg):
        try:
            image_in = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except CvBridgeError as e:
            print(e)
        
        ## Image preprocessing
        # Is the image in BGR or RGB format?
        image_in = cv2.cvtColor(image_in, cv2.COLOR_BGR2RGB)
        # cv2.imshow("Image Input", image_in)
        
        # resize image
        scale_percent = 50  # percent of original size
        width = int(image_in.shape[1] * scale_percent / 100)
        height = int(image_in.shape[0] * scale_percent / 100)
        dim = (width, height)
        
        image_in_resized = cv2.resize(image_in, dim, interpolation=cv2.INTER_AREA)
        # print(image_in.shape, "-->", image_in_resized.shape)
        
        ## semantic segmentation
        image_out_resized = self.seg.segmentation(image_in_resized)
        
        # print(image_out_resized.shape, "-->", image_in.shape)
        image_out_resized = image_out_resized.astype(np.uint8)

        ## semantic extraction
        self.generate_and_publish_convex_hull(image_out_resized, msg.header.frame_id)

        # NOTE: we use INTER_NEAREST because values are discrete labels
        image_out = cv2.resize(image_out_resized, (image_in.shape[1], image_in.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
        
        ## Visualize semantic images
        # Convert network label to color
        colored_output = self.seg_color_fn(image_out, self.seg_color_ref)
        colored_output = np.squeeze(colored_output)
        colored_output = colored_output.astype(np.uint8)

        try:
            image_pub = self.bridge.cv2_to_imgmsg(colored_output, encoding="passthrough")
        except CvBridgeError as e:
            print(e)

        if msg.header.frame_id == "camera1":
            self.image_pub_cam1.publish(image_pub)
        elif msg.header.frame_id == "camera6":
            self.image_pub_cam6.publish(image_pub)
        else:
            print("publisher not spepcify for this image.")


    def generate_and_publish_convex_hull(self, image, cam_frame_id):
        if cam_frame_id == "camera1":
            cam = self.cam1
        elif cam_frame_id == "camera6":
            cam = self.cam6
        print("using hardcoded data")
        vertices = generate_convex_hull(image, vis=False)
        print("vertices:\n", vertices)
        
        self.cam_back_project_convex_hull(cam, vertices)


    def cam_back_project_convex_hull(self, cam, vertices):
        if self.plane is None:
            print("not received plane estimation parameters yet")
            return
        elif vertices is None:
            print("not received vertices")
            return
        
        print("vertices received!")
        print(vertices)
        x = vertices
        # x = np.array([[400, 300, 1100, 1000, 400],
        #               [1150, 1200, 1200, 1150, 1150]])

        d_vec, C_vec = cam.pixel_to_ray_vec(x)
        intersection_vec = self.plane.plane_ray_intersection_vec(d_vec, C_vec)
        marker = visualize_marker([0, 0, 0], frame_id="velodyne", mkr_type="line_strip", scale=0.1,
                                  points=intersection_vec.T)
        self.pub_convex_hull_markers.publish(marker)


    def plane_callback(self, msg):
        self.plane = Plane3D(msg.coef[0], msg.coef[1], msg.coef[2], msg.coef[3])
        # print("plane received: ", self.plane.param.T)


# main
def main(args):
    vss = VisionSemanticSegmentationNode()
    rospy.init_node('vision_semantic_segmentation')
    rate = rospy.Rate(15)  # ROS Rate at 15Hz, note that from rosbag, the image comes at 12Hz

    while not rospy.is_shutdown():
        rate.sleep()


if __name__ == '__main__':
    main(sys.argv)
