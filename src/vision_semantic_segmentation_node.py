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

from camera import camera_setup_6
from network.deeplab_v3_plus.config.demo import cfg
from semantic_segmentation import SemanticSegmentation  # source code


# classes
class VisionSemanticSegmentationNode:

    def __init__(self):
        self.image_sub_cam1 = rospy.Subscriber("/camera1/image_raw", Image, self.image_callback)
        self.image_sub_cam6 = rospy.Subscriber("/camera6/image_raw", Image, self.image_callback)
        
        self.image_pub_cam1 = rospy.Publisher("/camera1/semantic_mask", Image, queue_size=1)
        self.image_pub_cam6 = rospy.Publisher("/camera6/semantic_mask", Image, queue_size=1)
        
        # Load the configuration
        # By default we are using the configuration config/avl.yaml
        config_file = osp.dirname(__file__) + '/../config/avl.yaml'
        cfg.merge_from_file(config_file)
        self.seg = SemanticSegmentation(cfg)
        
        self.bridge = CvBridge()


    def image_callback(self, msg):
        try:
            image_in = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except CvBridgeError as e:
            print(e)
        
        ## ========== Image preprocessing
        image_in = cv2.cvtColor(image_in, cv2.COLOR_BGR2RGB)
        
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
        image_out = np.empty((image_out_resized.shape[0], image_out_resized.shape[1], 3)).astype(np.uint8)
        for i in range(3):
            image_out[:,:,i] = image_out_resized

        try:
            image_pub = self.bridge.cv2_to_imgmsg(image_out, encoding="passthrough")
        except CvBridgeError as e:
            print(e)
        
        image_pub.header.frame_id = msg.header.frame_id

        if msg.header.frame_id == "camera1":
            self.image_pub_cam1.publish(image_pub)
        elif msg.header.frame_id == "camera6":
            self.image_pub_cam6.publish(image_pub)
        else:
            print("publisher not spepcify for this image.")


# main
def main(args):
    rospy.init_node('vision_semantic_segmentation')
    vss = VisionSemanticSegmentationNode()
    rate = rospy.Rate(15)  # ROS Rate at 15Hz, note that from rosbag, the image comes at 12Hz

    while not rospy.is_shutdown():
        rate.sleep()


if __name__ == '__main__':
    main(sys.argv)
