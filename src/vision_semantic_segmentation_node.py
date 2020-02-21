#!/usr/bin/env python
""" Semantic Segmentation Ros Wrapper

Author: Hengyuan Zhang
Date:February 14, 2020
"""

# module
from __future__ import absolute_import, division, print_function, unicode_literals # python2 compatibility
import sys

import cv2
import os.path as osp
import rospy
import numpy as np

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from semantic_segmentation import SemanticSegmentation # source code

# parameters


# classes
class VisionSemanticSegmentationNode:

    def __init__(self):
        self.image_pub_cam1 = rospy.Publisher("/camera1/semantic",Image, queue_size=1)
        self.image_pub_cam6 = rospy.Publisher("/camera6/semantic",Image, queue_size=1)

        self.bridge = CvBridge()
        # By default we are using the configuration config/avl.yaml
        self.seg = SemanticSegmentation(config_file=osp.dirname(__file__) + '/../config/avl.yaml')

        self.image_sub_cam1 = rospy.Subscriber("/camera1/image_raw",Image,self.callback)
        self.image_sub_cam6 = rospy.Subscriber("/camera6/image_raw",Image,self.callback)
        

    def callback(self, msg):
        try:
            image_in = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except CvBridgeError as e:
            print(e)

        scale_percent = 25 # percent of original size
        width = int(image_in.shape[1] * scale_percent / 100)
        height = int(image_in.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        image_in_resized = cv2.resize(image_in, dim, interpolation = cv2.INTER_AREA)
        
        # print(image_in.shape, "-->", image_in_resized.shape)
        image_out_resized = self.seg.segmentation(image_in_resized)
        # print(image_out_resized.shape, "-->", image_in.shape)
        
        # NOTE: we use INTER_NEAREST because values are discrete labels
        image_out = cv2.resize(image_out_resized, (image_in.shape[1], image_in.shape[0]) , interpolation = cv2.INTER_NEAREST)


        try:
            image_out = np.stack((image_out, image_out, image_out), axis=2)
            image_out = image_out.astype(np.uint8)
            image_pub = self.bridge.cv2_to_imgmsg(image_out, encoding="passthrough")
        except CvBridgeError as e:
            print(e)

        
        if msg.header.frame_id == "camera1":
            self.image_pub_cam1.publish(image_pub)
        elif msg.header.frame_id == "camera6":
            self.image_pub_cam6.publish(image_pub)
        else:
            print("publisher not spepcify for this image.")

# main
def main(args):
    vss = VisionSemanticSegmentationNode()
    rospy.init_node('vision_semantic_segmentation')
    rate = rospy.Rate(15) # ROS Rate at 15Hz, note that from rosbag, the image comes at 12Hz
    
    while not rospy.is_shutdown():
        rate.sleep()

if __name__ == '__main__':
    main(sys.argv)
