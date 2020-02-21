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
from shape_msgs.msg import Plane
from visualization_msgs.msg import Marker

from semantic_segmentation import SemanticSegmentation # source code
from vis import visualize_marker
from plane_3d import Plane3D
from camera import camera_setup_6
from semantic_convex_hull import generate_convex_hull


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
        self.plane_sub = rospy.Subscriber("/estimated_plane", Plane, self.plane_callback)
        self.pub_convex_hull_markers = rospy.Publisher("/estimated_convex_hull_rviz", Marker, queue_size=10)
        self.plane = None
        self.cam6 = camera_setup_6()
        self.cam1 = camera_setup_6()
        print("WARNING: using camera 6 data for camera 1 since it is not calibrated!") 

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
        
	self.generate_and_publish_convex_hull(image_out_resized, msg.header.frame_id)

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

    def generate_and_publish_convex_hull(self, image, cam_frame_id):
        if cam_frame_id == "camera1":
            cam = self.cam1
        elif cam_frame_id == "camera6":
            cam = self.cam6
        img = cv2.imread('/mnt/avl_shared/user-files/henry/pylidarmot/src/vision_semantic_segmentation/network_output_example/preds/3118.jpg')
        print("using hardcoded data")
        vertices = generate_convex_hull(img, vis=False)
        print("vertices:\n", vertices)
        self.cam_back_project_convex_hull(cam, vertices)

    def cam_back_project_convex_hull(self, cam, vertices):
        if self.plane is None:
            print("not received plane estimation parameters yet")
            return
        # x = vertices
        x = np.array([[400, 300, 1100, 1000, 400],
                      [1150, 1200, 1200, 1150, 1150]])
    
        d_vec, C_vec = cam.pixel_to_ray_vec(x)
        intersection_vec = self.plane.plane_ray_intersection_vec(d_vec, C_vec)
        marker = visualize_marker([0,0,0], frame_id="velodyne", mkr_type="line_strip", scale=0.1, points=intersection_vec.T)
        self.pub_convex_hull_markers.publish(marker)

    def plane_callback(self, msg):
        self.plane = Plane3D(msg.coef[0], msg.coef[1], msg.coef[2], msg.coef[3])
        print("plane received: ", self.plane.param)


# main
def main(args):
    vss = VisionSemanticSegmentationNode()
    rospy.init_node('vision_semantic_segmentation')
    rate = rospy.Rate(15) # ROS Rate at 15Hz, note that from rosbag, the image comes at 12Hz
    
    while not rospy.is_shutdown():
        rate.sleep()

if __name__ == '__main__':
    main(sys.argv)
