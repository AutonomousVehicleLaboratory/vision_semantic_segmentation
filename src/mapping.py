#!/usr/bin/env python
""" Semantic mapping

Author: Henry Zhang
Date:February 24, 2020

"""

# module
import numpy as np
import cv2

import rospy
from tf import Transformer, TransformListener, TransformBroadcaster, LookupException, ConnectivityException, ExtrapolationException, TransformerROS
from tf.transformations import quaternion_matrix, euler_from_quaternion
from geometry_msgs.msg import PoseStamped, TransformStamped
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs import point_cloud2 as pc2

from camera import camera_setup_6
from homography import generate_homography
# parameters


# classes
class SemanticMapping:
    def __init__(self, discretization = 0.1, boundary = [[0, 40], [-10, 10]]):
        self.sub_pose = rospy.Subscriber("/current_pose", PoseStamped, self.pose_callback)
        self.sub_pcd = rospy.Subscriber("/reduced_map", PointCloud2, self.pcd_callback)
        self.image_sub_cam6 = rospy.Subscriber("/camera6/image_raw", Image, self.image_callback)

        self.tf_listener_ = TransformListener()
        self.br = TransformBroadcaster()
        self.tfmr = Transformer()
        self.tf_ros = TransformerROS()

        self.x_max = boundary[0][1] - boundary[0][0]
        self.y_max = boundary[1][1] - boundary[1][0]
        self.map = np.zeros((self.x_max, self.y_max))
        self.map_pose = None
        self.pose = None
        self.cam6 = camera_setup_6()
        self.bridge = CvBridge()
        self.pcd = None
    
    def pcd_callback(self, msg):
        rospy.logdebug("pcd data received")
        rospy.logdebug("pcd size: %d, %d", msg.height, msg.width)
        self.pcd = np.empty((msg.width,3))
        for i, el in enumerate( pc2.read_points(msg, field_names = ("x", "y", "z"), skip_nans=True)):
            self.pcd[i,:] = el

    def pose_callback(self, msg):
        self.pose = msg.pose
        if self.map_pose is None:
            self.map_pose = self.pose
            self.set_map_pose(self.pose)
        else:
            self.set_map_pose(self.map_pose)
            self.get_local_transformation()
            

    def set_map_pose(self, pose):
        m = TransformStamped()
        m.header.frame_id = "world"
        m.header.stamp = rospy.Time.now()
        m.child_frame_id = "local_map"
        m.transform.translation.x = pose.position.x
        m.transform.translation.y = pose.position.y
        m.transform.translation.z = pose.position.z
        m.transform.rotation.x = pose.orientation.x
        m.transform.rotation.y = pose.orientation.y
        m.transform.rotation.z = pose.orientation.z
        m.transform.rotation.w = pose.orientation.w
        self.br.sendTransformMessage(m)
    
    def get_local_transformation(self):
        try:
            (trans, rot) = self.tf_listener_.lookupTransform('/local_map', '/base_link', rospy.Time(0))
        except (LookupException, ConnectivityException, ExtrapolationException):
            print("exception")
        # pose.pose.orientation.w = 1.0    # Neutral orientation
        # tf_pose = self.tf_listener_.transformPose("/world", pose)
        # R_local = quaternion_matrix(tf_pose.pose.orientation)
        T = self.tf_ros.fromTranslationRotation(trans, rot)
        euler = euler_from_quaternion(rot)
        
        # print "Position of the pose in the local map:"
        # print trans, euler
        return T, trans, rot, euler
        
    def image_callback(self, msg):
        try:
            image_in = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except CvBridgeError as e:
            print(e)
        
        ## ========== Image preprocessing
        image_in = cv2.cvtColor(image_in, cv2.COLOR_BGR2RGB)
        self.mapping(image_in, self.pose)

    def mapping(self, im_src, pose):
        self.pose = pose
        flag = self.require_new_map(pose)
        rospy.logdebug("%s", "True" if flag else "False")
        if flag:
            self.map = self.transform_old_map( self.map_pose, pose)
            self.map_pose = pose
            self.set_map_pose(self.pose)

        """ Take in image, add semantic information to the local map """
        im_dst = self.transform_mask(im_src, pose)
        updated_map = self.update_map(im_dst)
        self.map = updated_map

    def require_new_map(self, pose):
        transform_matrix, trans, rot, euler = self.get_local_transformation()
        if np.abs(trans[0]) > 10 or np.abs(trans[1]) > 2:
            return True
        else:
            return False

    def transform_old_map(self, old_pose, new_pose):
        pass
    
    def transform_mask(self, im_src, pose):
        """ retrive map mask from current image """
        # prepare

        im_dst = im_src
        # transform
        # homography approach might be not suitable for non planar approach
        # im_dst = generate_homography(im_src, pts_src, pts_dst)
        return im_dst

    def update_map(self, im_dst):
        log_odds_map = self.update_log_odds(im_dst)
        return log_odds_map
    
    def update_log_odds(self, im_dst):
        pass

    def binarize(self, map_in):
        return map_in
# functions


# main
def main():
    rospy.init_node('semantic_mapping')
    sm = SemanticMapping()
    rospy.spin()


if __name__ == "__main__":
    main()