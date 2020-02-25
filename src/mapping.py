""" Semantic mapping

Author: Henry Zhang
Date:February 24, 2020

"""

# module
import numpy as np
from homography import generate_homography
import rospy
from tf import Transformer, TransformListener, TransformBroadcaster, LookupException, ConnectivityException, ExtrapolationException, TransformerROS
from geometry_msgs.msg import PoseStamped, TransformStamped

# parameters


# classes
class SemanticMapping:
    def __init__(self, discretization = 0.1, boundary = [[0, 40], [-10, 10]]):
        self.sub_pose = rospy.Subscriber("/current_pose", PoseStamped, self.pose_callback)
        self.tf_listener_ = TransformListener()
        self.br = TransformBroadcaster()
        self.tfmr = Transformer()
        self.tf_ros = TransformerROS()

        self.x_max = boundary[0][1] - boundary[0][0]
        self.y_max = boundary[1][1] - boundary[1][0]
        self.map = np.zeros((self.x_max, self.y_max))
        self.map_pose = None
        self.pose = None
    
    def pose_callback(self, msg):
        self.pose = msg.pose
        if self.map_pose is None:
            self.map_pose = self.pose
            self.set_map_pose(self.pose)
        else:
            self.set_map_pose(self.map_pose)
            self.calculate_transformation(msg)

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
    
    def calculate_transformation(self, pose):
        try:
            transform = self.tf_listener_.lookupTransform('/local_map', '/world', rospy.Time(0))
        except (LookupException, ConnectivityException, ExtrapolationException):
            print("exception")
        pose.pose.orientation.w = 1.0    # Neutral orientation
        tf_pose = self.tf_listener_.transformPose("/local_map", pose)
        print "Position of the pose in the local map:"
        
        print tf_pose

    # def mapping(self, im_src, pose):
    #     self.pose = pose
    #     if self.require_new_map(pose):
    #         self.map = self.transform_old_map( self.pose, pose)

    #     """ Take in image, add semantic information to the local map """
    #     im_dst = self.transform_mask(im_src, pose)
    #     updated_map = self.update_map(im_dst)
    #     self.map = updated_map

    # def require_new_map(self, pose):
    #     if np.abs(pose.position.x - self.map_pose.position.x) > 10 or 
    #        np.abs(pose.position.y - self.map_pose.position.x)

    # def transform_old_map(self, old_pose, new_pose):
    #     pass
    
    # def transform_mask(self, im_src, pose):
    #     # prepare

    #     pts_src = None
    #     pts_dst = None

    #     # transform
    #     im_dst = generate_homography(im_src, pts_src, pts_dst)
    #     return im_dst

    # def update_map(self, im_dst):
    #     log_odds_map = self.update_log_odds(im_dst)
    #     return log_odds_map
    
    # def update_log_odds(self, im_dst):
    #     pass

    # def binarize(self, map_in):
    #     return map_in
# functions


# main
def main():
    rospy.init_node('semantic_mapping')
    sm = SemanticMapping()
    rospy.spin()


if __name__ == "__main__":
    main()