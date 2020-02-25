""" Semantic mapping

Author: Henry Zhang
Date:February 24, 2020

"""

# module
import numpy as np
from homography import generate_homography

# parameters


# classes
class SemanticMapping:
    def __init__(self):
        pass
    
    def mapping(self, im_src):
        """ Take in image, add semantic information to the local map """
        im_dst = self.transorm_mask(im_src)
        updated_map = self.updata_map(im_dst)
    
    def transorm_mask(self, im_src):
        # prepare
        pts_src = None
        pts_dst = None

        # transform
        im_dst = generate_homography(im_src, pts_src, pts_dst)
        return im_dst

    def update_map(self, im_dst):
        log_odds_map = self.updata_log_odds(im_dst)
        processed_map = self.post_processing(log_odds_map)
    
    def post_processing(self, log_odds_map):
        clipped_map = self.clip(log_odds_map)
        binary_map = self.binarize(clipped_map)
        self.publish_clipped_map(clipped_map)
        self.publish_binary_map(binary_map)
        return clipped_map
    
    def updata_log_odds(im_dst):
        pass
# functions


# main
def main():
    pass

if __name__ == "__main__":
    main()