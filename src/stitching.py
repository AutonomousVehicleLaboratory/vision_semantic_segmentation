""" Stitch image

Author: 
Date:February 29, 2020
"""

# module
import pickle
import cv2
import numpy as np
from utils import homogenize, dehomogenize
# parameters


# classes


# functions

def color_map(map_local):
    """ color the map by which label has max number of points """
    d = 0.1
    boundary = [[-20, 50], [-10, 10]]
    catogories = [128, 140, 255, 107, 244] # values of labels in the iuput images of mapping
    catogories_color = np.array([
        [128, 64, 128], # road
        [140, 140, 200], # crosswalk
        [255, 255, 255], # lane
        [107, 142, 35], # vegetation
        [244, 35, 232] # sidewalk
    ])
    map_width = int((boundary[0][1] - boundary[0][0]) / d)
    map_height = int((boundary[1][1] - boundary[1][0]) / d)
    map_depth = len(catogories)
    color_map_temp = np.zeros((map_height, map_width, 3)).astype(np.uint8)
    
    map_sum = np.sum(map_local, axis=2) # get all zero mask
    map_argmax = np.argmax(map_local, axis=2)
    
    for i in range(len(catogories)):
        color_map_temp[map_argmax == i] = catogories_color[i]
    
    color_map_temp[map_sum == 0] = [0,0,0] # recover all zero positions
    
    return color_map_temp

def read_pickle():
    filename = "/home/henry/log_odds.pickle"
    with open(filename, "rb") as filehandler:
        picked_file = pickle.load(filehandler)
        return picked_file

def stitch_image(im_src_list, homography_list):
    imSize = im_src_list[0].shape
    anchor = np.array([
        [imSize[1], 0, 0, imSize[1]],
        [0, 0, imSize[0], imSize[0]]
    ])

    x = homogenize(anchor)
    x_t = np.array(x)
    h_t = np.eye(3)

    min_x = 0
    min_y = 0
    for h in homography_list[::-1]:
        x_t = np.matmul(h, x_t)
        h_t = np.matmul(h, h_t)
        print(h[0,2], h[1,2])
        min_x_t = np.min(x_t[0,:])
        min_y_t = np.max(x_t[1,:])
        if min_x_t < min_x:
            min_x = min_x_t
        if min_y_t < min_y:
            min_y = min_y_t

    x_dst = dehomogenize(x_t)
    out_size = [1080, 720]
    im_dst = np.zeros((out_size[1], out_size[0], 3)).astype(np.uint8)
    
    
    for i in range(len(homography_list)-1):
        x_t = np.array(x)
        h_t = np.eye(3)
        for h in homography_list[i:-1]:
            h_t = np.matmul(h, h_t)
        h_t[0,2] -= min_x
        h_t[1,2] += 100
        im_src = color_map(im_src_list[i])
        im_out = cv2.warpPerspective(im_src, h_t, (out_size[0], out_size[1]))
        mask = np.sum(im_out, axis=2) != 0
        im_dst[mask] = im_out[mask]
    cv2.imshow("im", im_dst)
    cv2.waitKey(0)
    return x_dst

# main
def main():
    dict_in = read_pickle()
    im_src_list = dict_in['log_odds']
    homography_list = dict_in['h']

    im_dst = stitch_image(im_src_list, homography_list)

if __name__ == "__main__":
    main()