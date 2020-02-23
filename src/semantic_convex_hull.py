#!/usr/bin/env python
# coding: utf-8

# In[56]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from skimage.measure import label, regionprops
from collections import Counter
import sys
import time
import copy
import pdb
sys.setrecursionlimit(100000)
def cc_recursive(img,pixExplored, marker,i,j, label):
    pixExplored[i,j] = marker
    #check all 8 neightbords and handle corner cases
    rows,cols = img.shape
    #(i-1,j-1)
    for m in range(i-1,i+2):
        for n in range(j-1,j+2):
            if((m-1 < 0) or (n-1 < 0) or(m+1 > rows) or (n+1 > cols)):
                continue
            else:

                if((img[m,n] == label) and (pixExplored[m,n] == 0) ):
                    cc_recursive(img,pixExplored, marker,m,n,label)
    
    
def connected_component(img, label):
    # your code here
    x,y = img.shape
    #explored pixels
    pixExplored = np.zeros(shape=(x,y))
    
    #Keeps track of the current CC
    marker = 1
    
    for i in range(0,x):
        for j in range(0,y):
            if((img[i,j] == label) and (pixExplored[i,j]==0)):
                #pixExplored[i,j] = marker
                
                #Update Neighbors
                cc_recursive(img,pixExplored, marker,i,j, label)
                #Update marker
                marker = marker+1

    return pixExplored

def generate_convex_hull(img, vis=False, index_care_about=1, index_to_vitualize=None, top_number=1, area_threshold=30):

    """
        Generate the convex hull
        Args:
            img: input img (h, w, 3)
            vis: True if vitualize the result; Only vitualize the last convex hull
            index_care_about: index that will be used to generate the convex hull
            index_to_vitualize: index that will be used to vitualize the result (index of the convex hull)
            top_number: the number most common label decided to choose
            area_threshold: only consider the connected component which contains the points greater than this area_threshold
        Returns:
            vertices: extracted vertices; list of numpy arrays; array shape- -- [2, number of vertices]
    """
    # cv2.imwrite("tempimage.jpg", img)
    # exit(0)
    rows, cols = img.shape
    img[img[:,:]!=index_care_about] = 0
    img[img[:,:]==index_care_about] = 1
    vertices = []
    
    kernel = np.ones((3,3), np.uint8)
    crosswalk = np.copy(img[:,:])
    if vis == True:
        plt.figure(0)
        plt.imshow(crosswalk)
    crosswalk = cv2.erode(crosswalk, kernel, iterations=1)

    if vis == True:
        plt.figure(1)
        plt.imshow(crosswalk)
    # crosswalks1 = connected_component(crosswalk, 1)
    crosswalks = label(crosswalk, connectivity=crosswalk.ndim)

    if np.all(crosswalks==0):
        return []
    if vis == True:
        plt.figure(2)
        plt.imshow(crosswalks)
    if index_to_vitualize == None:
        count = Counter(crosswalks[crosswalks!=0].reshape(-1)).most_common(top_number)
        index_to_vitualize = [x[0] for x in count if x[1] > area_threshold]

    for select_index in index_to_vitualize:
        chosen_crosswalk = np.copy(crosswalks)
        crosswalk_pts = np.zeros((1,2))
        indexes = np.where(chosen_crosswalk==select_index)

        crosswalk_pts = np.concatenate([np.array([i,j]).reshape(1,2) for (i,j) in zip(*indexes)])
        # Here I modefy comment the next line. And calculate all the vertices for 
        # chosen_crosswalk[chosen_crosswalk!=select_index] = 9


        crosswalk_pts = crosswalk_pts[1:, :]
        crosswalk_pts = np.fliplr(crosswalk_pts)
        hull = ConvexHull(points=crosswalk_pts, qhull_options='Q64')
        nodes = np.hstack((hull.vertices, hull.vertices[0]))
        vertices.append(crosswalk_pts[nodes, :].T)
        x_vertices = vertices[-1][0, :]
        y_vertices = vertices[-1][1, :]
    
    if vis == True:
        plt.figure(3)
        plt.imshow(chosen_crosswalk)

        fig = plt.figure(4)
        ax = fig.add_subplot(1,1,1)
        convex_hull_plot_2d(hull, ax=ax)  

        plt.figure(5)
        plt.imshow(img[:,:])
        
        plt.scatter(x_vertices, y_vertices, s=50, c='red', marker='o')
        plt.plot(x_vertices, y_vertices, c='red')
        plt.show()
    return vertices

def test_generate_convec_hull():
    import time

    img = cv2.imread('./tempimage.jpg', cv2.IMREAD_GRAYSCALE)

    tic = time.time()
    generate_convex_hull(img, vis=False)
    toc = time.time()
    print("running time: {:.6f}s".format(toc - tic))

def main():
    test_generate_convec_hull()

if __name__ == "__main__":
    main()
