#!/usr/bin/env python
# coding: utf-8

# In[56]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from skimage.measure import label, regionprops
import sys
import time
import copy

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

def generate_convex_hull(img, vis=False, index_care_about=14, index_to_vitualize=11):
    """
        Generate the convex hull
        Args:
            img: input img (h, w, 3)
            vis: True if vitualize the result
            index_care_about: index that will be used to generate the convex hull
            index_to_vitualize: index that will be used to vitualize the result (index of the convex hull)
        Returns:
            vertices: extracted vertices
    """
    rows, cols, _ = img.shape
    """
    for i in range(rows):
        for j in range(cols):
            if(img[i,j,0] != 14):
                img[i,j,0] = 0
            else:
                img[i,j,0] = 1
    """
    img2 = copy.deepcopy(img)
    img2[img[:,:,0]!=index_care_about,0] = 0
    img2[img[:,:,0]==index_care_about,0] = 1
    img = img2

    
    kernel = np.ones((3,3), np.uint8)
    crosswalk = np.copy(img[:,:,0])
    if vis == True:
        plt.figure(0)
        plt.imshow(crosswalk)
    crosswalk = cv2.erode(crosswalk, kernel, iterations=1)

    if vis == True:
        plt.figure(1)
        plt.imshow(crosswalk)
    #crosswalks1 = connected_component(crosswalk, 1)
    crosswalks = label(crosswalk, connectivity=crosswalk.ndim)


    if vis == True:
        plt.figure(2)
        plt.imshow(crosswalks)

    select_index = index_to_vitualize
    chosen_crosswalk = np.copy(crosswalks)
    crosswalk_pts = np.zeros((1,2))
    indexes = np.where(chosen_crosswalk==select_index)
    crosswalk_pts = np.concatenate([np.array([i,j]).reshape(1,2) for (i,j) in zip(*indexes)])
    chosen_crosswalk[chosen_crosswalk!=select_index] = 9
    """
    for i in range(rows):
        for j in range(cols):
            if(chosen_crosswalk[i,j] == select_index):
                crosswalk_pts = np.vstack((crosswalk_pts, np.array([i, j])))
            else:
                chosen_crosswalk[i,j] = 9
    """



    crosswalk_pts = crosswalk_pts[1:, :]
    crosswalk_pts = np.fliplr(crosswalk_pts)
    hull = ConvexHull(points=crosswalk_pts, qhull_options='Q64')
    
    nodes = np.hstack((hull.vertices, hull.vertices[0]))
    vertices = crosswalk_pts[nodes, :]
    x_vertices = vertices[:, 0]
    y_vertices = vertices[:, 1]
    
    if vis == True:
        plt.figure(3)
        plt.imshow(chosen_crosswalk)

        fig = plt.figure(4)
        ax = fig.add_subplot(1,1,1)
        convex_hull_plot_2d(hull, ax=ax)  

        plt.figure(5)
        plt.imshow(img[:,:,0])
        
        plt.scatter(x_vertices, y_vertices, s=50, c='red', marker='o')
        plt.plot(x_vertices, y_vertices, c='red')
        plt.show()
    
    return vertices.T

def test_generate_convec_hull():
    import time

    img = cv2.imread('/home/henry/Documents/projects/pylidarmot/src/vision_semantic_segmentation/network_output_example/preds/3118.jpg')

    tic = time.time()
    generate_convex_hull(img, False)
    toc = time.time()
    print("running time: {:.6f}s".format(toc - tic))

def main():
    test_generate_convec_hull()

if __name__ == "__main__":
    main()
