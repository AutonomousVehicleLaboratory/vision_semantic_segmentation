""" Render color map

Author: Henry Zhang
Date:March 01, 2020
"""

# module
import numpy as np

# parameters


# classes


# functions
def color_map_local(map_local, catogories, catogories_color):
    """ color the map by which label has max number of points """
    colored_map = np.zeros((map_local.shape[0], map_local.shape[1], 3)).astype(np.uint8)
    
    map_sum = np.sum(map_local, axis=2) # get all zero mask
    map_argmax = np.argmax(map_local, axis=2)
    
    for i in range(len(catogories)):
        colored_map[map_argmax == i] = catogories_color[i]
    
    colored_map[map_sum == 0] = [0,0,0] # recover all zero positions
    
    return colored_map

# main
def main():
    pass

if __name__ == "__main__":
    main()