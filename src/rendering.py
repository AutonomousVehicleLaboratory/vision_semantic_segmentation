""" Render color map

Author: Henry Zhang
Date:March 01, 2020
"""

# module
import numpy as np


# parameters


# classes


# functions
def render_bev_map(map_local, labels, label_colors):
    """
    Render the Bird's eye view semantic map, the color of each pixel is picked by the max number of points
    Args:
        map_local: np.ndarray (H, W, C)
        labels: np.ndarray the label of
        label_colors: the RGB color of each label

    Returns:

    """
    height, width = map_local.shape[:2]
    colored_map = np.zeros((height, width, 3)).astype(np.uint8)

    map_sum = np.sum(map_local, axis=2)  # get all zero mask
    map_argmax = np.argmax(map_local, axis=2)

    for i in range(len(labels)):
        colored_map[map_argmax == i] = label_colors[i]

    colored_map[map_sum == 0] = [0, 0, 0]  # recover all zero positions

    return colored_map


# main
def main():
    pass


if __name__ == "__main__":
    main()
