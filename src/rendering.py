""" Render color map

Author: Henry Zhang
Date:March 01, 2020
"""

# module
import numpy as np

# parameters
label_colors = np.array([
            [128, 64, 128], # road
            [140, 140, 200], # crosswalk
            [255, 255, 255], # lane
            [107, 142, 35], # vegetation
            [244, 35, 232] # sidewalk
        ])

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

def fill_black(img):
    xmax, ymax = img.shape[0], img.shape[1]
    img_stacked = np.vstack([img[1:xmax-1, 1:ymax-1, 0].reshape([-1, xmax-2, ymax-2]),
               img[0:xmax-2, 1:ymax-1, 0].reshape([-1, xmax-2, ymax-2]),
               img[2:xmax, 1:ymax-1, 0].reshape([-1, xmax-2, ymax-2]),
               img[1:xmax-1, 0:ymax-2, 0].reshape([-1, xmax-2, ymax-2]),
               img[0:xmax-2, 0:ymax-2, 0].reshape([-1, xmax-2, ymax-2]),
               img[2:xmax, 0:ymax-2, 0].reshape([-1, xmax-2, ymax-2]),
               img[1:xmax-1, 2:ymax, 0].reshape([-1, xmax-2, ymax-2]),
               img[0:xmax-2, 2:ymax, 0].reshape([-1, xmax-2, ymax-2]),
               img[2:xmax, 2:ymax, 0].reshape([-1, xmax-2, ymax-2])])
    mask_dict = {}
    for i in range(len(label_colors)):
        mask_dict[i] = np.any(img_stacked == label_colors[i,0], axis=0)
    
    img_out = np.zeros((xmax-2, ymax-2), dtype=np.uint8)
    for label in [0, 3, 4, 2, 1]:
        img_out[mask_dict[label]] = label_colors[label, 0]
    
    img_out = np.concatenate([img_out.reshape([xmax-2, ymax-2, 1]), 
                         img_out.reshape([xmax-2, ymax-2, 1]),
                         img_out.reshape([xmax-2, ymax-2, 1])], axis=2)
    img_out = resume_color(img_out)

    return img_out

def resume_color(img):
    for i in range(len(label_colors)):
        mask = img[:,:,0] == label_colors[i,0]
        img[mask] = label_colors[i]
    return img

def fill_black_for_loop(img):
    from scipy.stats import mode
    img_filled = np.zeros(img.shape, dtype=np.uint8)
    xmax, ymax = img.shape[0], img.shape[1]
    for x in range(1, xmax-1):
        for y in range(1, ymax-1):
            color_list = []
            for i in range(x-1, x+2):
                for j in range(y-1, y+2):
                    if img[i,j, 0] != 0:
                        color_list.append(img[i,j,0])
            xy_mode, count = mode(color_list)
            img_filled[x,y] = xy_mode if xy_mode.shape[0] != 0 else 0

    img_filled = resume_color(img_filled)
    
    return img_filled

def test_filter():
    import cv2
    from matplotlib import pyplot as plt
    img = cv2.imread('/home/henry/Pictures/global_map.png')
    img = img[900:1300, 400:800]#  cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    fig.canvas.manager.full_screen_toggle()

    img_filled = fill_black(img)
    img_filled = fill_black_for_loop(img_filled)

    ax1.imshow(img)
    ax2.imshow(img_filled)

    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    plt.show()


# main
def main():
    test_filter()

if __name__ == "__main__":
    main()