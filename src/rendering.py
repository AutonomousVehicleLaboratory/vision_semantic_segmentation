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
    
    np.save('/home/henry/Pictures/map_local.npy', map_local)

    map_sum = np.sum(map_local, axis=2) # get all zero mask
    map_argmax = np.argmax(map_local, axis=2)
    
    for i in range(len(catogories)):
        colored_map[map_argmax == i] = catogories_color[i]
    
    colored_map[map_sum == 0] = [0,0,0] # recover all zero positions
    
    return colored_map

def fill_black(img):
    """ fill the black area according to the labels in its 3*3 neighbor.
    the approach is based on a priority list
    this approach will expand the prioritized labels
    """
    priority_list = [0, 3, 4, 2, 1] # from low to high priority
    xmax, ymax = img.shape[0], img.shape[1]

    # constructing 3*3 area for faster option
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
    
    # get colors
    for label in priority_list:
        img_out[mask_dict[label]] = label_colors[label, 0]

    # img_out[img[1:xmax-1,1:ymax-1, 0]!=0] = img[1:xmax-1, 1:ymax-1, 0][img[1:xmax-1,1:ymax-1, 0]!=0]
    # expand to three channels
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
    """ fill the black area with the most popular label within its 3*3 neighbor """
    from scipy.stats import mode
    img_filled = np.zeros(img.shape, dtype=np.uint8)
    xmax, ymax = img.shape[0], img.shape[1]
    print(xmax, ymax)
    for x in range(1, xmax-1):
        if x % 100 == 0:
            print(x)
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
    # img = img[900:1300, 400:800]#  cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img[0:6000, 0:5000]
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    fig.canvas.manager.full_screen_toggle()

    img_filled = fill_black(img)
    print('done firts')
    img_filled = fill_black_for_loop(img_filled)

    cv2.imwrite('/home/henry/Pictures/global_map_rendered.png', img_filled)

    ax1.imshow(img)
    ax2.imshow(img_filled)

    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    plt.show()


def test_separate_map():
    map_local = np.load('/home/henry/Pictures/map_local.npy')[550:-200, 150:450]
    from matplotlib import pyplot as plt
    # visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
    fig.canvas.manager.full_screen_toggle()

    
    im1 = ax1.imshow(map_local[:,:,3])
    fig.colorbar(im1, ax=ax1)
    ax1.set_title('vegetation layer')

    im2 = ax2.imshow((map_local[:,:,3] > 5).astype(np.uint8))
    fig.colorbar(im2, ax=ax2)
    ax2.set_title('larger than 5')

    im3 = ax3.imshow((map_local[:,:,3] > 10).astype(np.uint8))
    fig.colorbar(im3, ax=ax3)
    ax3.set_title('larger than 10')

    im4 = ax4.imshow((map_local[:,:,3] > 20).astype(np.uint8))
    fig.colorbar(im4, ax=ax4)
    ax4.set_title('larger than 20')

    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    plt.show()

def color_map_portion(map_local, priority, categories_color, portion = [0.01, 0.01, 0.01, 0.01, 0.01]):
    """ color the map by whether the labels have a perceptage higher than the specified value
    Params:
        map_local: original data each layer correspond to a channel
        categories_color: corresponding color for each channel
        priority: priority ordered from low to high, higher will overwrite lower colors
        portion: specify the threshold portion for each category, default to minimum requirement
    """
    colored_map = np.zeros((map_local.shape[0], map_local.shape[1], 3)).astype(np.uint8)
    
    map_portion = map_local / np.sum(map_local, axis=2)[:,:,None]
    map_portion = map_portion[:,:,priority]
    categories_color = categories_color[priority]

    for i in range(len(priority)):
        colored_map[map_portion[:,:,i] >= portion[i]] = categories_color[i][[0,1,2]]
    
    return colored_map

def test_render_portion():
    # categories = [107, 244, 128, 255, 140] # values of labels in the iuput images of mapping
    # category_colors = np.array([
    #         [107, 142, 35], # vegetation
    #         [244, 35, 232], # sidewalk
    #         [128, 64, 128], # road
    #         [255, 255, 255], # lane
    #         [140, 140, 200] # crosswalk
    #     ])
    priority = [3,4,0,2,1]
    map_local = np.load('/home/henry/Pictures/map_local_small.npy')[550:-200, 150:450]

    
    
    from matplotlib import pyplot as plt
    # visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
    fig.canvas.manager.full_screen_toggle()

    
    im1 = ax1.imshow(color_map_portion(map_local, priority, label_colors, portion = [0.01, 0.01, 0.01, 0.01, 0.01]))
    fig.colorbar(im1, ax=ax1)
    ax1.set_title('side walk layer')

    im2 = ax2.imshow(color_map_portion(map_local, priority, label_colors, portion = [0.1, 0.1, 0.5, 0.15, 0.05]))
    fig.colorbar(im2, ax=ax2)
    ax2.set_title('larger than 0.05')
    
    map_local = apply_filter(map_local)
    
    im3 = ax3.imshow(color_map_portion(map_local, priority, label_colors, portion = [0.1, 0.1, 0.5, 0.1, 0.05]))
    fig.colorbar(im3, ax=ax3)
    ax3.set_title('larger than 0.15')

    im4 = ax4.imshow(color_map_portion(map_local, priority, label_colors, portion = [0.1, 0.1, 0.5, 0.12, 0.05]))
    fig.colorbar(im4, ax=ax4)
    ax4.set_title('larger than 0.5')

    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    plt.show()

def apply_filter(src):
    import cv2
    ddepth = -1

    kernel_size = 3
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
    kernel /= (kernel_size * kernel_size)
    
    dst = cv2.filter2D(src, ddepth, kernel)
    
    return dst

def exec_render_portion():
    import cv2
    priority = [3,4,0,2,1]
    map_local = np.load('/home/henry/Pictures/map_local_0721.npy')

    map_local = apply_filter(map_local)
    
    color_map = color_map_portion(map_local, priority, label_colors, portion = [0.1, 0.1, 0.5, 0.1, 0.05])

    # color_map = fill_black(color_map)

    cv2.imwrite('/home/henry/Pictures/global_map_new.png', color_map)

# main
def main():
    # test_filter()
    # test_separate_map()
    # test_render_portion()
    exec_render_portion()


if __name__ == "__main__":
    main()