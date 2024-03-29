import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def convert_labels(gmap, mask=None):
    """ covert colors to labels """
    if mask is None:
        mask = np.ones((gmap.shape[0], gmap.shape[1]))
    else:
        mask = mask[:gmap.shape[0],:gmap.shape[1]]
    global_map = np.zeros((gmap.shape[0], gmap.shape[1]))
    global_map[np.logical_and(np.all(gmap == np.array([128, 64, 128]), axis=-1), mask)] = 1  # road
    global_map[np.logical_and(np.all(gmap == np.array([140, 140, 200]), axis=-1), mask)] = 2  # crosswalk
    global_map[np.logical_and(np.all(gmap == np.array([255, 255, 255]), axis=-1), mask)] = 3  # lane
    global_map[np.logical_and(np.all(gmap == np.array([244, 35, 232]), axis=-1), mask)] = 4  # sidewalk
    global_map[np.logical_and(np.all(gmap == np.array([107, 142, 35]), axis=-1), mask)] = 5  # vegetation
    return global_map


def read_img(global_map_path, mask=None):
    """ read the global map file and covert colors to labels """
    gmap = cv2.imread(global_map_path)
    # gmap = np.rot90(gmap, k=1, axes=(0, 1))
    global_map = convert_labels(gmap, mask)
    return gmap, global_map


class Test:
    def __init__(self, ground_truth_dir="./", shift_h=0, shift_w=0, logger=None):
        """
            Load the ground truth map and do transformations for it. Preprocess and store it for faster testing.
            ground_truth_dir: dir path to ground truth map
            preprocess: reprocess the rgb ground truth map to interger label if true.
        """
        truth_file_path = os.path.join(ground_truth_dir, "truth.npy")

        if os.path.exists(truth_file_path):
            print(truth_file_path, "exists, openning it.")
            with open(truth_file_path, 'rb') as f:
                self.ground_truth_mask = np.load(f)
        else:
            # preprocess to generate the file
            print(truth_file_path, "does not exist, preprocess the ground truth to generate it.")
            crosswalks = cv2.imread(os.path.join(ground_truth_dir, "bev-5cm-crosswalks.jpg"))
            road = cv2.imread(os.path.join(ground_truth_dir, "bev-5cm-road.jpg"))
            lane = cv2.imread(os.path.join(ground_truth_dir, "bev-5cm-lanes.jpg"))
            mask = cv2.imread(os.path.join(ground_truth_dir, "bev-5cm-mask.jpg"))
            # search if the lane marks have ground truth
            w, h = road.shape[:2]
            # white color in mask corresponds to valid region
            mask = cv2.resize(mask, (int(h / 4), int(w / 4)))
            mask2 = np.zeros((int(w / 4), int(h / 4)))
            mask2[np.all(mask == np.array([255, 255, 255]), axis=-1)] = 1
            mask = mask2
            self.mask = mask
            # downsample the image
            crosswalks = cv2.resize(crosswalks, (int(h / 4), int(w / 4)))
            road = cv2.resize(road, (int(h / 4), int(w / 4)))
            lane = cv2.resize(lane, (int(h / 4), int(w / 4)))
            # only use the region within the mask
            self.ground_truth_mask = np.zeros((road.shape[0], road.shape[1]))
            self.ground_truth_mask[np.logical_and(np.any(road > 0, axis=-1), mask)] = 1  # road
            self.ground_truth_mask[np.logical_and(np.any(lane > 0, axis=-1), mask)] = 3  # lanes
            self.ground_truth_mask[np.logical_and(np.any(crosswalks > 0, axis=-1), mask)] = 2  # crosswalk
            with open("truth.npy", 'wb') as f:
                np.save(f, self.ground_truth_mask)
            with open("mask.npy", 'wb') as f:
                np.save(f, mask)
        else:
            with open("truth.npy", 'rb') as f:
                self.ground_truth_mask = np.load(f)
            with open("mask.npy", 'rb') as f:
                self.mask = np.load(f)
        self.d = {0: "road", 1: "crosswalk", 2: "lane"}
        self.class_lists = [1, 2, 3]
        self.shift_w = shift_w
        self.shift_h = shift_h
        self.logger = logger

    def full_test(self, dir_path="./global_maps", visualize=False, latex_mode=False, verbose=False):
        """
            test all the generated maps in dir_path folders
            dir_path: dir path to generated maps
        """
        file_lists = os.listdir(dir_path)
        file_lists = [x for x in file_lists if ".png" in x]
        path_lists = [os.path.join(dir_path, x) for x in file_lists]
        iou_array = []
        miss_array = []
        for path in path_lists:
            print("You are testing\t" + path.split("/")[-1])
            _, generate_map = read_img(path, self.mask)
            gmap = self.ground_truth_mask[self.shift_w:generate_map.shape[0] + self.shift_w,
                   self.shift_h:generate_map.shape[1] + self.shift_h]
            iou_lists, miss = self.iou(gmap, generate_map, latex_mode=latex_mode, verbose=verbose)
            iou_array.append(np.array(iou_lists).reshape(1, -1))
            miss_array.append(miss)
            if visualize:
                mask = np.zeros(generate_map.shape)
                for cls in self.class_lists:
                    mask = np.logical_or(mask, generate_map == cls)
                generate_map[np.logical_not(mask)] = 0
                self.imshow(gmap, generate_map)
        miss = np.mean(miss_array)
        miss_percent = miss * 100
        iou_array = np.concatenate(iou_array, axis=0)
        iou_lists = np.mean(iou_array, axis=0)
        print("Final Batch evaluation")
        print("IOU for {}: {}\t{}: {}\t{}:{}\tmIOU: {}".format(self.d[0], iou_lists[0], self.d[1], iou_lists[1],
                                                               self.d[2], iou_lists[2],
                                                               np.mean(iou_lists)))
        print("Overall Missing rate: {}".format(miss))
        if latex_mode:
            print(f"&{iou_lists[0]:.3f}&{iou_lists[1]:.3f}&{iou_lists[2]:.3f}&{np.mean(iou_lists):.3f}&{miss_percent:.3g}\\\\ \\hline")

    def test_single_map(self, global_map):
        """ Calculate and print the IoU, accuracy and missing rate
            of the global_map and ground truth. 
            global_map: the semantic global map
        """
        generate_map = convert_labels(global_map)
        gmap = self.ground_truth_mask[self.shift_w:generate_map.shape[0] + self.shift_w,
               self.shift_h:generate_map.shape[1] + self.shift_h]
        self.iou(gmap, generate_map, verbose=True)

    def iou(self, gmap, generate_map, latex_mode=False, verbose=False):
        """
            Calculate and print the IoU, accuracy, missing rate
            gmap: ground truth map with interger labels
            generate_map: generated map with interger labels
        """
        iou_lists = []
        acc_lists = []
        for cls in self.class_lists:
            gmap_layer = gmap == cls
            map_layer = generate_map == cls
            intersection = float(np.sum(gmap_layer * map_layer))
            union = float(np.sum(gmap_layer) + np.sum(map_layer) - intersection)
            iou = intersection / union
            iou_lists.append(iou)
            acc = intersection / np.sum(gmap_layer)
            acc_lists.append(acc)
        miss = 1 - np.sum(np.logical_and((gmap > 0), (generate_map > 0))) / np.sum(gmap > 0)
        accuracy = np.sum((gmap == generate_map)[gmap > 0]) / np.sum(gmap > 0)
        if verbose:
            if not latex_mode:
                print("IOU for {}: {}\t{}: {}\t{}:{}\tmIOU: {}".format(self.d[0], iou_lists[0], self.d[1],
                                                                       iou_lists[1],
                                                                       self.d[2], iou_lists[2],
                                                                       np.mean(iou_lists)))
                print("Accuracy for {}: {}\t{}: {}\t{}:{}\tmean Accuracy: {}".format(self.d[0], acc_lists[0],
                                                                                     self.d[1], acc_lists[1],
                                                                                     self.d[2],
                                                                                     acc_lists[2],
                                                                                     accuracy))
                print("Overall Missing rate: {}".format(miss))
            else:
                miss_percent = miss * 100
                print(f"&{iou_lists[0]:.3f}&{iou_lists[1]:.3f}&{iou_lists[2]:.3f}&{np.mean(iou_lists):.3f}&{miss_percent:.3g}\\\\ \\hline")
        return iou_lists, miss

    def imshow(self, img1, img2):
        fig, axes = plt.subplots(1, 2)
        axes[0].matshow(img1)
        axes[1].matshow(img2)
        plt.show()


if __name__ == "__main__":
    visualize = False  # True if visualizing global maps and ground truth, default to no visualization
    latex_mode = False # True if generate latex code of tabels
    verbose = True # True if print evaluation results for every image False if print final average result
    import sys

    # add arguement -v for visualization
    if len(sys.argv) > 1:
        if sys.argv[1] == '-v':
            visualize = True

    test = Test(ground_truth_dir="./ground_truth")
    test.full_test(dir_path="./global_maps", visualize=visualize, latex_mode=latex_mode, verbose=verbose)
