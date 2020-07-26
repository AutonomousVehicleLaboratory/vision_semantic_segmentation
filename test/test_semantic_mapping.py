import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


def read_img(global_map_path):
    """ read the global map file and covert colors to labels """
    gmap = cv2.imread(global_map_path)
    # gmap = np.rot90(gmap, k=1, axes=(0, 1))
    global_map = np.zeros((gmap.shape[0], gmap.shape[1]))
    global_map[np.all(gmap == np.array([128, 64, 128]), axis=-1)] = 1  # road
    global_map[np.all(gmap == np.array([140, 140, 200]), axis=-1)] = 2  # crosswalk
    global_map[np.all(gmap == np.array([244, 35, 232]), axis=-1)] = 3  # sidewalk
    global_map[np.all(gmap == np.array([255, 255, 255]), axis=-1)] = 4  # lane
    global_map[np.all(gmap == np.array([107, 142, 35]), axis=-1)] = 5  # vegetation
    return gmap, global_map


class Test:
    def __init__(self, ground_truth_dir="./"):
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
            w, h = road.shape[:2]
            crosswalks = cv2.resize(crosswalks, (int(h / 4), int(w / 4)))
            road = cv2.resize(road, (int(h / 4), int(w / 4)))
            self.ground_truth_mask = np.zeros((road.shape[0], road.shape[1]))
            self.ground_truth_mask[np.any(road > 0, axis=-1)] = 1  # road
            self.ground_truth_mask[np.any(crosswalks > 0, axis=-1)] = 2  # crosswalk
            with open(truth_file_path, 'wb') as f:
                np.save(f, self.ground_truth_mask)

        self.d = {0: "road", 1: "crosswalk"}
        self.class_lists = [1, 2]

    def full_test(self, dir_path="./global_maps", visualize=False):
        """
            test all the generated maps in dir_path folders
            dir_path: dir path to generated maps
        """
        shift_w = 0  # 200
        shift_h = 0  # 500
        file_lists = os.listdir(dir_path)
        file_lists = [x for x in file_lists if ".png" in x]
        path_lists = [os.path.join(dir_path, x) for x in file_lists]
        for path in path_lists:
            print("You are testing " + path.split("/")[-1])
            _, generate_map = read_img(path)
            gmap = self.ground_truth_mask[shift_w:generate_map.shape[0] + shift_w,
                   shift_h:generate_map.shape[1] + shift_h]
            self.iou(gmap, generate_map)
            if visualize:
                # pdb.set_trace()
                mask = np.zeros(generate_map.shape)
                for cls in self.class_lists:
                    mask = np.logical_or(mask, generate_map == cls)
                generate_map[np.logical_not(mask)] = 0
                self.imshow(gmap, generate_map)

    def iou(self, gmap, generate_map):
        """
            calculate the iou, accuracy, missing rate
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
        miss = 1 - np.sum(np.logical_and((gmap > 0), (generate_map > 0))) / float(np.sum(gmap > 0))
        accuracy = np.sum((gmap == generate_map)[gmap > 0]) / float(np.sum(gmap > 0))

        print("IOU for {}: {}\t{}: {}\tmIOU: {}".format(self.d[0], iou_lists[0], self.d[1], iou_lists[1],
                                                        np.mean(iou_lists)))
        print("Accuracy for {}: {}\t{}: {}\tmean Accuracy: {}".format(self.d[0], acc_lists[0], self.d[1], acc_lists[1],
                                                                      accuracy))
        print("Overall Missing rate: {}".format(miss))

    def imshow(self, img1, img2):
        fig, axes = plt.subplots(1, 2)
        axes[0].matshow(img1)
        axes[1].matshow(img2)
        plt.show()


if __name__ == "__main__":

    visualize = False # default to no visualization
    import sys

    # add arguement -v for visualization
    if len(sys.argv) > 1:
        if sys.argv[1] == '-v':
            visualize = True

    test = Test(ground_truth_dir="./ground_truth")
    test.full_test(dir_path="./global_maps", visualize=visualize)
