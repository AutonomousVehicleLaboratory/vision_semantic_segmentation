import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import pdb


def read_img(global_map_path):
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
    def __init__(self, ground_truth_dir="./ground_truth", preprocess=True):
        """
            Load the ground truth map and do transformations for it. Preprocess and store it for faster testing.
            ground_truth_dir: dir path to ground truth map
            preprocess: reprocess the rgb ground truth map to interger label if true.
        """
        if preprocess:
            crosswalks = cv2.imread(os.path.join(ground_truth_dir, "bev-5cm-crosswalks.jpg"))
            road = cv2.imread(os.path.join(ground_truth_dir, "bev-5cm-road.jpg"))
            # search if the lane marks have ground truth
            file_lists = os.listdir(ground_truth_dir)
            binary_lists = ["lane" in x for x in file_lists]
            index = [i for i, val in enumerate(binary_lists) if val]
            if True in binary_lists:
                lane_file_path = os.path.join(ground_truth_dir, file_lists[index[0]])
                lane = cv2.imread(lane_file_path)
            w, h = road.shape[:2]
            crosswalks = cv2.resize(crosswalks, (int(h / 4), int(w / 4)))
            road = cv2.resize(road, (int(h / 4), int(w / 4)))
            self.ground_truth_mask = np.zeros((road.shape[0], road.shape[1]))
            self.ground_truth_mask[np.any(road > 0, axis=-1)] = 1  # road
            if True in binary_lists:
                lane = cv2.resize(lane, (int(h / 4), int(w / 4)))
                self.ground_truth_mask[np.any(lane > 0, axis=-1)] = 3  # lanes
            self.ground_truth_mask[np.any(crosswalks > 0, axis=-1)] = 2  # crosswalk

            with open("truth.npy", 'wb') as f:
                np.save(f, self.ground_truth_mask)
        else:
            with open("truth.npy", 'rb') as f:
                self.ground_truth_mask = np.load(f)
        # rotate the ground truth mask to align with the generated map
        # self.ground_truth_mask = np.rot90(self.ground_truth_mask, k=1, axes=(0, 1))
        self.d = {0: "road", 1: "crosswalk", 2: "lane"}
        self.class_lists = [1, 2]
        if 3 in self.ground_truth_mask:
            self.class_lists.append(3)

    def full_test(self, dir_path="./global_maps", visualize=False, latex_mode=False):
        """
            test all the generated maps in dir_path folders
            dir_path: dir path to generated maps
        """
        shift_w = 0  # 200
        shift_h = 0  # 500
        file_lists = os.listdir(dir_path)
        file_lists = [x for x in file_lists if ".png" in x]
        path_lists = [os.path.join(dir_path, x) for x in file_lists]
        iou_array = []
        miss_array = []
        for path in path_lists:
            print("You are testing " + path.split("/")[-1])
            _, generate_map = read_img(path)
            gmap = self.ground_truth_mask[shift_w:generate_map.shape[0] + shift_w,
                   shift_h:generate_map.shape[1] + shift_h]
            iou_lists, miss = self.iou(gmap, generate_map, latex_mode=latex_mode)
            iou_array.append(np.array(iou_lists).reshape(1,-1))
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
        iou_lists = np.mean(iou_array,axis=0)
        print("Average evaluation")
        if len(self.class_lists) == 2:
            print("IOU for {}: {}\t{}: {}\tmIOU: {}".format(self.d[0], iou_lists[0], self.d[1], iou_lists[1],
                                                            np.mean(iou_lists)))
            print("Overall Missing rate: {}".format(miss))
        else:
            print("IOU for {}: {}\t{}: {}\t{}:{}\tmIOU: {}".format(self.d[0], iou_lists[0], self.d[1], iou_lists[1],
                                                                   self.d[2], iou_lists[2],
                                                                   np.mean(iou_lists)))
            print("Overall Missing rate: {}".format(miss))
        #print(f"&{iou_lists[0]:.3f}&{iou_lists[1]:.3f}&{iou_lists[2]:.3f}&{np.mean(iou_lists):.3f}&{miss_percent:.3g}\\\\ \\hline")

    def iou(self, gmap, generate_map, latex_mode=False, verbose=False):
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
            intersection = np.sum(gmap_layer * map_layer)
            union = np.sum(gmap_layer) + np.sum(map_layer) - intersection
            iou = intersection / union
            iou_lists.append(iou)
            acc = intersection / np.sum(gmap_layer)
            acc_lists.append(acc)
        miss = 1 - np.sum(np.logical_and((gmap > 0), (generate_map > 0))) / np.sum(gmap > 0)
        accuracy = np.sum((gmap == generate_map)[gmap > 0]) / np.sum(gmap > 0)
        if verbose:
            if not latex_mode:
                if len(self.class_lists) == 2:
                    print("IOU for {}: {}\t{}: {}\tmIOU: {}".format(self.d[0], iou_lists[0], self.d[1], iou_lists[1],
                                                                    np.mean(iou_lists)))
                    print("Accuracy for {}: {}\t{}: {}\tmean Accuracy: {}".format(self.d[0], acc_lists[0], self.d[1],
                                                                                  acc_lists[1],
                                                                                  accuracy))
                    print("Overall Missing rate: {}".format(miss))
                else:
                    print("IOU for {}: {}\t{}: {}\t{}:{}\tmIOU: {}".format(self.d[0], iou_lists[0], self.d[1], iou_lists[1],
                                                                           self.d[2], iou_lists[2],
                                                                           np.mean(iou_lists)))
                    print("Accuracy for {}: {}\t{}: {}\t{}:{}\tmean Accuracy: {}".format(self.d[0], acc_lists[0],
                                                                                         self.d[1], acc_lists[1], self.d[2],
                                                                                         acc_lists[2],
                                                                                         accuracy))
                    print("Overall Missing rate: {}".format(miss))
            else:
                # in latex mode output latex code directly. This is serving for generating latex tables
                if len(self.class_lists) == 2:
                    raise ValueError(
                        "Please put lane ground truth map in correct folder. Latex Mode works only when all 3 labels are given.")
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
    preprocess = False
    latex_mode = False
    visualize = False
    test = Test(ground_truth_dir="./ground_truth", preprocess=preprocess)
    test.full_test(dir_path="./global_maps", visualize=visualize, latex_mode=latex_mode)
