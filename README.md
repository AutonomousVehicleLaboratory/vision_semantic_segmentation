# Semantic Segmentation Node

## Setup

Assume that you are using the docker image `astuff_autoware_nvidia` 

0. prerequisite, ROS, this code has been tested with ROS Kinetic.
1. clone this repository to a src folder of a catkin workspace
2. Run `bash ./setup.sh` in the root directory to setup python packages (inside virtual environment if you want)
3. source ros (Add ros into your bash path `source /opt/ros/kinetic/setup.bash` for convenience)
4. go to your workspace, run `catkin_make`
   - If you have dependencies in other catkin workspace, source them before running `catkin_make`
5. source your setup file in devel directory of your catkin workspace (`source devel/setup.bash`)
6. config you semantic segmentation as below instructions

## NODE INFO

Input: \
type: sensor_msgs.msg Image \
topic: /camera{id}/image_raw

Output: \
type: sensor_msgs.msg Image \
topic: /camera{id}/semantic

Notice: this input topic is not published by vehicle, it is from vision_darknet_detect/launch/vision_yolo3_detect.launch, there a image_transport decode the compressed image to image_raw. We need to find another place to do this so that there is no dependency on that.

## Semantic Segmentation

### Run the semantic segmentation network

1. Download the trained weight from Google Drive (`Living Laboratory-AVLResearch-Publication-IROS2020`). 

2. Create your local configuration by creating a copy from the template YAML file

   ```
   cp config/template.yaml config/avl.yaml
   ```

3. Open `config/avl.yaml` and set the `MODEL.WEIGHT` as the path to the trained weight. Set the `DATASET_CONFIG` to the path to the configuration file of the dataset so that you can visualize the semantic output in color, it is in the `config/class_19.json` in this repository.  

4. Make sure `DATASET.NUM_CLASSES` is equal to the number of classes. 

## TODO

- [x] ros wrapper for camera1 and camera6
- [x] SemanticSegmentation class
- [x] integration test with Autoware
- [ ] record frequency and delay

