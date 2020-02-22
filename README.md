# Semantic Segmentation Node

## Setup

Assume that you are using the docker image `astuff_autoware_nvidia` 

1. Add ros into your bash path `source /opt/ros/kinetic/setup.bash`
2. Run `bash ./setup.sh` in the root directory

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

3. Open `config/avl.yaml` and set the `MODEL.WEIGHT` as the path to the trained weight. 

4. Make sure `DATASET.NUM_CLASSES` is equal to the number of classes. 

## TODO

- [x] ros wrapper for camera1 and camera6
- [x] SemanticSegmentation class
- [ ] integration test with Autoware
- [ ] record frequency and delay

