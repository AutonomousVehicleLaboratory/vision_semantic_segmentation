# Semantic Segmentation Node

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

Download the trained weight from Google Drive (`Living Laboratory -AVLResearch-Publication-IROS2020`). 

Open `src/network/experiments/video_generation/avl.yaml` and set the `MODEL.WEIGHT` as the path to the trained weight. 

Make sure `DATASET.NUM_CLASSES` is equal to the number of classes. Currently we are using `39`.

## TODO

- [x] ros wrapper for camera1 and camera6
- [ ] SemanticSegmentation class
- [ ] integration test with Autoware
- [ ] record frequency and delay