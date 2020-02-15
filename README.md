# Semantic Segmentation Node

## NODE INFO
Input: \
type: sensor_msgs.msg Image \
topic: /camera{id}/image_raw

Output: \
type: sensor_msgs.msg Image \
topic: /camera{id}/semantic

Notice: this input topic is not published by vehicle, it is from vision_darknet_detect/launch/vision_yolo3_detect.launch, there a image_transport decode the compressed image to image_raw. We need to find another place to do this so that there is no dependency on that.

## TODO
- [x] ros wrapper for camera1 and camera6 \
- [ ] SemanticSegmentation class \
- [ ] integration test with Autoware \
- [ ] record frequency and delay