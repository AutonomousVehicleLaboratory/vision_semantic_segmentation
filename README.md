# Probabilistic Semantic Mapping

This is the source code for the paper [Probabilistic Semantic Mapping for Urban Autonomous Driving Applications](https://arxiv.org/abs/2006.04894). It will fuse the LiDAR point cloud with the semantic segmented 2D camera image together and create a bird's-eye-view semantic map of the environment. 

## Dependencies

If you use Autoware to play a rosbag and load the point cloud map, then dependency 1 and 2 will be satisfied.

1. The code requires a rosbag with CAMERA information and localization information.

2. The code requires point map being published. A ROS package `map_reduction` from AVL repository will extract a local point cloud around the ego vehicle.

3. If you are testing the planar assumption method, the code also subscribe 'plane' from road_estimation, if no received, it will use a fake plane.

## Set Up

Docker environment: we are going to use the `astuff_autoware_nvidia` docker image as our primary develop environment. 

This code has been tested in ROS 1, Kinetic version. 

After log into the container for the first time,

1. Clone this repository in the src folder of a catkin workspace
2. Run `bash ./setup.sh` in the root directory to setup python packages (inside virtual environment if you want)
3. Add ros into your bash path: `source /opt/ros/kinetic/setup.bash` for convenience
4. Go to your workspace, run `catkin_make`
   - If you have dependencies in other catkin workspace, source them before running `catkin_make`
5. Source your setup file in `devel` directory of your catkin workspace (`source devel/setup.bash`)
6. Adjust the configuration of the semantic segmentation network as instructed in the next section. 



<u>Setup the semantic segmentation network</u>

1. Download the trained weight from Google Drive (`Living Laboratory-AVLResearch-Publication-IROS2020`). 

2. Create your local configuration by creating a copy from the template YAML file

   ```
   cp config/template.yaml config/avl.yaml
   ```

3. Open `config/avl.yaml` and set the `MODEL.WEIGHT` as the path to the trained weight. Set the `DATASET_CONFIG` to the path to the configuration file of the dataset so that you can visualize the semantic output in color, it is in the `config/class_19.json` in this repository.  

4. Make sure `DATASET.NUM_CLASSES` is equal to the number of classes. 

## To Run

1. Source the ros environment and workspace

2. Load the point cloud map and play the rosbag in Autoware

3. Run the command

   ```
   roslaunch camera1_mapping.launch
   ```

   This assumes you have already compiled the 'map_reduction' ROS package

4. Start Rviz for visualization

## ROS Node Information 

`vision_semantic_segmentation_node.py`  is 

Subscribing

```
type: sensor_msgs.msg Image \
topic: /camera{id}/image_raw
```

Publishing

```
type: sensor_msgs.msg Image \
topic: /camera{id}/semantic
```

`mapping.py` is 

Subscribing

```
type: sensor_msgs.msg Image \
topic: /camera{id}/semantic

type: 
topic: /reduced_map

type: 
topic: /points_raw
```

Publishing

```
topic: /semantic_local_map
topic: /semantic_point_cloud
```

## TODO

- [x] ros wrapper for camera1 and camera6
- [x] SemanticSegmentation class
- [x] integration test with Autoware
- [ ] record frequency and delay
- [ ] semantic mapping

## Credits

Author: David, Henry, Qinru, Hao. 