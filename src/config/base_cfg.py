# The basic configuration system
import os.path as osp
import sys

# Add src directory into the path
sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), "../")))

from yacs.config import CfgNode as CN

from src.network.deeplab_v3_plus.config.demo import cfg as network_cfg

_C = CN()


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()


# Usually I will use UPPER CASE for non-parametric variables, and lower case for parametric variables because it can be
# directly pass into the function as key-value pairs.

# --------------------------------------------------------------------------- #
# General Configuration
# --------------------------------------------------------------------------- #
# We will create a sub-folder with this name in the output directory
# _C.TASK_NAME = "vanilla_confusion_matrix"
_C.TASK_NAME = "cfn_mtx_with_intensity"

# '@' here means the root directory of the project
_C.OUTPUT_DIR = "@/outputs"

# If the time stamp reaches this threshold, we will stop generating the map.
# Usually, our start time frame is 390. If you want a shorter test time, you can set it to 1581541270, which is about
# 20 seconds.
_C.TEST_END_TIME = 1581541450

# Ground truth semantic map label directory
_C.GROUND_TRUTH_DIR = ""

# Set the random seed of the network for reproducibility
_C.RNG_SEED = -1

# The associate index of each label in the semantic segmentation network
_C.LABELS = [2, 1, 8, 10, 3]
# The name of the label
_C.LABELS_NAMES = ["road", "crosswalk", "lane", "vegetation", "sidewalk"]
# The RGB color of each label. We will use this to identify the label of each RGB pixel
_C.LABEL_COLORS = [
    [128, 64, 128],  # road
    [140, 140, 200],  # crosswalk
    [255, 255, 255],  # lane
    [107, 142, 35],  # vegetation
    [244, 35, 232],  # sidewalk
]

# --------------------------------------------------------------------------- #
# Mapping Configuration
# --------------------------------------------------------------------------- #
_C.MAPPING = CN()

# The resolution of the occupancy grid in meters
_C.MAPPING.RESOLUTION = 0.1
# The boundary of the occupancy grid, in meters. The format of the boundary is [[xmin, xmax], [ymin, ymax]]
_C.MAPPING.BOUNDARY = [[100, 300], [800, 1000]]
# This variable defines the way how we estimate the depth from the image. If use "points_map", then we are using the
# offline point cloud map. If use the points_raw", then we are using the the online point cloud map, i.e. the output
# from the LiDAR per frame.
_C.MAPPING.DEPTH_METHOD = 'points_map'

# Point cloud setting
_C.MAPPING.PCD = CN()
# If True, use the point cloud intensity data to augment our semantic BEV estimation
_C.MAPPING.PCD.USE_INTENSITY = True
_C.MAPPING.PCD.RANGE_MAX = 100.0

_C.MAPPING.CONFUSION_MTX = CN()
# The load path of the confusion matrix
_C.MAPPING.CONFUSION_MTX.LOAD_PATH = ""

# --------------------------------------------------------------------------- #
# Vision Semantic Segmentation Configuration
# --------------------------------------------------------------------------- #
_C.VISION_SEM_SEG = CN()

# Determine the scale of the input image, from 0 to 1.
_C.VISION_SEM_SEG.IMAGE_SCALE = 1.0

# --------------------------------------------------------------------------- #
# Semantic Segmentation Network Configuration
# --------------------------------------------------------------------------- #
network_cfg.TRAIN_DATASET = "Mapillary"
network_cfg.DATASET_CONFIG = "/mnt/avl_shared/qinru/iros2020/resnext50_os8/config.json"

network_cfg.MODEL.TYPE = "DeepLabv3+"
# Path to Pre-trained or checkpointed weights
network_cfg.MODEL.WEIGHT = "/mnt/avl_shared/qinru/iros2020/resnext50_os8/run1/model_best.pth"
# When set to True, the model will use synchronized batch normalization
network_cfg.MODEL.SYNC_BN = False
network_cfg.MODEL.DECODER.LOW_LEVEL_OUT_CHANNELS = 256
network_cfg.MODEL.BACKBONE = "resnext50_32x4d"
network_cfg.MODEL.OUTPUT_STRIDE = 8

network_cfg.DATASET.NAME = "AVL"
network_cfg.DATASET.IN_CHANNELS = 3
network_cfg.DATASET.NUM_CLASSES = 19

_C.VISION_SEM_SEG.SEM_SEG_NETWORK = network_cfg
