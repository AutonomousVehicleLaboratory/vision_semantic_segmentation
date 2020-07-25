# The basic configuration system
from yacs.config import CfgNode as CN

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

# The resolution of the occupancy grid in meters
_C.RESOLUTION = 0.1
# The boundary of the occupancy grid, in meters. The format of the boundary is [[xmin, xmax], [ymin, ymax]]
_C.BOUNDARY = [[100, 300], [800, 1000]]
# This variable defines the way how we estimate the depth from the image. If use "points_map", then we are using the
# offline point cloud map. If use the points_raw", then we are using the the online point cloud map, i.e. the output
# from the LiDAR per frame.
_C.DEPTH_METHOD = 'points_map'

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

# Point cloud setting
_C.PCD = CN()
# The point cloud intensity threshold. We use this to identify the high intensity area in the point cloud map.
_C.PCD.INTENSITY_THLD = 15
_C.PCD.USE_INTENSITY = True  # Use intensity to augment the data if True

_C.CONFUSION_MTX = CN()
# The load path of the confusion matrix
_C.CONFUSION_MTX.LOAD_PATH = "/home/users/qinru/codebase/ros_workspace/src/vision_semantic_segmentation/external_data/confusion_matrix/run_trad_cnn/cfn_mtx.npy"
