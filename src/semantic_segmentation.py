""" Semantic Segmentation Class

Author: Qinru
Date:February 14, 2020
"""

# module
import numpy as np
import cv2
# parameters

import os.path as osp
import sys
import torch
import torch.nn as nn
import torchvision.transforms as T

sys.path.insert(0, osp.join(osp.dirname(__file__), "network"))

from deeplab_v3_plus.models.build import build_model


# classes
class SemanticSegmentation():
    def __init__(self, config_file):
        """

        Args:
            config_file: The path to the network configuration file
        """

        # load the configuration
        # import on-the-fly to avoid overwriting cfg
        from deeplab_v3_plus.config.demo import cfg
        cfg.merge_from_file(config_file)

        self.model = build_model(cfg)[0]
        self.model = nn.DataParallel(self.model).cuda()

        # Load weight
        checkpoint = torch.load(cfg.MODEL.WEIGHT, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint.pop('model'))

        # Build transform
        self.transform = T.Compose([
            T.ToPILImage(),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), True),
        ])

    def segmentation(self, image_in):
        """
        Generate the semantic segmentation from the input image
        Args:
            image_in: numpy array (h, w, 3)

        Returns:
            the semantic segmentation mask

        """
        self.model.eval()
        with torch.no_grad():
            image_tensor = self.transform(image_in)
            image_tensor = image_tensor.unsqueeze(dim=0).cuda()
            preds = self.model(image_tensor, upsample_pred=False)
            preds = torch.argmax(preds, dim=1).squeeze().cpu().numpy()
            return preds

# functions


# main
def main():
    pass

if __name__ == "__main__":
    main()
    network = SemanticSegmentation(config_file="/home/qinru/avl/semantic/deeplab/experiments/video_generation/avl.yaml")

    import PIL.Image as Image
    import matplotlib.pyplot as plt
    import numpy as np

    image = Image.open(
        "/home/qinru/avl/semantic/deeplab/data/mapillary-vistas-dataset_public_v1.1_processed/validation/images/0B5qssoIEl6LguVQjoRiDQ.jpg")
    image = np.array(image)

    segmentation = network.segmentation(image)

    plt.imshow(segmentation)
    plt.show()