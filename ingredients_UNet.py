#!/usr/bin/python3

from unet import Unet
from PIL import Image
import numpy as np
import cv2

import sys
import os

# Get the absolute path of the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
# Add the parent directory to sys.path
sys.path.append(parent_dir)

from post_processing.image_utlis import ImageUtils

class Ingredients_UNet(Unet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_top_layer(self, mask, top_layer_rgb):
        mask = np.array(mask)
        mod_img = np.zeros(
            [np.shape(mask)[0], np.shape(mask)[1], np.shape(mask)[2]]
        )
        if mask.ndim == 3:
            for height in range(mask.shape[0]):
                for width in range(mask.shape[1]):
                    if (mask[height][width] == top_layer_rgb).all(): 
                      mod_img[height][width] = mask[height][width][0]
        return Image.fromarray(np.uint8(mod_img))

if __name__ == "__main__":
    # Test initialisation for cheese
    Cheese_UNet = Ingredients_UNet(count=False, classes=["background","top_cheese","other_cheese"], model_path="logs/cheese/top_and_other/best_epoch_weights.pth")
    img_utils = ImageUtils()

    image = Image.open("test_image.png")
    mask = Cheese_UNet.detect_image(image)
    # mask.save("image_mask.png")
    # print(np.array(mask).shape)
    top_layer_mask = Cheese_UNet.get_top_layer(mask, [250, 106, 77])
    # top_layer_mask.show("Top Layer")

    binary_mask = Image.fromarray(
        img_utils.binarize_image(masked_img=np.array(top_layer_mask))
    )

    binary_mask_edges, cont = img_utils.find_edges_in_binary_image(np.array(binary_mask))
    # print(cont)
    center = img_utils.get_contour_center(cont)
    # draw center
    cv2.circle(binary_mask_edges, center, 2, (255, 255, 255), 1)

    binary_mask_edges = Image.fromarray(binary_mask_edges)
    # binary_mask_edges.show("top layer edges")
    binary_mask_edges = binary_mask_edges.convert('RGB')
    binary_mask_edges.save("top_layer_edges_center.png")

    
