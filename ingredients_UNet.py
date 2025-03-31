#!/usr/bin/python3

from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm

import sys
import os

# # Get the absolute path of the parent directory
current_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
    
# # Add the parent directory to sys.path
sys.path.append(current_dir)

from unet import Unet
# from post_processing.image_utlis import ImageUtils
from img_utils import ImageUtils

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
    Cheese_UNet = Ingredients_UNet(count=False, classes=["background","top_cheese","other_cheese"], model_path="logs/cheese/high_res/best_epoch_weights.pth", mix_type=1)
    img_utils = ImageUtils()

    load_directory = "/home/snaak/Documents/datasets/cheese/SNAAK_all_ingredient_test/imgs/"
    save_directory = "/home/snaak/Documents/datasets/cheese/SNAAK_all_ingredient_test/test_masks/"

    img_names = os.listdir(load_directory)
    for img_name in tqdm(img_names):
        if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            image_path  = os.path.join(load_directory, img_name)
            image       = Image.open(image_path)
            r_image     = Cheese_UNet.detect_image(image)
            top_layer_mask = Cheese_UNet.get_top_layer(r_image, [250, 106, 77])
            binary_mask = Image.fromarray(img_utils.binarize_image(masked_img=np.array(top_layer_mask)))
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)
            # r_image.save(os.path.join(save_directory, img_name))
            binary_mask.save(os.path.join(save_directory, img_name))
    

    # binary_mask_edges, cont = img_utils.find_edges_in_binary_image(np.array(binary_mask))
    # # print(cont)
    # center = img_utils.get_contour_center(cont)
    # # draw center
    # cv2.circle(binary_mask_edges, center, 2, (255, 255, 255), 1)

    # binary_mask_edges = Image.fromarray(binary_mask_edges)
    # # binary_mask_edges.show("top layer edges")
    # binary_mask_edges = binary_mask_edges.convert('RGB')
    # binary_mask_edges.save("top_layer_edges_center.png")

    
