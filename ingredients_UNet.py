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
        self.img_utils = ImageUtils()

    def get_top_layer(self, image, top_layer_rgb):
        mask = np.array(self.detect_image(image)) # TODO: change this in parent class to assign class ID to pixels instead of RGB
        mod_img = np.zeros(
            [np.shape(mask)[0], np.shape(mask)[1], np.shape(mask)[2]]
        )
        if mask.ndim == 3:
            for height in range(mask.shape[0]):
                for width in range(mask.shape[1]):
                    if (mask[height][width] == top_layer_rgb).all(): 
                      mod_img[height][width] = mask[height][width][0]
        return Image.fromarray(np.uint8(mod_img))

    def get_top_layer_binary(self, image, top_layer_rgb):
        top_layer_mask = np.array(self.get_top_layer(image, top_layer_rgb))
        binary_mask = Image.fromarray(self.img_utils.binarize_image(masked_img=np.array(top_layer_mask)))
        # find contour with max area
        contours, _ = cv2.findContours(np.array(binary_mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            print("No contours found")
            # return black image
            binary_mask = np.zeros_like(np.array(binary_mask))
            max_contour_binary_mask = np.zeros_like(np.array(binary_mask))
            return binary_mask, max_contour_binary_mask
            
        max_contour = max(contours, key=cv2.contourArea)
        
        # create a mask for the largest contour
        max_contour_mask = np.zeros_like(np.array(binary_mask))
        cv2.drawContours(max_contour_mask, [max_contour], -1, (255), thickness=cv2.FILLED)
        # create a binary mask
        max_contour_binary_mask = np.zeros_like(np.array(binary_mask))
        max_contour_binary_mask[max_contour_mask == 255] = 255
        
        
        return binary_mask, max_contour_binary_mask

if __name__ == "__main__":
    # Test initialisation for cheese
    Cheese_UNet = Ingredients_UNet(count=False, classes=["background","top_cheese","other_cheese"], model_path="logs/cheese/cheese_check/best_epoch_weights.pth", mix_type=1, num_classes=3)  
    #Ham_UNet = Ingredients_UNet(count=False, classes=["background","top_ham","other_ham"], model_path="logs/ham/multiingredient_bologna/best_epoch_weights.pth", mix_type=0, num_classes=5)        
    img_utils = ImageUtils()

    # for directory
    load_directory = "/home/snaak/Documents/data/testsets/CHE_images_041425_1_T/"
    save_directory = "/home/snaak/Documents/data/testsets/CHE_images_041425_1_T/cheese_check_model/pred_masks/"
    binary_save_directory = "/home/snaak/Documents/data/CHE_images_041425/cheese_check_model/pred_binary_masks/"
    
    img_names = os.listdir(load_directory)
    for img_name in tqdm(img_names):
        if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            image_path  = os.path.join(load_directory, img_name)
            image       = Image.open(image_path)
            #print("Opened Image:", image_path)
            #r_image     = Ham_UNet.detect_image(image)
            r_image     = Cheese_UNet.detect_image(image)

            # top_layer_mask = Cheese_UNet.get_top_layer(np.array(r_image), [250, 250, 55])
            # binary_mask = Image.fromarray(img_utils.binarize_image(masked_img=np.array(top_layer_mask)))
            # binary_mask = Cheese_UNet.get_top_layer_binary(image, [250, 250, 55])
            #binary_mask, max_contour_binary_mask = Ham_UNet.get_top_layer_binary(image, [61, 61, 245])
            binary_mask, max_contour_binary_mask = Cheese_UNet.get_top_layer_binary(image, [250, 250, 55])
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)
            # r_image.save(os.path.join(save_directory, img_name))
            r_image.save(os.path.join(save_directory, img_name))
            if not os.path.exists(binary_save_directory):
                os.makedirs(binary_save_directory)
            if max_contour_binary_mask is not None:
                max_contour_binary_mask = Image.fromarray(max_contour_binary_mask)
                max_contour_binary_mask.save(os.path.join(binary_save_directory, img_name))
            else:
                # save black image
                max_contour_binary_mask = Image.fromarray(np.zeros_like(np.array(image)))
                max_contour_binary_mask.save(os.path.join(binary_save_directory, img_name))


    # for image
    # image = Image.open("/home/snaak/Documents/manipulation_ws/src/snaak_vision/src/segmentation/cheese_input_image.jpg")
    # # r_image     = Cheese_UNet.detect_image(image)

    # # top_layer_mask = Cheese_UNet.get_top_layer(np.array(r_image), [250, 250, 55])
    # # binary_mask = Image.fromarray(img_utils.binarize_image(masked_img=np.array(top_layer_mask)))
    # binary_mask, max_contour_binary_mask = Cheese_UNet.get_top_layer_binary(image, [250, 250, 55])
    # binary_mask.show("Binary Top Layer Mask")

    # binary_mask_edges, cont = img_utils.find_edges_in_binary_image(np.array(binary_mask))
    # # print(cont)
    # center = img_utils.get_contour_center(cont)
    # # draw center
    # cv2.circle(binary_mask_edges, center, 2, (255, 255, 255), 1)

    # binary_mask_edges = Image.fromarray(binary_mask_edges)
    # # binary_mask_edges.show("top layer edges")
    # binary_mask_edges = binary_mask_edges.convert('RGB')
    # binary_mask_edges.save("top_layer_edges_center.png")

    
