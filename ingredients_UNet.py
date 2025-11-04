#!/usr/bin/python3

from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm

import sys
import os

# # Get the absolute path of the parent directory
current_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))

# # Add the parent directory to sys.path
sys.path.append(current_dir)

from unet import Unet

# from post_processing.image_utlis import ImageUtils
from img_utils import ImageUtils
from PIL import Image as Im

FOV_WIDTH = 0.775  # metres
FOV_HEIGHT = 0.435  # metres
SW_CHECKER_THRESHOLD = 3  # cm

IMG_WIDTH = 848
IMG_HEIGHT = 480

# Cheese Dimensions in metres
CHEESE_WIDTH = 0.090
CHEESE_HEIGHT = 0.095

# Cheese bin coords
CHEESE_BIN_XMIN = 250
CHEESE_BIN_YMIN = 0
CHEESE_BIN_XMAX = 470
CHEESE_BIN_YMAX = 340

TRAY_BOX_PIX = (
    250,
    20,
    630,
    300,
)  # (x1, y1, x2, y2) coordinates of the tray box in the image

# Ham Dimensions in metres
# 1098 pix/m ; ham_radius = 52 pix
HAM_RADIUS = 0.05  # metres


class Ingredients_UNet(Unet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.img_utils = ImageUtils()

    def get_top_layer(self, image, top_layer_rgb):
        mask = self.detect_image(image)
         # TODO: change this in parent class to assign class ID to pixels instead of RGB
        # mask.save("/home/snaak/Documents/manipulation_ws/src/snaak_vision/src/segmentation/test_raw_mask.jpg")
        mask = np.array(mask)
        mod_img = np.zeros([np.shape(mask)[0], np.shape(mask)[1], np.shape(mask)[2]])
        if mask.ndim == 3:
            for height in range(mask.shape[0]):
                for width in range(mask.shape[1]):
                    if (mask[height][width] == top_layer_rgb).all():
                        mod_img[height][width] = mask[height][width][0]
        return Image.fromarray(np.uint8(mod_img))

    def get_top_layer_binary(self, image, top_layer_rgb):
        # image.save("/home/snaak/Documents/manipulation_ws/src/snaak_vision/src/segmentation/test_source_img.jpg")
        top_layer_mask = self.get_top_layer(image, top_layer_rgb)
        # top_layer_mask.save("/home/snaak/Documents/manipulation_ws/src/snaak_vision/src/segmentation/test_top_layer_mask.jpg")
        top_layer_mask = np.array(top_layer_mask)
        binary_mask = Image.fromarray(
            self.img_utils.binarize_image(masked_img=np.array(top_layer_mask))
        )
        # find contour with max area
        contours, _ = cv2.findContours(
            np.array(binary_mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            print("No contours found")
            # return black image
            binary_mask = np.zeros_like(np.array(binary_mask))
            max_contour_binary_mask = np.zeros_like(np.array(binary_mask))
            return binary_mask, max_contour_binary_mask, 0

        max_contour = max(contours, key=cv2.contourArea)

        # create a mask for the largest contour
        max_contour_mask = np.zeros_like(np.array(binary_mask))
        cv2.drawContours(
            max_contour_mask, [max_contour], -1, (255,), thickness=cv2.FILLED
        )
        # create a binary mask
        max_contour_binary_mask = np.zeros_like(np.array(binary_mask))
        max_contour_binary_mask[max_contour_mask == 255] = 255
        return binary_mask, max_contour_binary_mask, cv2.contourArea(max_contour)


if __name__ == "__main__":

    # pixels to metres
    pixels_to_m = ((FOV_WIDTH / IMG_WIDTH) + (FOV_HEIGHT / IMG_HEIGHT)) / 2

    # ingredient dimensions
    cheese_width = CHEESE_WIDTH
    cheese_height = CHEESE_HEIGHT
    ham_radius = HAM_RADIUS
    cheese_area_pixels = cheese_width * cheese_height * (1 / pixels_to_m**2)
    ham_area_pixels = np.pi * (ham_radius**2) * (1 / pixels_to_m**2)

    # initialise model
    # Cheese_UNet = Ingredients_UNet(
    #     count=False,
    #     classes=["background", "top_cheese", "other_cheese"],
    #     model_path="logs/cheese/UNet_CHE_000/best_epoch_weights.pth",
    #     mix_type=0,
    #     num_classes=3,
    # )
    # Ham_UNet = Ingredients_UNet(
    #     count=False,
    #     classes=["background", "", "", "top_ham", "other_ham"],
    #     model_path="logs/ham/bologna_check/best_epoch_weights.pth",
    #     mix_type=0,
    #     num_classes=5,
    # )
    Bread_UNet = Ingredients_UNet(
        count = False,
        classes = ["background", "top_bread", "other_bread"],
        mix_type = 0,
        num_classes = 3,
        model_path = "logs/bread/UNet_BRE_001/best_epoch_weights.pth"
    )
    # img_utils = ImageUtils()

    # for directory
    load_directory = "/home/snaak/Documents/data/Testsets/SCH_001"
    save_directory = "/home/snaak/Documents/manipulation_ws/src/snaak_vision/src/segmentation/UNet/UNet_BRE_001_Tests/SCH_001_results/"
    # # binary_save_directory = "/home/snaak/Documents/datasets/testsets/BRE_images_090925_T/bread_model/pred_binary_masks/"
    
    # test images in directory
    img_names = os.listdir(load_directory)
    for img_name in tqdm(img_names):
        if img_name.lower().endswith(
            (
                ".bmp",
                ".dib",
                ".png",
                ".jpg",
                ".jpeg",
                ".pbm",
                ".pgm",
                ".ppm",
                ".tif",
                ".tiff",
            )
        ):
            image_path = os.path.join(load_directory, img_name)
            image = Image.open(image_path)
            output = Bread_UNet.detect_image(image)
            # save outputs
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)
            output.save(os.path.join(save_directory, img_name))
    
    # test single image
    
    # image_path = "/home/snaak/Documents/manipulation_ws/src/snaak_vision/src/segmentation/bread_pickup_unet_input_image.jpg"
    # image = Image.open(image_path)
    # top_layer_binary, max_contour_top_layer_binary, max_contour_area = Bread_UNet.get_top_layer_binary(image, [250, 106, 77])

    # test pickup images
    # img_names = os.listdir(load_directory)
    # for img_name in tqdm(img_names):
    #     if img_name.lower().endswith(
    #         (
    #             ".bmp",
    #             ".dib",
    #             ".png",
    #             ".jpg",
    #             ".jpeg",
    #             ".pbm",
    #             ".pgm",
    #             ".ppm",
    #             ".tif",
    #             ".tiff",
    #         )
    #     ):
    #         image_path = os.path.join(load_directory, img_name)
    #         image = Image.open(image_path)
    #         # print("Opened Image:", image_path)
    #         # r_image     = Ham_UNet.detect_image(image)
    #         # r_image     = Cheese_UNet.detect_image(image)

    #         # top_layer_mask = Cheese_UNet.get_top_layer(np.array(r_image), [250, 250, 55])
    #         # binary_mask = Image.fromarray(img_utils.binarize_image(masked_img=np.array(top_layer_mask)))
    #         # binary_mask = Cheese_UNet.get_top_layer_binary(image, [250, 250, 55])
    #         # binary_mask, max_contour_binary_mask = Ham_UNet.get_top_layer_binary(image, [61, 61, 245])
    #         binary_mask, max_contour_binary_mask, max_contour_area = (
    #             Cheese_UNet.get_top_layer_binary(image, [250, 250, 55])
    #         )

    #         if max_contour_area > 1.2 * cheese_area_pixels:
    #             # get_logger().info(
    #             #     f"Cheese area is too large: {max_contour_area} > {cheese_area_pixels}, cropping out bottom 25% of bin and trying again..."
    #             # )

    #             # crop out bottom 33% of the bin
    #             bin_mask = np.zeros_like(np.array(image))
    #             bin_mask[
    #                 CHEESE_BIN_YMIN : CHEESE_BIN_YMAX
    #                 - (CHEESE_BIN_YMAX - CHEESE_BIN_YMIN) // 2,
    #                 CHEESE_BIN_XMIN:CHEESE_BIN_XMAX,
    #             ] = 255
    #             image = cv2.bitwise_and(bin_mask, np.array(image))

    #             mask, max_contour_binary_mask, max_contour_area = (
    #                 Cheese_UNet.get_top_layer_binary(
    #                     Im.fromarray(image), [250, 250, 55]
    #                 )
    #             )

    #             if max_contour_area > 1.2 * cheese_area_pixels:
    #                 # self.get_logger().info(
    #                 #     f"Cheese area is still too large: {max_contour_area} > {cheese_area_pixels}, skipping this image..."
    #                 # )
    #                 # raise Exception(
    #                 #     f"Cheese area after 75% crop is still too large: {max_contour_area} > {self.cheese_area_pixels}"
    #                 # )
    #                 print(
    #                     f"Cheese area after 50% crop is still too large: {max_contour_area} > {cheese_area_pixels}"
    #                 )

    #                 max_contour_binary_mask = np.zeros_like(np.array(image))

    #         # save mask
    #         # if not os.path.exists(save_directory):
    #         #     os.makedirs(save_directory)
    #         # r_image.save(os.path.join(save_directory, img_name))

    #         # save binary mask
    #         if not os.path.exists(binary_save_directory):
    #             os.makedirs(binary_save_directory)
    #         if max_contour_binary_mask is not None:
    #             max_contour_binary_mask = Image.fromarray(max_contour_binary_mask)
    #             max_contour_binary_mask.save(
    #                 os.path.join(binary_save_directory, img_name)
    #             )
    #         else:
    #             # save black image
    #             max_contour_binary_mask = Image.fromarray(
    #                 np.zeros_like(np.array(image))
    #             )
    #             max_contour_binary_mask.save(
    #                 os.path.join(binary_save_directory, img_name)
    #             )

    # # test assembly images
    # img_names = os.listdir(load_directory)
    # for img_name in tqdm(img_names):
    #     if img_name.lower().endswith(
    #         (
    #             ".bmp",
    #             ".dib",
    #             ".png",
    #             ".jpg",
    #             ".jpeg",
    #             ".pbm",
    #             ".pgm",
    #             ".ppm",
    #             ".tif",
    #             ".tiff",
    #         )
    #     ):
    #         image_path = os.path.join(load_directory, img_name)
    #         image = Image.open(image_path)

    #         # assembly_mask = np.zeros_like(np.array(image))
    #         # assembly_mask[
    #         #     TRAY_BOX_PIX[1] : TRAY_BOX_PIX[3], TRAY_BOX_PIX[0] : TRAY_BOX_PIX[2]
    #         # ] = 255
    #         # unet_input_image = cv2.bitwise_and(np.array(image), assembly_mask)
    #         unet_input_image = np.array(image)

    #         # cv2.imshow("unet_input_image", unet_input_image)
    #         # cv2.waitKey(0)
    #         # cv2.destroyAllWindows()

    #         r_image = Cheese_UNet.detect_image(Im.fromarray(unet_input_image))
    #         # r_image = Ham_UNet.detect_image(Im.fromarray(unet_input_image))

    #         # cv2.imshow("r_image", np.array(r_image))
    #         # cv2.waitKey(0)
    #         # cv2.destroyAllWindows()

    #         # Get the cheese mask using UNet
    #         binary_mask, max_contour_binary_mask, max_contour_area = (
    #             Cheese_UNet.get_top_layer_binary(
    #                 Im.fromarray(unet_input_image), [250, 106, 77]
    #             )
    #         )
    #         # binary_mask, max_contour_binary_mask, max_contour_area = Ham_UNet.get_top_layer_binary(
    #         #     Im.fromarray(unet_input_image), [61, 61, 245]
    #         # )

    #         # save overlay image
    #         if not os.path.exists(save_directory):
    #             os.makedirs(save_directory)
    #         r_image.save(os.path.join(save_directory, img_name))


    #         # save binary mask
    #         if not os.path.exists(binary_save_directory):
    #             os.makedirs(binary_save_directory)
    #         if max_contour_binary_mask is not None:
    #             max_contour_binary_mask = Image.fromarray(max_contour_binary_mask)
    #             max_contour_binary_mask.save(
    #                 os.path.join(binary_save_directory, img_name)
    #             )
    #         else:
    #             # save black image
    #             max_contour_binary_mask = Image.fromarray(
    #                 np.zeros_like(np.array(image))
    #             )
    #             max_contour_binary_mask.save(
    #                 os.path.join(binary_save_directory, img_name)
    #             )
