# To draw only the edges from the segmented mask onto the original image

import cv2
import numpy as np
from PIL import Image
from labelme import utils
import os


def get_edges_from_mask(mask):
    edge_mask = np.zeros([np.shape(mask)[0], np.shape(mask)[1]])
    for h in range(mask.shape[0]):
        for w in range(mask.shape[1]):
            # print(mask[i,j])
            if mask[h, w] != 0:
                if (
                    mask[h + 1][w] != mask[h, w]
                    or mask[h - 1][w] != mask[h, w]
                    or mask[h][w + 1] != mask[h, w]
                    or mask[h][w - 1] != mask[h, w]
                ):
                    # print(mask[i, j])
                    edge_mask[h, w] = mask[h, w]
    return edge_mask


def draw_edges(image, edges):
    if image.mode != edges.mode:
        edges = edges.convert(image.mode)
    image = Image.blend(image, edges, 0.5)
    return image


if __name__ == "__main__":

    masks_dir_path = "test_images/Thailand Project/white_white/results/masks/"

    images_dir_path = "test_images/Thailand Project/white_white/results/imgs/"

    blended_image_savepath = (
        "test_images/Thailand Project/white_white/results/blended_images/"
    )

    masks = sorted(os.listdir(masks_dir_path))
    images = sorted(os.listdir(images_dir_path))

    if not len(masks) == len(images):
        print(
            "Make sure the number of ground truth masks and predicted masks are the same!"
        )
    else:
        num_imgs = len(masks)

    for index in range(num_imgs):
        mask = Image.open(masks_dir_path + masks[index])
        image = Image.open(images_dir_path + images[index])

        edges = get_edges_from_mask(np.array(mask))
        utils.lblsave("edges.png", edges)
        edges = Image.open("edges.png")

        blended_image = draw_edges(image, edges)
        cv2.imwrite(blended_image_savepath+images[index], np.array(blended_image))
        
