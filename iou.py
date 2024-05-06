# calculate the Intersection Over Union (IoU) score for the segmented images

import os
import tqdm
import numpy as np
from PIL import Image

ground_truth_dir_path = "test_images/Thailand Project/white_white/results/ground_truth/"

predicted_masks_dir_path = "test_images/Thailand Project/white_white/results/masks/"

ground_truth_masks = sorted(os.listdir(ground_truth_dir_path))

predicted_masks = sorted(os.listdir(predicted_masks_dir_path))

if not len(ground_truth_masks) == len(predicted_masks):
    print("Make sure the number of ground truth masks and predicted masks are the same!")
else:
    num_imgs = len(ground_truth_masks)

for index in range(num_imgs):
    ground_truth_mask = np.array(Image.open(ground_truth_dir_path+ground_truth_masks[index]))
    predicted_mask = np.array(Image.open(predicted_masks_dir_path+predicted_masks[index]))
    #print("Dimensions of Masks. Ground Truth:", ground_truth_mask.ndim, "Predicted Mask:", predicted_mask.ndim)
    if ground_truth_mask.shape[0] == predicted_mask.shape[0] and ground_truth_mask.shape[1] == predicted_mask.shape[1]:
        tp=0
        fp=0
        for i in range(ground_truth_mask.shape[0]):
            for j in range(ground_truth_mask.shape[1]):
                if ground_truth_mask[i, j] == predicted_mask[i, j]:
                    tp = tp+1
                else:
                    fp = fp+1
        iou = tp / (tp + fp)
        print("IoU of mask ", ground_truth_masks[index], iou)
    else:
        print("Ground Truth and Predicted Masks need to be of the same dimension")
