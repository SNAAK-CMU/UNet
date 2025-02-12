import os
import random

import numpy as np
from PIL import Image
from tqdm import tqdm

#------------------------------------------------------#
# If you want to add a test set, modify trainval_percent
# Modify train_percent to change the ratio of the validation set to 9:1
#
# Currently, the library uses the test set as a verification set and does not divide the test set separately.
#------------------------------------------------------#
trainval_percent    = 1.0 # the test set has been pre separated, if not, change this to 0.95 ~ 5% test set
train_percent       = 0.80
#-------------------------------------------------------#
# Point to the folder where the VOC data set is located
# Default points to the VOC data set in the root directory
#-------------------------------------------------------#
VOCdevkit_path      = '/home/snaak/Documents/datasets/cheese/training_sets'

if __name__ == "__main__":
    random.seed(0)
    print("Generate txt in ImageSets.")
    segfilepath     = os.path.join(VOCdevkit_path, 'masks')
    saveBasePath    = os.path.join(VOCdevkit_path, 'Segment_config')
    
    temp_seg = os.listdir(segfilepath)
    total_seg = []
    for seg in temp_seg:
        if seg.endswith(".png"):
            total_seg.append(seg)

    num     = len(total_seg)  
    list    = range(num)  
    tv      = int(num*trainval_percent)  
    tr      = int(tv*train_percent)  
    trainval= random.sample(list,tv)  
    train   = random.sample(trainval,tr)  
    
    print("train and val size",tv)
    print("train size",tr)
    ftrainval   = open(os.path.join(saveBasePath,'trainval.txt'), 'w')  
    ftest       = open(os.path.join(saveBasePath,'test.txt'), 'w')  
    ftrain      = open(os.path.join(saveBasePath,'train.txt'), 'w')  
    fval        = open(os.path.join(saveBasePath,'val.txt'), 'w')  
    
    for i in list:  
        name = total_seg[i][:-4]+'\n'  
        if i in trainval:  
            ftrainval.write(name)  
            if i in train:  
                ftrain.write(name)  
            else:  
                fval.write(name)  
        else:  
            ftest.write(name)  
    
    ftrainval.close()  
    ftrain.close()  
    fval.close()  
    ftest.close()
    print("Generate txt in ImageSets done.")

    print("Checking the dataset format, this may take a while...")
    classes_nums        = np.zeros([256], int)
    for i in tqdm(list):
        name            = total_seg[i]
        png_file_name   = os.path.join(segfilepath, name)
        if not os.path.exists(png_file_name):
            raise ValueError("The label image %s was not detected. Please check whether the file exists in the specific path and whether the suffix is ​​png."%(png_file_name))
        
        png             = np.array(Image.open(png_file_name), np.uint8)
        if len(np.shape(png)) > 2:
            print("The shape of the label image %s is %s, which is not a grayscale image or an eight-bit color image. Please check the data set format carefully."%(name, str(np.shape(png))))
            print("The label image needs to be a grayscale image or an eight-bit color image. The value of each pixel of the label is the type to which the pixel belongs."%(name, str(np.shape(png))))

        classes_nums += np.bincount(np.reshape(png, [-1]), minlength=256)
            
    print("Print the value and number of pixels.")
    print('-' * 37)
    print("| %15s | %15s |"%("Key", "Value"))
    print('-' * 37)
    for i in range(256):
        if classes_nums[i] > 0:
            print("| %15s | %15s |"%(str(i), str(classes_nums[i])))
            print('-' * 37)
    
    if classes_nums[255] > 0 and classes_nums[0] > 0 and np.sum(classes_nums[1:255]) == 0:
        print("It is detected that the pixel values ​​in the label only contain 0 and 255, and the data format is incorrect.")
        print("For a two-class classification problem, the label needs to be modified so that the pixel value of the background is 0 and the pixel value of the target is 1.")
    elif classes_nums[0] > 0 and np.sum(classes_nums[1:]) == 0:
        print("It is detected that the label only contains background pixels, and the data format is incorrect. Please check the data set format carefully.")

    print("The pictures in Jpeg images should be .jpg files, and the pictures in segmentation class should be .png files.")
    print("If the format is wrong, please refer to:")
    print("https://github.com/bubbliiiing/segmentation-format-fix")