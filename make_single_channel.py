# to make multi-channel masks into single channel masks
import numpy as np
#import cupy as np
from PIL import Image
from labelme import utils
import os
from tqdm import tqdm
import multiprocessing


def get_mod_mask(npa):
    mask_color_type_1=[255, 106, 77] # top cheese color
    mask_color_type_2=[250, 250, 55] # other cheese color

    if npa.ndim == 3:
        mod_img = np.zeros([np.shape(npa)[0], np.shape(npa)[1]])
        for height in range(npa.shape[0]):
            for width in range(npa.shape[1]):
                # in each pixel
                mask_channels = npa[height][width] # get the RGB channels
                for _ in range(mask_channels.shape[0]):
                    # making the value of each pixel of the image is the type to which the pixel belongs. here I check the color of the pixel and assign it a value accordingly
                    if (mask_channels == mask_color_type_1).all():
                        mod_img[height][width] = 1
                    elif (mask_channels == mask_color_type_2).all():
                        mod_img[height][width] = 2
                    # elif (mask_channels == mask_color_type_3).all():
                    #     # mod_img[height][width] = 3
                    else:
                        mod_img[height][width] = 0
        return mod_img
            

def process_masks_multithread(load_folderpath, save_folderpath):
    # TODO process masks in parallel but ensure that the order is maintained and names do not change
    pass

def process_single_mask(filepath, load_folderpath, save_folderpath):
    image = Image.open(load_folderpath+filepath)
    npa = np.array(image)
    mod_img = get_mod_mask(npa)
    savepath = save_folderpath+filepath
    utils.lblsave(savepath, mod_img)
    print("Saved mask to", savepath)

def process_masks(load_folderpath, save_folderpath):
    multichannel_mask_names = os.listdir(load_folderpath)
    
    print("There are "+str(len(multichannel_mask_names))+" to convert. Processing:")
    # load all masks
    for filepath in tqdm(multichannel_mask_names, total=len(multichannel_mask_names)):
        image = Image.open(load_folderpath+filepath)
        npa = np.array(image)
        mod_img = get_mod_mask(npa)
        savepath = save_folderpath+filepath
        utils.lblsave(savepath, mod_img)
    
    print("Saved masks to", save_folderpath)

            
if __name__ == "__main__":
    
    load_folderpath = "/home/snaak/Documents/datasets/cheese/multiingredient_cheese_pickup/multichannel_masks/"
    save_folderpath = "/home/snaak/Documents/datasets/cheese/multiingredient_cheese_pickup/masks/"

    process_masks(load_folderpath=load_folderpath, save_folderpath=save_folderpath)   
    
        
