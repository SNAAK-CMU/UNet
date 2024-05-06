# to make multi-channel masks into single channel masks
import numpy as np
#import cupy as np
from PIL import Image
from labelme import utils
import os
from numba import jit, cuda
from tqdm import tqdm


#filepath = "img_03.png"
#savepath = "img_03_sc.png"

#@jit(target_backend='cuda')
def get_mod_mask(npa):
    if npa.ndim == 3:
        mod_img = np.zeros([np.shape(npa)[0], np.shape(npa)[1]])
        for height in range(npa.shape[0]):
            for width in range(npa.shape[1]):
                mask_channels = npa[height][width]
                #print(mask_channels)
                for class_ID in range(mask_channels.shape[0]):
                    if not mask_channels[2] == 0:
                        mod_img[height][width] = 1
                    elif not mask_channels[1] == 0 and mask_channels[2] == 0:
                        mod_img[height][width] = 2
                    else:
                        mod_img[height][width] == 0
                    
        return mod_img
    elif npa.ndim == 2:
        # its already a single channel image
        return npa
    else:
        return None
            

def process_masks(load_folderpath, save_folderpath):
    multichannel_masks = os.listdir(load_folderpath)
    
    print("There are "+str(len(multichannel_masks))+" masks to convert. Processing:")

    for filepath in tqdm(multichannel_masks, total=len(multichannel_masks)):
        image = Image.open(load_folderpath+filepath)
        npa = np.array(image)
        mod_img = get_mod_mask(npa)
        savepath = save_folderpath+filepath
        utils.lblsave(savepath, mod_img)    
        
    print("Saved masks to", save_folderpath)
            
if __name__ == "__main__":
    
    load_folderpath = "test_images/Thailand Project/white_white/results/multichannel_masks/"
    save_folderpath = "test_images/Thailand Project/white_white/results/masks/"

    process_masks(load_folderpath=load_folderpath, save_folderpath=save_folderpath)   
    
    
        
