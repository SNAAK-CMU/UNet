# To make multi-channel masks into single-channel masks
import numpy as np
from PIL import Image
from labelme import utils
import os
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool


def get_mod_mask(npa, mask_color_type_1=None, mask_color_type_2=None):
    """
    Converts a multi-channel mask into a single-channel mask based on specified colors.
    """
    if npa.ndim == 3:
        # Create an empty single-channel mask
        mod_img = np.zeros((npa.shape[0], npa.shape[1]), dtype=np.uint8)

        # Vectorized comparison for mask_color_type_1
        mask_1 = np.all(npa == mask_color_type_1, axis=-1)
        mod_img[mask_1] = 3

        # Vectorized comparison for mask_color_type_2
        mask_2 = np.all(npa == mask_color_type_2, axis=-1)
        mod_img[mask_2] = 4

        # Any other pixel remains 0
    elif npa.ndim == 2:
        # If the image is already single channel
        mod_img = npa
    else:
        raise ValueError("Image is not 2D or 3D. Please check the image format.")
    
    return mod_img


def process_single_mask(filepath, load_folderpath, save_folderpath, mask_color_type_1, mask_color_type_2):
    """
    Processes a single mask file: converts it to single-channel and saves it.
    """
    try:
        savepath = os.path.join(save_folderpath, filepath)
        if os.path.exists(savepath):
            print(f"File {savepath} already exists. Skipping.")
            return

        image = Image.open(os.path.join(load_folderpath, filepath))
        npa = np.array(image)
        mod_img = get_mod_mask(npa, mask_color_type_1, mask_color_type_2)
        utils.lblsave(savepath, mod_img)
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

def process_wrapper(args):
    """
    Wrapper function for multiprocessing to handle arguments.
    """
    filepath, load_folderpath, save_folderpath, mask_color_type_1, mask_color_type_2 = args
    process_single_mask(filepath, load_folderpath, save_folderpath, mask_color_type_1, mask_color_type_2)


def process_masks_multiprocessing(load_folderpath, save_folderpath, mask_color_type_1, mask_color_type_2):
    """
    Processes all masks in the folder using multiprocessing for faster execution.
    """
    # Create the save folder if it doesn't exist
    if not os.path.exists(save_folderpath):
        os.makedirs(save_folderpath)
        print(f"Created directory: {save_folderpath}")
    else:
        print(f"Directory exists: {save_folderpath}")

    # Get the list of all multichannel mask names
    multichannel_mask_names = os.listdir(load_folderpath)
    print(f"There are {len(multichannel_mask_names)} masks to convert. Processing:")

    # Prepare arguments for the wrapper function
    args = [
        (filepath, load_folderpath, save_folderpath, mask_color_type_1, mask_color_type_2)
        for filepath in multichannel_mask_names
    ]

    # Use multiprocessing Pool
    with Pool(processes=multiprocessing.cpu_count() - 10) as pool:  # Leave 2 cores free
        list(tqdm(pool.imap(process_wrapper, args), total=len(multichannel_mask_names)))

    print(f"Saved masks to {save_folderpath}")


def printimg(im):
    """
    Prints detailed information about an image, including its shape, dtype, and unique values.
    """
    print("Image shape: ", im.shape)
    print("Image dtype: ", im.dtype)
    print("Image min value: ", np.min(im))
    print("Image max value: ", np.max(im))
    print("dimension of each value: ", im[0, 0].shape if im.ndim == 3 else "N/A")
    print("total number of pixels: ", im.size)

    # Check if the image is RGB or BGR
    if im.ndim == 3:  # For RGB/BGR images
        if (im[0, 0, 0] > im[0, 0, 2]):  # Compare the first pixel's Red and Blue channels
            print("The image is likely in BGR format.")
        else:
            print("The image is likely in RGB format.")
    
    # Print unique values
    if im.ndim == 3:  # For RGB images
        unique_colors = np.unique(im.reshape(-1, im.shape[2]), axis=0)
        print("Unique colors in the image (RGB):")
        print(unique_colors)
    else:  # For single-channel images
        unique_values = np.unique(im)
        print("Unique values in the image (single channel):")
        print(unique_values)

if __name__ == "__main__":
    # Input and output folder paths
    load_folderpath = "/home/snaak/Documents/datasets/bologna/multiingredient_bologna_kiosk/og_color_masks"
    save_folderpath = "/home/snaak/Documents/datasets/bologna/multiingredient_bologna_kiosk/og_png_class_masks"

    # Define mask colors
    mask_color_type_1 = [61, 61, 245]  # Top bologna color
    mask_color_type_2 = [64, 188, 240]  # Other bologna color

    # for sandwich check images
    # mask_color_type_2=[250, 50, 83] # top cheese color - augment first then convert to single channel
    # mask_color_type_1=[61, 61, 245] # other cheese color - augment first then convert to single channel
    
    # for ingredient pickup images
#     mask_color_type_1=[255, 106, 77] # top cheese color - augment first then convert to single channel
# +   mask_color_type_2=[250, 250, 55] # other cheese color - augment first then convert to single channel


    # Test pixel values
    # Uncomment to test with a specific image
    # test_image_name = "randbc1_006667.png"
    # test_image_path = os.path.join(load_folderpath, test_image_name)
    # print("Test image path: ", test_image_path)
    # test_image = Image.open(test_image_path)
    # test_image = np.array(test_image)
    # printimg(test_image)

    # Process all masks using multiprocessing
    process_masks_multiprocessing(
        load_folderpath=load_folderpath,
        save_folderpath=save_folderpath,
        mask_color_type_1=mask_color_type_1,
        mask_color_type_2=mask_color_type_2
    )