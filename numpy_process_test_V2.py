import numpy as np
import matplotlib.pyplot as plt
from os import listdir, mkdir, chdir, getcwd
from os.path import exists, isdir
import json as json
import imageio as imageio
import spectral as spy
from skimage.morphology import area_closing


def make_rgb(data, mask, lower_bds=(0, 0), upper_bds=(512, 512)):
    # pix_mid_ix = (lower_bds[0] + upper_bds[0]) // 2
    # pix_mid_iy = (lower_bds[1] + upper_bds[1]) // 2
    # while not mask[pix_mid_ix, pix_mid_iy]:
    #     pix_mid_iy += 1
    #     pix_mid_ix += 1
    # for band in range(0, data.shape[2], 10):
    #     print(f"{band}, {data[pix_mid_ix, pix_mid_iy, band]}")

    blue_band = 30
    green_band = 50
    red_band =  90

    blue_range = 0.3 # np.max(data[lower_bds[0]:upper_bds[0], lower_bds[1]:upper_bds[1], 20])
    green_range = 0.4 # np.max(data[lower_bds[0]:upper_bds[0], lower_bds[1]:upper_bds[1], 52])
    red_range = 0.3 # np.max(data[lower_bds[0]:upper_bds[0], lower_bds[1]:upper_bds[1], 120])

    im = np.zeros((data.shape[0], data.shape[1], 3), dtype=np.float16)
    im[:, :, 0] = 0.5 * data[:, :, blue_band] / blue_range
    im[:, :, 1] = 0.5 * data[:, :, green_band] / green_range
    im[:, :, 2] = 0.5 * data[:, :, red_band] / red_range

    for channel in range(0, 3):
        im[im[:, :, channel] > 0.5, channel] = 0.5

    im[mask, :] *= 2.0
    im_rgb = (im * 255.0).astype(np.uint8)
    rgb = np.transpose(im_rgb, axes=[1,0,2])
    rgb = np.flip(rgb, axis=1)
    return rgb

def apply_boolean(source_dir, dest_dir):
   
    # Make the output director
    if not exists(dest_dir):
        mkdir(dest_dir)

    for fname in listdir(source_dir):
        # Use for reading/writing
        full_fname = source_dir + fname

        # Unique id - use for naming new files
        fname_parts = fname.split(".")[0].split('_')
        if len(fname_parts) < 4:
            print(f"Skipping {fname}, not a valid file name")
            continue
        unique_id = fname[0:-4]

        # Load data and use boolean logic to select plant pixels
        data = np.load(full_fname)
        data_avg_lum = np.mean(data[:, :, 10:92], axis=2) # find mean of bands 10-92
        data_bool_less_than_white = data_avg_lum < 0.45  # avg less than 0.45 should remove white objects
        data_bool_greater_than_black = data_avg_lum > 0.075  # avg greater than 0.075 should remove black objects
        data_bool_brightness = np.logical_and(data_bool_less_than_white, data_bool_greater_than_black) # limit to pixels between black and white limit
        data_bool_slope = (data[:,:,123]/data[:,:,92])>4  # green pixels often have a steep slope between the red edge (92) and the NIR (123)
        data_bool_greater_than_red = data[:,:,52] > 1.2 * data[:,:,20] 
        data_bool_greater_than_blue = data[:,:,52] > 1.3 * data[:,:,67]
        data_bool = np.logical_and(np.logical_and(data_bool_greater_than_red, data_bool_greater_than_blue, data_bool_slope), data_bool_less_than_white, data_bool_greater_than_black)
        data_bool_bright_and_slope = np.logical_and(data_bool_brightness, data_bool_slope)
        data_bool_green_bump = np.logical_and(data_bool_greater_than_red, data_bool_greater_than_blue)
        data_bool = np.logical_and(data_bool_green_bump, data_bool_bright_and_slope)

        mask_pixs = np.where(data_bool)
        upper_right = [0, 0]
        lower_left = [data_bool.shape[0], data_bool.shape[1]]
        for pix in mask_pixs[0]:
            lower_left[0] = min(lower_left[0], pix)
            upper_right[0] = max(upper_right[0], pix)
        for pix in mask_pixs[1]:
            lower_left[1] = min(lower_left[1], pix)
            upper_right[1] = max(upper_right[1], pix)

        mask_cleaned_up = area_closing(data_bool)
        # Save the mask as a boolean numpy
        np.save(dest_dir + unique_id + "_limited.npy", mask_cleaned_up)

        rgb = make_rgb(data=data, mask=mask_cleaned_up, lower_bds=lower_left, upper_bds=upper_right)
        # Save the image as a mask
        imageio.imwrite(dest_dir + unique_id + "_mask.png", rgb.astype(np.uint8))


if __name__ == '__main__':
    
    dir = getcwd()
    if "millarn" in dir:
        source_dir = "/Users/millarn/VSCode/data/cherry/numpy/"
        dest_dir = "/Users/millarn/VSCode/data/cherry/numpy_output/"
    else:
        source_dir = "/Users/cindygrimm/VSCode/data/cherry/numpy/"
        dest_dir = "/Users/cindygrimm/VSCode/data/cherry/masked/"

    apply_boolean(source_dir, dest_dir)

