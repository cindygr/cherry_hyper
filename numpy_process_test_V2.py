import numpy as np
import matplotlib.pyplot as plt
from os import listdir, mkdir, chdir, getcwd
from os.path import exists, isdir
import xmltodict
import json as json
import imageio as imageio
import spectral as spy

source_dir = "/Users/millarn/VSCode/data/cherry/numpy/"
dest_dir = "/Users/millarn/VSCode/data/cherry/numpy_output/"


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
        np.save(dest_dir + unique_id + "_limited.npy", data_bool)
        im_bool = data_bool.astype(np.uint8)*255
        rgb = np.transpose(im_bool, axes=[1,0])
        rgb = np.flip(rgb, axis=1)
        imageio.imwrite(dest_dir + unique_id + "_mask.png", rgb.astype(np.uint8))

if __name__ == '__main__':
    
    numpy_data_small_dir = "/Users/millarn/VSCode/data/cherry/numpy/"
    numpy_data_output_dir = "/Users/millarn/VSCode/data/cherry/numpy_output/"
    apply_boolean(numpy_data_small_dir, numpy_data_output_dir)

