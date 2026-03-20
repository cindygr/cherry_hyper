import numpy as np
import matplotlib.pyplot as plt
from os import listdir, mkdir, chdir, getcwd
from os.path import exists, isdir
import xmltodict
import json as json
import imageio as imageio

source_dir = "/Users/millarn/VSCode/data/cherry/numpy_small/"
dest_dir = "/Users/millarn/VSCode/data/cherry/numpy_small_output/"

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
        unique_id = fname[0:20]

        # Load data and select all pixels where band 52 is > 0.3
        data = np.load(full_fname)
        #data_bool = data[:,:,52]>0.3 & data[:,:,18]>0.3 & data[:,:,10]>0.3
        data_avg_lum = np.mean(data[:, :, 10:92], axis=2)
        data_bool_less_than_white = data_avg_lum < 0.45
        data_bool_greater_than_black = data_avg_lum > 0.075
        data_bool_brightness = np.logical_and(data_bool_less_than_white, data_bool_greater_than_black)
        data_bool_slope = (data[:,:,123]/data[:,:,92])>4
        data_bool_greater_than_red = data[:,:,52] > 1.2 * data[:,:,20]
        data_bool_greater_than_blue = data[:,:,52] > 1.3 * data[:,:,67]
        data_bool = np.logical_and(np.logical_and(data_bool_greater_than_red, data_bool_greater_than_blue, data_bool_slope), data_bool_less_than_white, data_bool_greater_than_black)
        data_bool_bright_and_slope = np.logical_and(data_bool_brightness, data_bool_slope)
        data_bool_green_bump = np.logical_and(data_bool_greater_than_red, data_bool_greater_than_blue)
        data_bool = np.logical_and(data_bool_green_bump, data_bool_bright_and_slope)
        #data_bool = np.transpose(data_bool, axes=[1,0])
        rgb = np.flip(data_bool, axis=1)

        # Display a quick RGB composite
        # SPECIM cameras often use bands around:
            # R ≈ 60, G ≈ 30, B ≈ 10 (this varies by model!)
    

        fig, axs = plt.subplots(2, 3, figsize=(16, 16))
        axs[0, 0].imshow(data_bool_brightness)
        axs[0, 0].set_title("Brightness")
        axs[0, 1].imshow(data_bool_greater_than_blue)
        axs[0, 1].set_title("Greater than blue")
        axs[0, 2].imshow(data_bool_greater_than_red)
        axs[0, 2].set_title("Greater than red")
        axs[1, 0].imshow(data_bool_slope)
        axs[1, 0].set_title("Slope")
        axs[1, 1].imshow(data_bool_green_bump)
        axs[1, 1].set_title("Green Bump")

        axs[1, 2].imshow(data_bool)
        axs[1, 2].set_title("All")
        plt.axis("off")
        plt.show()


if __name__ == '__main__':
    
    numpy_data_small_dir = "/Users/millarn/VSCode/data/cherry/numpy_small/"
    numpy_data_output_dir = "/Users/millarn/VSCode/data/cherry/numpy_output/"
    apply_boolean(numpy_data_small_dir, numpy_data_output_dir)

