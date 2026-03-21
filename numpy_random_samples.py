import numpy as np
import matplotlib.pyplot as plt
from os import listdir, mkdir, chdir, getcwd
from os.path import exists, isdir
import xmltodict
import json as json
import imageio as imageio

source_dir = "/Users/millarn/VSCode/data/cherry/numpy_flattened_arrays/"
dest_dir = "/Users/millarn/VSCode/data/cherry/numpy_random_samples/"


def select_sample_pixels(source_dir, dest_dir):

    im1_name = "P0_D13_S0_P157_11-13_flattened.npy"
    im2_name = "P0_D13_S0_P157_12-15_flattened.npy"
    im3_name = "P0_D13_S0_P401_11-13_flattened.npy"
    im4_name = "P0_D13_S0_P157_12-15_flattened.npy"

    ims = [im1_name, im2_name, im3_name, im4_name]

    N_total = 1000
    N_per_image = N_total // len(ims)
    index = 0
    for im in ims:
        data = np.load(source_dir + im)
        num_pixels = data.shape[0]
        selected_indices = np.random.choice(num_pixels, N_per_image, replace=False)
        selected_data = data[selected_indices, :]
        np.save(dest_dir + f"sample_{index}.npy", selected_data)
        index += 1



if __name__ == '__main__':
    
    select_sample_pixels(source_dir, dest_dir)

