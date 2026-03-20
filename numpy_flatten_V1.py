import numpy as np
import matplotlib.pyplot as plt
from os import listdir, mkdir, chdir, getcwd
from os.path import exists, isdir
import xmltodict
import json as json
import imageio as imageio

source_dir = "/Users/millarn/VSCode/data/cherry/numpy_output/"
dest_dir = "/Users/millarn/VSCode/data/cherry/numpy_flattened_arrays/"

def flatten_arrays(source_dir, dest_dir):
   
    # Make the output director
    if not exists(dest_dir):
        mkdir(dest_dir)

    numpy_data_dir = source_dir + "numpy/"
    mask_data_dir = source_dir + "numpy_output/"
    for fname in listdir(numpy_data_dir):
        # Use for reading/writing
        unique_id = fname[0:-4]
        full_data_fname = numpy_data_dir + fname
        full_mask_name =  mask_data_dir + fname[0:-4] + "_limited.npy"

        # Unique id - use for naming new files
        fname_parts = fname.split("_")
        if len(fname_parts) < 4:
            print(f"Skipping {fname}, not a valid file name")
            continue
       
        # Load data 
        data = np.load(full_data_fname)
        mask = np.load(full_mask_name)
        
        #  Get number of pixels
        num_pixels = np.count_nonzero(mask)

        # Make empty array to hold flattened data
        flattened_data = np.zeros((num_pixels,134), dtype=np.float16)

        # Loop through each pixel and add to flattened array
        pixel_index = 0
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if mask[i, j] == True:
                    flattened_data[pixel_index] = data[i, j, 20:154]  # add bands 20-154 to flattened array
                    pixel_index += 1
        np.save(dest_dir + unique_id + "_flattened.npy", flattened_data)


if __name__ == '__main__':
    
    all_datadir = "/Users/millarn/VSCode/data/cherry/"
    numpy_data_output_dir = "/Users/millarn/VSCode/data/cherry/numpy_flattened_arrays/"
    flatten_arrays(all_datadir, numpy_data_output_dir)
