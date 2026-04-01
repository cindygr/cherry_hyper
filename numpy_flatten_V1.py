import numpy as np
import matplotlib.pyplot as plt
from os import listdir, mkdir, chdir, getcwd
from os.path import exists, isdir
import xmltodict
import json as json
import imageio as imageio

# Make one big n pixels X n channels file for each image
def flatten_arrays(source_dir_orig, source_dir_masked, dest_dir):
   
    # Make the output director
    if not exists(dest_dir):
        mkdir(dest_dir)

    for fname in listdir(source_dir_orig):
        # Use for reading/writing
        unique_id = fname[0:-4]
        full_data_fname = source_dir_orig + fname
        full_mask_name =  source_dir_masked + fname[0:-4] + "_limited.npy"
        
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

        # Make map of pixel locations where mask is True
        map_name =   fname[0:-4] + "_map.json"
        map = []

        # Loop through each pixel and add to flattened array
        pixel_index = 0
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if mask[i, j] == True:
                    flattened_data[pixel_index] = data[i, j, 20:154]  # add bands 20-154 to flattened array
                    map.append((i, j))  # save original pixel location
                    pixel_index += 1
        np.save(dest_dir + unique_id + "_flattened.npy", flattened_data)

        with open(dest_dir + map_name, 'w') as f:
            json.dump(map, f)


if __name__ == '__main__':
    dir = getcwd()
    if "millarn" in dir:
        source_data_dir = "/Users/millarn/VSCode/data/cherry/numpy/"
        source_mask_dir = "/Users/millarn/VSCode/data/cherry/numpy_output/"
        dest_dir = "/Users/millarn/VSCode/data/cherry/numpy_flattened_arrays/"
    else:
        source_data_dir = "/Users/cindygrimm/VSCode/data/cherry/numpy/"
        source_mask_dir = "/Users/cindygrimm/VSCode/data/cherry/masked/"
        dest_dir = "/Users/cindygrimm/VSCode/data/cherry/flattened/"
    
    all_datadir = "/Users/millarn/VSCode/data/cherry/"
    numpy_data_output_dir = "/Users/millarn/VSCode/data/cherry/numpy_flattened_arrays/"
    flatten_arrays(source_dir_orig=source_data_dir, source_dir_masked=source_mask_dir, dest_dir=dest_dir)
