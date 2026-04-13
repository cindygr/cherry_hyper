import numpy as np
import matplotlib.pyplot as plt
from os import listdir, mkdir, chdir, getcwd
from os.path import exists, isdir
import json as json
import imageio as imageio
from magic_numbers import HyperSpectralCherryNumbers


# Make one big n pixels X n channels file for each image
def flatten_arrays(source_dir_orig, source_dir_masked, dest_dir, normalize=None):
   
    # Make the output director
    if not exists(dest_dir):
        mkdir(dest_dir)

    mn = HyperSpectralCherryNumbers()

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
       
        # Load data - original data plus mask
        data = np.load(full_data_fname)
        mask = np.load(full_mask_name)
        
        #  Get number of pixels
        num_pixels = np.count_nonzero(mask)

        # Make empty array to hold flattened data
        flattened_data = np.zeros((num_pixels, mn.n_spectral()), dtype=np.float32)

        # Make map of pixel locations where mask is True
        map_name =   fname[0:-4] + "_map.json"
        map = []

        # Loop through each pixel and add to flattened array
        pixel_index = 0
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if mask[i, j] == True:
                    flattened_data[pixel_index] = data[i, j, mn.clip_range[0]: mn.clip_range[1]]  # add bands 20-154 to flattened array
                    if normalize == "Integral":
                        sum_all = np.sum(flattened_data[pixel_index])
                        if np.isclose(sum_all, 0.0):
                            sum_all = 1.0
                        flattened_data[pixel_index] = flattened_data[pixel_index] / sum_all
                    elif normalize == "Random":
                        sum_all = np.mean(flattened_data[pixel_index])
                        if np.isclose(sum_all, 0.0):
                            sum_all = 1.0
                        flattened_data[pixel_index, 0:15] = np.random.uniform(0.0, 1.0, 15)
                        flattened_data[pixel_index, 0] = sum_all
                    elif normalize == "Features":
                        sum_all = np.mean(flattened_data[pixel_index])
                        blue_val = np.min(data[i, j, mn.clip_range[0]:mn.green_range[0]])
                        green_val = np.max(data[i, j, mn.green_range[0]:mn.green_range[1]])
                        red_val = np.min(data[i, j, mn.green_range[1]:mn.nir_plateau])
                        nir_val = data[i, j, mn.nir_plateau]
                        uv_val = np.max(data[i, j, mn.clip_range[0]:mn.red_channel])

                        if np.isclose(sum_all, 0.0):
                            print(f"Image {fname} pixel {i} {j} {np.mean(data[i, j, :])}")
                            sum_all = 1.0
                        if np.isclose(green_val, 0.0):
                            green_val = 1.0
                        flattened_data[pixel_index, 0] = sum_all
                        flattened_data[pixel_index, 1] = blue_val / green_val
                        flattened_data[pixel_index, 2] = red_val / green_val
                        flattened_data[pixel_index, 3] = nir_val / green_val
                        flattened_data[pixel_index, 4] = uv_val / green_val
                        flattened_data[pixel_index, 5] = green_val
                    map.append((i, j))  # save original pixel location
                    pixel_index += 1
        if normalize == "Random":
            flattened_data = flattened_data[:, 0:15]
        elif normalize == "Features":
            flattened_data = flattened_data[:, 0:6]
        np.save(dest_dir + unique_id + "_flattened.npy", flattened_data)

        with open(dest_dir + map_name, 'w') as f:
            json.dump(map, f)


if __name__ == '__main__':
    dir = getcwd()
    feature = "_integral"
    str_feature = "Integral"
    if "millarn" in dir:
        source_data_dir = "/Users/millarn/VSCode/data/cherry/numpy/"
        source_mask_dir = "/Users/millarn/VSCode/data/cherry/numpy_output/"
        dest_dir = "/Users/millarn/VSCode/data/cherry/numpy_flattened_arrays/"
    else:
        source_data_dir = "/Users/cindygrimm/VSCode/data/cherry/numpy/"
        source_mask_dir = "/Users/cindygrimm/VSCode/data/cherry/masked/"
        dest_dir = f"/Users/cindygrimm/VSCode/data/cherry/flattened{feature}/"
    
    all_datadir = "/Users/millarn/VSCode/data/cherry/"
    numpy_data_output_dir = "/Users/millarn/VSCode/data/cherry/numpy_flattened_arrays/"
    # Integral Random
    #flatten_arrays(source_dir_orig=source_data_dir, source_dir_masked=source_mask_dir, dest_dir=dest_dir, normalize="Features")
    flatten_arrays(source_dir_orig=source_data_dir, source_dir_masked=source_mask_dir, dest_dir=dest_dir, normalize=str_feature)
