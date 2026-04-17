import numpy as np
import matplotlib.pyplot as plt
from os import listdir, mkdir, chdir, getcwd
from os.path import exists, isdir
import xmltodict
import json as json
import imageio as imageio


def select_sample_pixels(source_dir, dest_dir):

    plant_ids = [1, 4]
    day_ids = [13, 31, 45]
    side_ids = [0, 1]
    ims = []
    for pid in plant_ids:
        for day in day_ids:
            for side in side_ids:
                fname = f"P{pid}_D{day}_S{side}_P"
                for try_name in listdir(source_dir):
                    if not ".npy" in try_name:
                        continue
                    if fname in try_name:
                        ims.append(try_name)
    #im1_name = "P0_D13_S0_P157_11-13_flattened.npy"
    #im2_name = "P0_D13_S0_P157_12-15_flattened.npy"
    #im3_name = "P0_D13_S0_P401_11-13_flattened.npy"
    #im4_name = "P0_D13_S0_P401_12-15_flattened.npy"

    #ims = [im1_name, im2_name, im3_name, im4_name]

    n_total = 50000
    n_per_image = n_total // len(ims)
    index = 0
    all_data = None
    for im in ims:
        data = np.load(source_dir + im)
        num_pixels = data.shape[0]
        selected_indices = np.random.choice(num_pixels, n_per_image, replace=False)
        selected_data = data[selected_indices, :]
        try:
            all_data = np.vstack((all_data, selected_data))
        except:
            all_data = np.copy(selected_data)
        index += 1

    print(f"Mean {np.mean(data, axis=0)}")
    print(f"SD {np.std(data, axis=0)}")
    print(f"Min {np.min(data, axis=0)}")
    print(f"Max {np.max(data, axis=0)}")
    np.save(dest_dir + f"features{index}_{n_total}.npy", all_data)



if __name__ == '__main__':
    dir = getcwd()
    feature = "_integral"
    feature = "_features"
    if "millarn" in dir:
        source_dir = "/Users/millarn/VSCode/data/cherry/numpy_flattened_arrays/"
        dest_dir = "/Users/millarn/VSCode/data/cherry/numpy_flattened_arrays/"
    else:
        source_dir = f"/Users/cindygrimm/VSCode/data/cherry/flattened{feature}/"
        dest_dir = f"/Users/cindygrimm/VSCode/data/cherry/flattened{feature}/"
    
    select_sample_pixels(source_dir, dest_dir)

