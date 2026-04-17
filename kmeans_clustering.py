#!/usr/bin/env python3

# This file produces a set of clusters to form the "words" in the supervised clustering step
# The input data is of the following form (one for each image):
#.  Numpy arrays (n valid pixels, clipped hyperspectral data)
# Magic numbers: Number of clusters, clip of 

# There are no shortage of kmeans implementations out there - using scipy's
import numpy as np
from scipy.cluster.vq import kmeans, vq, whiten
from skimage.color import rgb2hsv
import matplotlib.pyplot as plt
from skimage import color as skimage_color
import json as json
from os import listdir, mkdir, chdir, getcwd
from os.path import exists, isdir
from create_masks import make_rgb
import imageio as imageio
from magic_numbers import HyperSpectralCherryNumbers


def read_and_cluster_hyper(fname, n_clusters):
    """ Read in the data, cluster the pixels 
    @fname - name of data file to cluster
    @n_clusters - number of clusters """

    # Read in the data file
    data = np.load(fname)

    # Remove the mean from each channel and scale by sd
    data_mean = np.mean(data, axis=0)
    data_normalized = np.copy(data)
    for ix in range(0, data.shape[1]):
        data_normalized[:, ix] = data_normalized[:, ix] - data_mean[ix]
    data_sd = np.std(data_normalized, axis=0)
    for ix in range(0, data.shape[1]):
        data_normalized[:, ix] = data_normalized[:, ix] / data_sd[ix]

    # Do the actual clustering
    centers = kmeans(data_normalized, n_clusters)
    # Get the ids for each row in the data - returns a list of cluster ids (0, 1, 2,..)
    ids = vq(data_normalized, centers[0])

    # Sort the clusters by luninance in the original data set
    centers_unwhitened = np.zeros((n_clusters, data.shape[1]))
    for ix in range(0, data.shape[1]):
        centers_unwhitened[:, ix] = centers[0][:, ix] * data_sd[ix] + data_mean[ix]

    if data_normalized.shape[1] > 15:
        # The original multispectral data
        center_mean = np.average(centers_unwhitened, axis=1)
    else:
        # Either features or random - use the luminance channel to sort
        center_mean = centers_unwhitened[:, 0]
    index_sort = center_mean.argsort()
    centers_sorted = centers[0][index_sort, :]
    centers_unwhitened = centers_unwhitened[index_sort, :]
    return centers_sorted, data_mean, data_sd, centers_unwhitened


def plot_centers(centers, y_scl=2.25):
    nrows = int(np.sqrt(len(centers)) + 1)
    ncols = 1 + len(centers) // nrows
    fig, axs = plt.subplots(nrows, ncols)

    mn = HyperSpectralCherryNumbers()
    xs = np.array(range(0, mn.n_spectral())) + mn.clip_range[0]
    if centers.shape[1] != mn.n_spectral():
        xs = np.array(range(0, centers.shape[1]))

    y_max = 2.25
    lims = np.max(centers, axis=0)

    for indx, center in enumerate(centers):
        row = indx // ncols
        col = indx % ncols
        if len(center) > 20:
            axs[row, col].set_ylim(0, y_max)
            axs[row, col].plot([mn.clip_range[0], mn.clip_range[0]], [0, y_max], '--k')
            axs[row, col].plot([mn.clip_range[1], mn.clip_range[1]], [0, y_max], '--k')
            axs[row, col].plot([mn.blue_channel, mn.blue_channel], [0, y_max], ':b')
            axs[row, col].plot([mn.red_channel, mn.red_channel], [0, y_max], ':r')
            axs[row, col].plot([mn.green_channel, mn.green_channel], [0, y_max], '-g')
            axs[row, col].plot([mn.green_range[0], mn.green_range[0], mn.green_range[1], mn.green_range[1]], [0.1, mn.green_max, mn.green_max, 0.1], '-g')
            axs[row, col].plot(xs, center * y_scl)

            axs[row, col].plot([mn.red_nir_split, mn.red_nir_split, mn.nir_plateau], [mn.red_max, mn.nir_max, mn.nir_max], ':r')

            green_val = center[mn.green_channel - mn.clip_range[0]]
            blue_val = center[mn.blue_channel - mn.clip_range[0]]
            red_val = center[mn.red_channel - mn.clip_range[0]]
            red_nir_ratio = center[mn.nir_plateau - mn.clip_range[0]] / center[mn.red_nir_split - mn.clip_range[0]]
            lum = np.mean(center[:mn.red_nir_split - mn.clip_range[0]])
            lum_sd = np.std(center)
            axs[row, col].set_title(f"G-R {green_val/red_val:0.2f} G-B {green_val/blue_val:0.2f} \nL {lum:0.2f} SD {lum_sd:0.2f} RNIR {red_nir_ratio:0.2f}")
        else:
            if len(center) == 6:
                plot_labels = ["L", "G/B", "G/R", "G/NIR", "G/UV", "G"]
                copy_center = np.copy(center)
                copy_center[3] /= lims[3]
                axs[row, col].set_ylim(0, 1.5)
                axs[row, col].bar(plot_labels, copy_center)
            else:
                axs[row, col].plot(center)
                axs[row, col].set_title(f"Lum {center[0]}")

    fig.tight_layout()
    plt.show()


def process_one_image(data_base_name, dest_base_name, clusters, data_mean, data_sd):
    """ Make two output files: One is a list of clusters (size of n pixels) the other is an image colored by cluster"""

    data_name = data_base_name + "_flattened.npy"
    map_name = data_base_name + "_map.json"
    clusters_name = dest_base_name + "_clusters.json"
    img_name = dest_base_name + "_clusters.png"
    data_flattened = np.load(data_name)

    # Apply the same whitening as we did the original data
    for ix in range(0, data_flattened.shape[1]):
        data_flattened[:, ix] = (data_flattened[:, ix] - data_mean[ix]) / data_sd[ix]

    # This maps the index in the numpy file to a pixel
    map = []
    with open(map_name, "r") as f:
        map = json.load(f)
    
    # Get the ids for the flattened data
    ids = vq(data_flattened, clusters)

    n_clusters = clusters.shape[0]

    # Turn the labeled image into a color one
    # custom color for each non-zero label.
    colors = np.array(
        [
            [125, 0, 125],  
            [0, 0, 255],  
            [0, 125, 125],  
            [0, 255, 0],  
            [125, 125, 0],  
            [255, 0, 0],  
            [255, 255, 255],  
        ],
        dtype=np.uint8,
        )
    while colors.shape[0] < n_clusters:
        if 2 * colors.shape[0] < n_clusters:
            new_colors = np.zeros((2 * colors.shape[0] - 1, 3), dtype=np.uint8)
            for indx in range(0, colors.shape[0]):
                new_colors[2 * indx, :] = colors[indx, :]
            for indx in range(0, colors.shape[0] - 1):
                indx_prev = indx * 2
                indx_next = indx * 2 + 2
                new_colors[2 * indx + 1, :] = (new_colors[indx_prev, :] // 2 + new_colors[indx_next, :] // 2)
            colors = new_colors
        else:
            new_colors = np.zeros((n_clusters, 3), dtype=np.uint8)
            n_dupl = n_clusters - colors.shape[0]
            for indx in range(1, colors.shape[0]+1):
                new_colors[-indx, :] = colors[-indx, :]
            for indx in range(0, n_dupl):
                new_colors[2 * indx, :] = colors[indx, :]
            for indx in range(0, n_dupl - 1):
                indx_prev = indx * 2
                indx_next = indx * 2 + 2
                new_colors[2 * indx + 1, :] = (new_colors[indx_prev, :] // 2 + new_colors[indx_next, :] // 2)
            colors = new_colors

    # Visualization - color each pixel by its cluster value
    im_rgb = np.zeros((512, 512, 3), dtype=np.uint8)  # Blank image
    ids_list = []
    counts = [0] * n_clusters
    for pix, id in zip(map, ids[0]):
        im_rgb[pix[0], pix[1]] = np.transpose(colors[id, :])
        ids_list.append((pix[0], pix[1], int(id)))
        counts[id] += 1
    
    n_x_pix = im_rgb.shape[0] // n_clusters - 1
    x_indx = im_rgb.shape[0] - 1
    for id in range(0, n_clusters):
        for ix in range(x_indx - n_x_pix, x_indx):
            for iy in range(0, 15):
                im_rgb[ix, iy, :] = np.transpose(colors[id, :])
        x_indx -= n_x_pix

    # The usual fix to make the image come out the right way
    im_rgb = np.transpose(im_rgb, axes=[1,0,2])
    im_rgb = np.flip(im_rgb, axis=1)

    # Create a unique signature for each image by counting the distribution of clusters
    signature = []
    for c in counts:
        signature.append(c / len(ids[0]))
    with open(clusters_name, "w") as f:
        my_dict = {"Ids": ids_list, "Counts": counts, "Signature": signature}
        json.dump(my_dict, f, indent=2)
    print(f"ID {data_base_name} {signature}")

    imageio.imwrite(img_name, im_rgb) 
    
    return ids, im_rgb


def loop_all_data(source_dir, dest_dir, centers, data_mean, data_sd):
    # Make the output director
    if not exists(dest_dir):
        mkdir(dest_dir)

    for fname in listdir(source_dir):
        if fname.endswith(".npy") and "flattened" in fname:
            # get name without the _flattened.npy
            base_name = fname[0:-14]

            process_one_image(data_base_name=source_dir + base_name, dest_base_name=dest_dir + base_name, clusters=centers, data_mean=data_mean, data_sd=data_sd)


def make_classification_directories(source_dir, dest_dir):
    magic_numbers = HyperSpectralCherryNumbers()
    
    # Make the output director
    if not exists(dest_dir):
        mkdir(dest_dir)
        mkdir(dest_dir + "Infected")
        mkdir(dest_dir + "Not infected")

    test_group1 = {4:"P401", 8:"P406", 0:"P157", 7:"P163", 9:"P338"}
    test_group2 = {1:"P20", 2:"P33", 5:"P418", 10:"P421", 6:"P430"}

    for fname in listdir(source_dir):
        if not fname.endswith(".json"):
            continue

        with open(source_dir + fname, "r") as f:
            data = json.load(f)

            unique_id = fname.split("_")
            plant_id = int(unique_id[0][1:])
            last_day = True if "D45" in unique_id[1] else False
            if plant_id in magic_numbers.infected:
                fname_out = dest_dir + "Infected/" + fname[0:-14] + ".npy"
                fname_out2 = dest_dir + "last_day_infected/" + fname[0:-14] + ".npy"
            else:
                fname_out = dest_dir + "Not infected/" + fname[0:-14] + ".npy"
                fname_out2 = dest_dir + "last_day_uninfected/" + fname[0:-14] + ".npy"
            signature = data["Signature"]
            np.save(fname_out, np.array(signature))

            if last_day:
                np.save(fname_out2, np.array(signature))

            if plant_id in test_group1:
                fname_out = dest_dir + "test_group1/" + fname[0:-14] + ".npy"
                fname_out2 = dest_dir + "last_day_test1/" + fname[0:-14] + ".npy"
            else:
                fname_out = dest_dir + "test_group2/" + fname[0:-14] + ".npy"
                fname_out2 = dest_dir + "last_day_test2/" + fname[0:-14] + ".npy"
            signature = data["Signature"]
            np.save(fname_out, np.array(signature))
            if last_day:
                np.save(fname_out2, np.array(signature))


if __name__ == '__main__':
    dir = getcwd()    
    feature = "_random"
    feature = "_integral"
    feature = "_features"
    if "millarn" in dir:
        source_data_dir = "/Users/millarn/VSCode/data/cherry/numpy_flattened_arrays/"
        source_mask_dir = "/Users/millarn/VSCode/data/cherry/numpy_flattened_arrays/"
        source_dir = "/Users/millarn/VSCode/data/cherry/numpy_flattened_arrays/"
        dest_dir = "/Users/millarn/VSCode/data/cherry/clusters/"
        dest_signature_dir = "/Users/millarn/VSCode/data/cherry/signatures/"
    else:
        source_data_dir = "/Users/millarn/VSCode/data/cherry/numpy/"
        source_mask_dir = "/Users/millarn/VSCode/data/cherry/numpy/"
        source_dir = f"/Users/cindygrimm/VSCode/data/cherry/flattened{feature}/"
        dest_dir = "/Users/cindygrimm/VSCode/data/cherry/clusters/"
        dest_signature_dir = "/Users/cindygrimm/VSCode/data/cherry/signatures/"

    # Where the one flattend file used for building the clusters is
    fname = source_dir + f"features12_50000.npy"
    n_clusters = 6
    centers, data_mean, data_sd, centers_unwhitened = read_and_cluster_hyper(fname, n_clusters=n_clusters)

    fname_save_clusters = f"{fname[:-4]}_clusters_{n_clusters}.npy"
    fname_save_unwhitened_clusters = f"{fname[:-4]}_unwhitened_clusters_{n_clusters}.npy"
    center_plus_mean_sd = np.vstack((data_mean, data_sd, centers))
    center_unwhitened_plus_mean_sd = np.vstack((data_mean, data_sd, centers_unwhitened))
    np.save(fname_save_clusters, center_plus_mean_sd)
    np.save(fname_save_unwhitened_clusters, center_unwhitened_plus_mean_sd)

    loop_all_data(source_dir=source_dir, dest_dir=dest_dir, centers=centers, data_mean=data_mean, data_sd=data_sd)

    make_classification_directories(source_dir=dest_dir, dest_dir=dest_signature_dir)

    plot_centers(centers=centers_unwhitened, y_scl=100.0)

    print("done")
