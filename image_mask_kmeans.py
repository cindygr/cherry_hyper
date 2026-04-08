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
from skimage.color import label2rgb
import json as json
from os import listdir, mkdir, chdir, getcwd
from os.path import exists, isdir
from numpy_process_test_V2 import make_rgb
import imageio as imageio
from magic_numbers import HyperSpectralCherryNumbers


def read_and_cluster_hyper(fname, n_clusters):
    """ Read in the data, cluster the pixels 
    @fname - name of data file to cluster
    @n_clusters - number of clusters """

    # Read in the data file
    data = np.load(fname)

    # Remove the mean from each channel
    data_normalized = whiten(data).astype(np.double)
    # Do the actual clustering
    centers = kmeans(data_normalized, n_clusters)
    # Get the ids for each row in the data - returns a list of cluster ids (0, 1, 2,..)
    ids = vq(data_normalized, centers[0])

    # This gets the means of the clusters
    centers_unwhitened = np.zeros((n_clusters, data.shape[1]))
    for indx in range(0, n_clusters):
        b_cur_id = ids[0] == indx
        data_with_id = data[b_cur_id, :]
        data_avg_rows = data_with_id.mean(axis=0)
        centers_unwhitened[indx, :] = data_avg_rows

    center_mean = np.average(centers_unwhitened, axis=1)
    index_sort = center_mean.argsort()
    centers_unwhitened = centers_unwhitened[index_sort, :]
    return centers_unwhitened


def plot_centers(centers):
    nrows = int(np.sqrt(len(centers)) + 1)
    ncols = 1 + len(centers) // nrows
    fig, axs = plt.subplots(nrows, ncols)

    mn = HyperSpectralCherryNumbers()
    xs = np.array(range(0, mn.n_spectral())) + mn.clip_range[0]
    if centers[0].shape[0] != mn.n_spectral():
        xs = np.array(range(0, centers[0].shape[0]))

    for indx, center in enumerate(centers):
        row = indx // ncols
        col = indx % ncols
        axs[row, col].set_ylim(0, 2.5)
        axs[row, col].plot([mn.clip_range[0], mn.clip_range[0]], [0, 2.25], '--k')
        axs[row, col].plot([mn.clip_range[1], mn.clip_range[1]], [0, 2.25], '--k')
        axs[row, col].plot([mn.blue_channel, mn.blue_channel], [0, 2.25], ':b')
        axs[row, col].plot([mn.red_channel, mn.red_channel], [0, 2.25], ':r')
        axs[row, col].plot([mn.green_channel, mn.green_channel], [0, 2.25], '-g')
        axs[row, col].plot([mn.green_range[0], mn.green_range[0], mn.green_range[1], mn.green_range[1]], [0.1, mn.green_max, mn.green_max, 0.1], '-g')
        axs[row, col].plot(xs, center)

        axs[row, col].plot([mn.red_nir_split, mn.red_nir_split, mn.nir_plateau], [mn.red_max, mn.nir_max, mn.nir_max], ':r')

        green_val = center[mn.green_channel - mn.clip_range[0]]
        blue_val = center[mn.blue_channel - mn.clip_range[0]]
        red_val = center[mn.red_channel - mn.clip_range[0]]
        red_nir_ratio = center[mn.nir_plateau - mn.clip_range[0]] / center[mn.red_nir_split - mn.clip_range[0]]
        lum = np.mean(center[:mn.red_nir_split - mn.clip_range[0]])
        axs[row, col].set_title(f"G-R {green_val/red_val:0.2f} G-B {green_val/blue_val:0.2f} L {lum:0.2f} RNIR {red_nir_ratio:0.2f}")

    fig.tight_layout()
    plt.show()


def process_one_image(data_base_name, dest_base_name, clusters):
    """ Make two output files: One is a list of clusters (size of n pixels) the other is an image colored by cluster"""

    data_name = data_base_name + "_flattened.npy"
    map_name = data_base_name + "_map.json"
    clusters_name = dest_base_name + "_clusters.json"
    img_name = dest_base_name + "_clusters.png"
    data_flattened = np.load(data_name)

    # This maps the index in the numpy file to a pixel
    map = []
    with open(map_name, "r") as f:
        map = json.load(f)
    
    # Get the ids for the flattened data
    ids = vq(data_flattened, clusters)

    # Visualization - color each pixel by its cluster value
    im_label = np.zeros((512, 512))  # Blank image
    ids_list = []
    counts = [0] * len(clusters)
    for pix, id in zip(map, ids[0]):
        im_label[pix[0], pix[1]] = id + 1
        ids_list.append((pix[0], pix[1], int(id)))
        counts[id] += 1
    
    n_x_pix = 512 // len(clusters) - 1
    x_indx = 0
    for id in range(0, len(clusters)):
        im_label[x_indx:x_indx+n_x_pix, 0:15] = id
        x_indx += n_x_pix

    # Turn the labeled image into a color one
    im_rgb = label2rgb(im_label)

    # The usual fix to make the image come out the right way
    im_rgb = np.transpose(im_rgb, axes=[1,0,2])
    im_rgb = np.flip(im_rgb, axis=1)
    im_rgb = (im_rgb * 255.0).astype(np.uint8)

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


def loop_all_data(source_dir, dest_dir, centers):
    # Make the output director
    if not exists(dest_dir):
        mkdir(dest_dir)

    for fname in listdir(source_dir):
        if fname.endswith(".npy") and "flattened" in fname:
            # get name without the _flattened.npy
            base_name = fname[0:-14]

            process_one_image(data_base_name=source_dir + base_name, dest_base_name=dest_dir + base_name, clusters=centers)


def make_classification_directories(source_dir, dest_dir):
    magic_numbers = HyperSpectralCherryNumbers()
    
    # Make the output director
    if not exists(dest_dir):
        mkdir(dest_dir)
        mkdir(dest_dir + "Infected")
        mkdir(dest_dir + "Not infected")

    for fname in listdir(source_dir):
        if not fname.endswith(".json"):
            continue

        with open(source_dir + fname, "r") as f:
            data = json.load(f)

            unique_id = fname.split("_")
            plant_id = int(unique_id[0][1:])
            if plant_id in magic_numbers.infected:
                fname_out = dest_dir + "Infected/" + fname[0:-14] + ".npy"
            else:
                fname_out = dest_dir + "Not infected/" + fname[0:-14] + ".npy"
            signature = data["Signature"]
            np.save(fname_out, np.array(signature))


if __name__ == '__main__':
    dir = getcwd()    
    if "millarn" in dir:
        source_data_dir = "/Users/millarn/VSCode/data/cherry/numpy_flattened_arrays/"
        source_mask_dir = "/Users/millarn/VSCode/data/cherry/numpy_flattened_arrays/"
        source_dir = "/Users/millarn/VSCode/data/cherry/numpy_flattened_arrays/"
        dest_dir = "/Users/millarn/VSCode/data/cherry/clusters/"
        dest_signature_dir = "/Users/millarn/VSCode/data/cherry/signatures/"
    else:
        source_data_dir = "/Users/millarn/VSCode/data/cherry/numpy/"
        source_mask_dir = "/Users/millarn/VSCode/data/cherry/numpy/"
        source_dir = "/Users/cindygrimm/VSCode/data/cherry/flattened/"
        dest_dir = "/Users/cindygrimm/VSCode/data/cherry/clusters/"
        dest_signature_dir = "/Users/cindygrimm/VSCode/data/cherry/signatures/"

    # Where the one flattend file used for building the clusters is
    fname = source_dir + "sample_12_50000.npy"
    n_clusters = 7
    centers = read_and_cluster_hyper(fname, n_clusters=n_clusters)

    fname_save_clusters = f"{fname[:-4]}_{n_clusters}.npy"
    np.save(fname_save_clusters, centers)

    plot_centers(centers=centers)

    loop_all_data(source_dir=source_dir, dest_dir=dest_dir, centers=centers)

    make_classification_directories(source_dir=dest_dir, dest_dir=dest_signature_dir)
    print("done")
