#!/usr/bin/env python3

# This assignment introduces you to a common task (creating a segmentation mask) and a common tool (kmeans) for
#  doing clustering of data and the difference in color spaces.

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
    return centers_unwhitened, ids


def plot_centers(centers):
    nrows = 3
    ncols = 6
    fig, axs = plt.subplots(nrows, ncols)

    for indx, center in enumerate(centers):
        row = indx // ncols
        col = indx % ncols
        axs[row, col].plot(center)
        axs[row, col].set_ylim(0, 2.5)
    plt.show()


def process_one_image(data_base_name, dest_base_name, clusters):
    """ Make two output files: One is a list of clusters (size of n pixels) the other is an image colored by cluster"""

    data_name = data_base_name + "_flattened.npy"
    map_name = data_base_name + "_map.json"
    clusters_name = dest_base_name + "_clusters.json"
    signature_name = dest_base_name + "_signature.json"
    img_name = dest_base_name + "_clusters.png"
    data_flattened = np.load(data_name)
    map = []
    with open(map_name, "r") as f:
        map = json.load(f)
    
    # Get the ids for the flattened data
    ids = vq(data_flattened, clusters)

    # Want this to be a color map
    im_label = np.zeros((512, 512))
    ids_list = []
    counts = [0] * len(clusters)
    for pix, id in zip(map, ids[0]):
        im_label[pix[0], pix[1]] = id + 1
        ids_list.append((pix[0], pix[1], int(id)))
        counts[id] += 1
    im_rgb = label2rgb(im_label)

    im_rgb = np.transpose(im_rgb, axes=[1,0,2])
    im_rgb = np.flip(im_rgb, axis=1)
    im_rgb = (im_rgb * 255.0).astype(np.uint8)

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

            ids, im_rgb = process_one_image(data_base_name=source_dir + base_name, dest_base_name=dest_dir + base_name, clusters=centers)


    plt.imshow(im_rgb)
    plt.show()

if __name__ == '__main__':
    dir = getcwd()    
    if "millarn" in dir:
        source_data_dir = "/Users/millarn/VSCode/data/cherry/numpy_flattened_arrays/"
        source_mask_dir = "/Users/millarn/VSCode/data/cherry/numpy_flattened_arrays/"
        source_dir = "/Users/millarn/VSCode/data/cherry/numpy_flattened_arrays/"
        dest_dir = "/Users/millarn/VSCode/data/cherry/clusters/"
    else:
        source_data_dir = "/Users/millarn/VSCode/data/cherry/numpy/"
        source_mask_dir = "/Users/millarn/VSCode/data/cherry/numpy/"
        source_dir = "/Users/cindygrimm/VSCode/data/cherry/flattened/"
        dest_dir = "/Users/cindygrimm/VSCode/data/cherry/clusters/"

    fname = source_dir + "sample_12.npy"
    n_clusters = 5
    centers, ids = read_and_cluster_hyper(fname, n_clusters=n_clusters)

    fname_save_clusters = f"{fname[:-4]}_{n_clusters}.npy"
    np.save(fname_save_clusters, centers)

    # plot_centers(centers=centers)

    loop_all_data(source_dir=source_dir, dest_dir=dest_dir, centers=centers)
    print("done")
