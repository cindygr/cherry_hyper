#!/usr/bin/env python3

# This assignment introduces you to a common task (creating a segmentation mask) and a common tool (kmeans) for
#  doing clustering of data and the difference in color spaces.

# There are no shortage of kmeans implementations out there - using scipy's
import numpy as np
from scipy.cluster.vq import kmeans, vq, whiten

# Using imageio to read in the images and skimage to do the color conversion
import skimage
import imageio
from skimage.color import rgb2hsv
import matplotlib.pyplot as plt
import json as json
from os import listdir, mkdir, chdir, getcwd
from os.path import exists, isdir



def read_and_cluster_image(image_name, use_hsv, n_clusters):
    """ Read in the image, cluster the pixels by color (either rgb or hsv), then
    draw the clusters as an image mask, colored by both a random color and the center
    color of the cluster
    @image_name - name of image in Data
    @use_hsv - use hsv, y/n
    @n_clusters - number of clusters (up to 6)"""

    # Read in the file
    im_orig = imageio.imread("Data/" + image_name)
    # Make sure you just have rgb (for those images with an alpha channel)
    im_orig = im_orig[:, :, 0:3]

    # The plot to put the images in
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    # Make name for the image from the input parameters
    str_im_name = image_name.split('.')[0] + " "
    if use_hsv:
        str_im_name += "HSV"
    else:
        str_im_name += "RGB"

    str_im_name += f", k={n_clusters}"

    # This is how you draw an image in a matplotlib figure
    axs[0].imshow(im_orig)
    # This sets the title
    axs[0].set_title(str_im_name)

    # GUIDE
    # Step 1: If use_hsv is true, convert the image to hsv (see skimage rgb2hsv - skimage has a ton of these
    #  conversion routines)
    # Step 2: reshape the data to be an nx3 matrix
    #   kmeans assumes each row is a data point. So you have to give it a (widthXheight) X 3 matrix, not the image
    #   data as-is (WXHX3). See numpy reshape.
    # Step 3: Whiten the data
    # Step 4: Call kmeans with the whitened data to get out the centers
    #   Note: kmeans returns a tuple with the centers in the first part and the overall fit in the second
    # Step 5: Get the ids out using vq
    #   This also returns a tuple; the ids for each pixel are in the first part
    #   You might find the syntax data[ids == i, 0:3] = rgb_color[i] useful - this gets all the data elements
    #     with ids with value i and sets them to the color in rgb_color
    # Step 5: Create a mask image, and set the colors by rgb_color[ id for pixel ]
    # Step 6: Create a second mask image, setting the color to be the average color of the cluster
    #    Two ways to do this
    #       1) "undo" the whitening step on the returned cluster (harder)
    #       2) Calculate the means of the clusters in the original data
    #           np.mean(data[ids == c])
    #
    # Note: To do the HSV option, get the RGB version to work. Then go back and do the HSV one
    #   Simplest way to do this: Copy the code you did before and re-do after converting to hsv first
    #     Don't forget to take the color centers in the *original* image, not the hsv one
    #     Don't forget to rename your variables
    #   More complicated: Make a function. Most of the code is the same, except for a conversion to hsv at the beginning

    # An array of some default color values to use for making the rgb mask image
    rgb_color = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255], [255, 0, 255]]
    # BEGIN SOLUTION
    if use_hsv:
        im = rgb2hsv(im_orig)
    else:
        im = im_orig

    data_orig = im_orig.reshape((im.shape[0] * im.shape[1], im.shape[2]))
    data = im.reshape((im.shape[0] * im.shape[1], im.shape[2]))

    data_normalized = whiten(data)
    centers = kmeans(data_normalized, n_clusters)
    ids = vq(data_normalized, centers[0])
    im_clusters_rgb = np.zeros((im_orig.shape[0] * im_orig.shape[1], 3), dtype=im_orig.dtype)
    im_clusters = np.zeros((im_orig.shape[0] * im_orig.shape[1], 3), dtype=im_orig.dtype)

    for c in range(0, n_clusters):
        center_color = np.mean(data_orig[ids[0] == c, 0:3], axis=0)
        im_clusters[ids[0] == c, 0:3] = center_color
        im_clusters_rgb[ids[0] == c, 0:3] = rgb_color[c]

    im_clusters = im_clusters.reshape((im_orig.shape[0], im_orig.shape[1], 3))
    im_clusters_rgb = im_clusters_rgb.reshape((im_orig.shape[0], im_orig.shape[1], 3))

    axs[1].imshow(im_clusters_rgb)
    axs[2].imshow(im_clusters)
    # END SOLUTION
    axs[1].set_title("ID colored by rgb")
    axs[2].set_title("ID colored by cluster average")

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

    centers_unwhitened = np.zeros((n_clusters, data.shape[1]))
    for indx in range(0, n_clusters):
        b_cur_id = ids[0] == indx
        data_with_id = data[b_cur_id, :]
        data_avg_rows = data_with_id.mean(axis=0)
        centers_unwhitened[indx, :] = data_avg_rows
    # Call unwhiten here to make the centers be back where they were
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


def process_one_image(base_name, clusters):
    """ Make two output files: One is a list of clusters (size of n pixels) the other is an image colored by cluster"""

    data_name = base_name + "_flattened.npy"
    map_name = base_name + "_map.json"
    clusters_name = base_name + "_clusters.json"
    img_name = base_name + "_clusters.png"
    data_flattened = np.load(data_name)
    map = []
    with open(map_name, "r") as f:
        map = json.load(f)
    
    # Need to fix this
    ids = vq(data_flattened, clusters)

    # Want this to be a color map
    cols = [np.array([255, 0, 0]), np.array([125, 125, 0]), np.array([0, 255, 0])]
    im_rgb = np.zeros((512, 512, 3))
    for pix, id in zip(map, ids[0]):
        im_rgb[pix[0], pix[1], :] = cols[id].transpose()

    plt.imshow(im_rgb)
    plt.show()
    imageio.imwrite(img_name) 


def loop_all_data(source_dir, centers):
    for fname in listdir(source_dir):
        if fname.endswith(".npy") and "flattened" in fname:
            # get name without the _flattened.npy
            base_name = fname[0:-14]
            process_one_image(base_name=source_dir +base_name, clusters=centers)


if __name__ == '__main__':
    dir = getcwd()
    if "millarn" in dir:
        source_dir = "/Users/millarn/VSCode/data/cherry/numpy/"
        dest_dir = "/Users/millarn/VSCode/data/cherry/numpy_output/"
    else:
        source_dir = "/Users/cindygrimm/VSCode/data/cherry/numpy/"
        dest_dir = "/Users/cindygrimm/VSCode/data/cherry/masked/"

    fname = "/Users/millarn/VSCode/data/cherry/numpy_random_samples/sample_0.npy"
    n_clusters = 3
    centers, ids = read_and_cluster_hyper(fname, n_clusters=n_clusters)
   # plot_centers(centers=centers)
    loop_all_data(source_dir="/Users/millarn/VSCode/data/cherry/numpy_flattened_arrays/", centers=centers)
    print("done")
