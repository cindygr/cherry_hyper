import numpy as np
import matplotlib.pyplot as plt
from os import listdir, mkdir, chdir, getcwd
from os.path import exists, isdir
import json as json
import imageio as imageio
import spectral as spy
from skimage.morphology import area_closing
from magic_numbers import HyperSpectralCherryNumbers
from sklearn.linear_model import LinearRegression


# Render the original spectral image with the masked pixels colored by their labels
def make_rgb(data, mask, lower_bds=(0, 0), upper_bds=(512, 512)):
    magic_numbers = HyperSpectralCherryNumbers()
    # pix_mid_ix = (lower_bds[0] + upper_bds[0]) // 2
    # pix_mid_iy = (lower_bds[1] + upper_bds[1]) // 2
    # while not mask[pix_mid_ix, pix_mid_iy]:
    #     pix_mid_iy += 1
    #     pix_mid_ix += 1
    # for band in range(0, data.shape[2], 10):
    #     print(f"{band}, {data[pix_mid_ix, pix_mid_iy, band]}")

    blue_band = magic_numbers.blue_channel
    green_band = magic_numbers.green_channel
    red_band =  magic_numbers.blue_channel

    blue_range = 0.25 # np.max(data[lower_bds[0]:upper_bds[0], lower_bds[1]:upper_bds[1], 20])
    green_range = magic_numbers.green_max
    red_range = 0.35 # np.max(data[lower_bds[0]:upper_bds[0], lower_bds[1]:upper_bds[1], 120])


    im = np.zeros((data.shape[0], data.shape[1], 3), dtype=np.float16)
    perc_color = 0.6
    im[:, :, 0] = perc_color * data[:, :, blue_band] / blue_range
    im[:, :, 1] = perc_color * data[:, :, green_band] / green_range
    im[:, :, 2] = perc_color * data[:, :, red_band] / red_range

    for channel in range(0, 3):
        im[im[:, :, channel] > perc_color, channel] = perc_color

    im[mask, :] *= 1.0 / perc_color
    im[mask == False, 0] = 0.0

    im_both = np.zeros((2 * data.shape[0], data.shape[1], 3), dtype=np.float16)
    im_both[:data.shape[0], :, :] = im
    im_mask = mask.astype(np.float16)
    im_both[data.shape[0]:, :, 0] = im_mask
    im_both[data.shape[0]:, :, 1] = im_mask
    im_both[data.shape[0]:, :, 2] = im_mask
    im_rgb = (im_both * 255.0).astype(np.uint8)
    rgb = np.transpose(im_rgb, axes=[1,0,2])
    rgb = np.flip(rgb, axis=1)
    return rgb


def make_mask_image(mask_bright, mask_green_greater_blue, mask_green_greater_red, masK_slope, mask_green, mask):
        
    _, axs = plt.subplots(2, 3, figsize=(16, 16))
    axs[0, 0].imshow(mask_bright)
    axs[0, 0].set_title("Brightness")
    axs[0, 1].imshow(mask_green_greater_blue)
    axs[0, 1].set_title("Greater than blue")
    axs[0, 2].imshow(mask_green_greater_red)
    axs[0, 2].set_title("Greater than red")
    axs[1, 0].imshow(masK_slope)
    axs[1, 0].set_title("Slope")
    axs[1, 1].imshow(mask_green)
    axs[1, 1].set_title("Green Bump")

    axs[1, 2].imshow(mask)
    axs[1, 2].set_title("All")
    plt.axis("off")
    plt.show()


def make_mask(source_dir, dest_dir):
   
    # Make the output director
    if not exists(dest_dir):
        mkdir(dest_dir)

    mn = HyperSpectralCherryNumbers()
    for fname in listdir(source_dir):
        # Use for reading/writing
        full_fname = source_dir + fname

        # Unique id - use for naming new files
        fname_parts = fname.split(".")[0].split('_')
        if len(fname_parts) < 4:
            print(f"Skipping {fname}, not a valid file name")
            continue
        unique_id = fname[0:-4]

        # Load data and use boolean logic to select plant pixels
        #  The primary difference between leaf pixels and other ones is that the other spectrums are "flat"        
        #    - no bump at the green and no slope between red and near infra red
        data = np.load(full_fname)
        
        # If the sd of the whole spectrum is small, then it's really flat
        data_avg_sd = np.std(data[:, :, :], axis=2) # find mean of non near infra red bands

        # Use the near infra red/red ratio to trim out some background pixels
        data_avg_nir = data[:, :, mn.nir_plateau]  # Near infra red average
        data_avg_red = data[:, :, mn.red_nir_split] # Red average
        data_nir_red_ratio = data_avg_nir / data_avg_red

        data_avg_lf = np.zeros((data.shape[0], data.shape[1]))
        xs = np.arange(0, data.shape[2], step=1)
        for ix in range(0, data_avg_lf.shape[0]):
            for iy in range(0, data_avg_lf.shape[1]):
                # if the standard deviation is small, we know it's flat - skip
                if data_avg_sd[ix, iy] < mn.lum_sd_clip:
                    continue
                # Similarly, a nir/red ratio too small is also close to a line - skip
                if data_nir_red_ratio[ix, iy] < mn.ratio_nir_to_red:
                    continue

                # Only do the line fit if the pixel might be a leaf (saves a lot of computation)
                m, b = np.polyfit(xs, np.squeeze(data[ix, iy, :]), 1)
                ys = xs * m + b
                diffs = ys - data[ix, iy, :]
                diff = np.linalg.norm(diffs[mn.clip_range[0]:mn.clip_range[1]])
                data_avg_lf[ix, iy] = diff

        mask_is_line = data_avg_lf < 1.25   # Line fit error
        mask = np.logical_not(mask_is_line)

        mask_pixs = np.where(mask)
        upper_right = [0, 0]
        lower_left = [mask.shape[0], mask.shape[1]]
        for pix in mask_pixs[0]:
            lower_left[0] = min(lower_left[0], pix)
            upper_right[0] = max(upper_right[0], pix)
        for pix in mask_pixs[1]:
            lower_left[1] = min(lower_left[1], pix)
            upper_right[1] = max(upper_right[1], pix)

        mask_cleaned_up = area_closing(mask)
        # Save the mask as a boolean numpy
        np.save(dest_dir + unique_id + "_limited.npy", mask_cleaned_up)

        rgb = make_rgb(data=data, mask=mask_cleaned_up, lower_bds=lower_left, upper_bds=upper_right)
        # Save the image as a mask
        imageio.imwrite(dest_dir + unique_id + "_mask.png", rgb.astype(np.uint8))


if __name__ == '__main__':
    
    dir = getcwd()
    if "millarn" in dir:
        source_dir = "/Users/millarn/VSCode/data/cherry/numpy/"
        dest_dir = "/Users/millarn/VSCode/data/cherry/numpy_output/"
    else:
        source_dir = "/Users/cindygrimm/VSCode/data/cherry/numpy/"
        dest_dir = "/Users/cindygrimm/VSCode/data/cherry/masked/"

    make_mask(source_dir, dest_dir)

