#!/usr/bin/env python3

# This file creates a time series image of the masked plant for each plant, each side, for each color ratio
# Total luminance is mapped to intensity, ratio is mapped to hue

import numpy as np
from skimage.color import hsv2rgb
from magic_numbers import HyperSpectralCherryNumbers
from os import listdir, mkdir, rename, getcwd
from os.path import exists, isdir
from create_masks import make_rgb
import imageio as imageio
import json as json


def get_cutout(mask, im):
    # Find the bounding box for the image
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    pad = 2
    return im[rmin-pad:rmax+pad, cmin-pad:cmax+pad, :]


def get_feature_images(mn : HyperSpectralCherryNumbers, data_flattened, pix_data, f_indx):
    image = np.zeros((mn.im_width, mn.im_height, 3))

    # HSV
    lum_max = 0.0
    ratio_max = mn.max_ratio[f_indx-1]
    ratio_min = mn.min_ratio[f_indx-1]

    image[:, :, 0] = 0.0
    image[:, :, 1] = 0.0
    image[:, :, 2] = 0.0

    data_features = np.zeros((data_flattened.shape[0], 6))
    row_indx = 0
    for pix, data in zip(pix_data, data_flattened):
        feature_data = mn.get_pixel_data(data)
        data_features[row_indx, :] = feature_data
        row_indx = row_indx + 1
        image[pix[0], pix[1], 0] = mn.map_ratio_zero_one(feature_data[f_indx], indx=f_indx-1)
        image[pix[0], pix[1], 2] = feature_data[0]
        image[pix[0], pix[1], 1] = 0.5
        if feature_data[0] > lum_max:
            lum_max = feature_data[0]
        if feature_data[f_indx] > ratio_max:
            ratio_max = feature_data[f_indx]
            print(f"GR max {ratio_max}")

        if feature_data[f_indx] < ratio_min:
            ratio_min = feature_data[f_indx]
            print(f"GR min {ratio_min}")

    image[:, :, 2] = image[:, :, 2] / lum_max
    #print(f"Min {np.min(data_features, axis=0)} SmaxD {np.max(data_features, axis=0)}")
    #print(f"g_r min {np.min(image[:, :, 0])} {np.max(image[:, :, 0])}")
    #print(f"lum min {np.min(image[:, :, 2])} {np.max(image[:, :, 2])}")
    return image
        

def make_image_strip(mn : HyperSpectralCherryNumbers, plant_data_flattened, plant_pix_data, days, name):
    max_width = mn.min_width_cutout
    max_height = mn.min_height_cutout
    n_ratios = 4
    n_imgs = mn.n_images()
    im_strip = np.zeros((n_imgs * max_width, max_height * n_ratios, 3))
    for f_indx in range(1, 5):
        ims_rgb = []
        for data, pix in zip(plant_data_flattened, plant_pix_data):
            im_rgb = get_feature_images(mn=mn, data_flattened=data, pix_data=pix, f_indx=f_indx)

            im_mask = np.sum(im_rgb[:, :, :], axis=2) > 0.0
            im_cutout = get_cutout(im_mask, im_rgb)
            ims_rgb.append(im_cutout)
        
        iy = (f_indx-1) * max_height
        for indx, im in enumerate(ims_rgb):
            width = min(max_width, im.shape[0])
            height = min(max_height, im.shape[1])
            day = days[indx]
            ix = mn.days[day] * max_width
            im_strip[ix:ix + width, iy:iy + height, :] = im[:width, :height, :]

    im_strip_write = np.transpose(im_strip, axes=[1,0,2])
    im_strip_write = np.flip(im_strip_write, axis=1)
    im_strip_write = hsv2rgb(im_strip_write)
    img_name = dest_dir + f"{name}_all.png"
    imageio.imwrite(img_name, (im_strip_write * 255.0).astype(np.uint8)) 
    return im_strip


def im_strip_all(source_dir, dest_dir):
    # Make the output director
    if not exists(dest_dir):
        mkdir(dest_dir)

    mn = HyperSpectralCherryNumbers("ratios")
    n_ratios = 4
    n_imgs = mn.n_images()
    n_plants = 5
    n_sides = 2

    max_height = mn.min_height_cutout
    max_width = mn.min_width_cutout

    other_side_start = max_height * n_ratios * n_plants
    strip_height = max_height * n_ratios

    im_strip_infected = np.zeros((n_imgs * max_width, max_height * (n_ratios * n_plants * n_sides) + 20, 3))
    im_strip_not_infected = np.zeros((n_imgs * max_width, max_height * (n_ratios * n_plants * n_sides), 3))
    ims_strip_infected = []
    ims_strip_not_infected = []
    for _ in range(0, n_ratios):
        ims_strip_infected.append(np.zeros((n_imgs * max_width, max_height * (n_plants * n_sides) + 20, 3)))
        ims_strip_not_infected.append(np.zeros((n_imgs * max_width, max_height * (n_plants * n_sides) + 20, 3)))

    for plant_id in range(0, 10):
        flattened_data = [[], []]
        pix_data = [[], []]
        day_id = [[], []]

        for fname in listdir(source_dir):
            if fname.endswith(".npy") and "flattened" in fname:
                # get name without the _flattened.npy
                base_name = fname[0:-14]
                unique_id = base_name.split("_")
                plant_id_fname = int(unique_id[0][1:])
                if not plant_id == plant_id_fname:
                    continue
                side_id_fname = int(unique_id[2][1])

                f_data = np.load(source_dir + fname)
                flattened_data[side_id_fname].append(f_data)
                map_name = source_dir + base_name + "_map.json"
                map = []
                with open(map_name, "r") as f:
                    map = json.load(f)

                pix_data[side_id_fname].append(map)
                day_id[side_id_fname].append(int(unique_id[1][1:]))

        im_strip_s0 = make_image_strip(mn=mn, plant_data_flattened=flattened_data[0], plant_pix_data=pix_data[0], days=day_id[0], name=f"P{plant_id}_S0")
        im_strip_s1 = make_image_strip(mn=mn, plant_data_flattened=flattened_data[1], plant_pix_data=pix_data[1], days=day_id[1], name=f"P{plant_id}_S0")

        row = 0
        row_f = 0
        im_strip = None
        ims_strip = None
        for k_index, k in enumerate(mn.infected):
            if plant_id == k:
                row = k_index * strip_height
                row_f = k_index * max_height * 2
                im_strip = im_strip_infected
                ims_strip = ims_strip_infected
        for k_index, k in enumerate(mn.not_infected):
            if plant_id == k:
                row = k_index * strip_height
                row_f = k_index * max_height * 2
                im_strip = im_strip_not_infected
                ims_strip = ims_strip_not_infected

        im_strip[:, row:row+strip_height] = im_strip_s0
        row = row + other_side_start
        im_strip[:, row:row+strip_height] = im_strip_s1
        for fid in range(0, 4):
            iy_source = fid * max_height
            im_f = ims_strip[fid]
            # Side 0
            im_f[:, row_f:row_f + max_height] = im_strip_s0[:, iy_source:iy_source + max_height]
            im_f[:, row_f + max_height:row_f + 2 * max_height] = im_strip_s1[:, iy_source:iy_source + max_height]

    # The usual fix to make the image come out the right way
    im_lum = np.zeros((im_strip_infected.shape[0], 20, 3))
    for row_indx in range(0, 10):
        im_lum[:, row_indx, 2] = np.linspace(0, 1.0, im_strip_infected.shape[0])
        im_lum[:, row_indx, 1] = 0.5

        row = row_indx + 10
        im_lum[:, row, 0] = np.linspace(0, 1.0, im_strip_infected.shape[0])
        im_lum[:, row, 1] = 0.5
        im_lum[:, row, 2] = 1.0

    names = [mn.feature_name[1], mn.feature_name[2], mn.feature_name[3], mn.feature_name[4], "All"]
    ims_strip_infected.append(im_strip_infected)
    ims_strip_not_infected.append(im_strip_not_infected)
    for ims, kind in zip([ims_strip_infected, ims_strip_not_infected], ["Inf", "NotInf"]):
        for im, name in zip(ims, names):
            im[:, -20:, :] = im_lum
            im_fix = np.transpose(im, axes=[1,0,2])
            im_fix = np.flip(im_fix, axis=1)
            im_fix = hsv2rgb(im_fix)
            img_name = dest_dir + f"{kind}_{name}.png"
            imageio.imwrite(img_name, (im_fix * 255.0).astype(np.uint8)) 



def swap_sides(source_dir, do_plant_id, which_ids):
    dirs = ["numpy", "rgb_images", "masked", "flattened_raw", "flattened_random", "flattened_integral", "flattened_features"]
    #dirs = ["flattened_random", "flattened_integral", "flattened_features"]
    days = {13:0, 18:1, 19:2, 20:3, 21:4, 24:5, 25:6, 26:7, 31:8, 32:9, 33:10, 34:11, 38:12, 39:13, 40:14, 41:15, 42:16, 45:17}
    for dir in dirs:
        rename_list = []
        move_back = []
        for fname in listdir(source_dir + dir):
            if fname[0] == "." or len(fname) < 10:
                continue

            base_name = fname.split(".")
            unique_id = base_name[0].split("_")
            if len(unique_id) < 5:
                continue
            if not unique_id[0][0] == "P":
                continue

            plant_id = int(unique_id[0][1:])
            day_id = int(unique_id[1][1:])
            side_id = int(unique_id[2][1:])

            indx_day = days[day_id]
            if plant_id != do_plant_id:
                continue
            if which_ids[indx_day]:
                side_label = ["B", "A"]
            else:
                side_label = ["A", "B"]

            if plant_id < 10:
                plant_id_new = 20 + plant_id
            else:
                plant_id_new = 20 + plant_id - 1

            if len(unique_id) == 5:
                name_end = f"{unique_id[3]}_{unique_id[4]}.{base_name[1]}"
            elif len(unique_id) == 6:
                name_end = f"{unique_id[3]}_{unique_id[4]}_{unique_id[5]}.{base_name[1]}"

            plant_name = f"P{plant_id_new}_D{day_id}_S{side_label[side_id]}_{name_end}"
            if side_label[side_id] == "A":
                plant_name_num = f"P{plant_id_new-20}_D{day_id}_S0_{name_end}"
            else:
                plant_name_num = f"P{plant_id_new-20}_D{day_id}_S1_{name_end}"
            src_name = source_dir + dir + "/" + fname
            dest_name = source_dir + dir + "/" + plant_name            
            move_back.append((dest_name, source_dir + dir + "/" + plant_name_num))

            rename_list.append((src_name, dest_name))

        for (old_name, new_name) in rename_list:
            print(f"Renaming {old_name} {new_name}")
            rename(old_name, new_name)

        for (old_name, new_name) in move_back:
            print(f"move back {old_name} {new_name}")
            rename(old_name, new_name)


if __name__ == '__main__':
    dir = getcwd()    
    if "millarn" in dir:
        source_data_dir = "/Users/millarn/VSCode/data/cherry/numpy_flattened_arrays/"
        dest_dir = "/Users/millarn/VSCode/data/cherry/vizualize/"
    else:
        source_data_dir = "/Users/millarn/VSCode/data/cherry/numpy/"
        source_dir = f"/Users/cindygrimm/VSCode/data/cherry/flattened_raw/"
        dest_dir = "/Users/cindygrimm/VSCode/data/cherry/vizualize/"

    """
    days = {13:0, 18:1, 19:2, 20:3, 21:4, 24:5, 25:6, 26:7, 31:8, 32:9, 33:10, 34:11, 38:12, 39:13, 40:14, 41:15, 42:16, 45:17}
    swap_sides(source_dir="/Users/cindygrimm/VSCode/data/cherry/", do_plant_id=9, 
               which_ids=[True, True, True, True, True,   # 13, 18, 19, 20, 21
                          True, True, True, False, False,   # 24, 25, 26, 31, 32
                          False, False, False, True, True,  # 33, 34, 38, 39, 40
                          True, True, True])            # 41, 42, 45
                          #"""
    im_strip_all(source_dir=source_dir, dest_dir=dest_dir)
