#!/usr/bin/env python3

# All the "magic" numbers used in the processing


import numpy as np


class HyperSpectralCherryNumbers:
    data_type = {"raw", "normalized", "ratios", "luminance"}
    feature_name = ["Lum", "UV-G", "B-G", "R-G", "NR-G", "G"]
    def __init__(self, process_type="raw"):
        # What kind of processing to do
        self.process_type = process_type

        self.im_width = 512
        self.im_height = 512

        self.red_channel = 90
        self.green_channel = 52
        self.blue_channel = 30

        self.clip_range = [15, 154]

        self.green_range = [40, 65]
        self.red_nir_split = 98
        self.nir_plateau = 120
        self.nir_range = [self.nir_plateau, 154]

        self.red_max = 0.25
        self.green_max = 0.45
        self.nir_max = 1.8

        self.avg_lum_min = 0.075 # Was 0.75
        self.avg_lum_max = 0.5
        self.lum_sd_clip = 0.1

        self.max_lum = 1.65
        # UV-g, B-g, R-g, NIR-G
        self.min_ratio =  (0.2, 0.13, 0.05, 0.5)
        self.max_ratio =  (3.1, 1.1,  1.4,  21.0)
        self.mean_ratio = (1.0, 0.52,  0.48,  4.9)
        self.sd_ratio =   (0.27, 0.12, 0.14, 1.5)

        self.min_width_cutout = 230
        self.min_height_cutout = 335

        self.ratio_green_to_red = 1.2
        self.ratio_green_to_blue = 1.3
        self.ratio_nir_to_red = 1.7

        self.days = {13:0, 18:1, 19:2, 20:3, 21:4, 24:5, 25:6, 26:7, 31:8, 32:9, 33:10, 34:11, 38:12, 39:13, 40:14, 41:15, 42:16, 45:17}

        self.infected = {1:"P20", 2:"P33", 0:"P157", 6:"P163", 8:"P338"}
        self.not_infected = {3:"P401", 7:"P406", 4:"P418", 9:"P421", 5:"P430"}        

    def fname_tag(self):
        return f"_{self.fname_tag}"
    
    def n_images(self):
        return len(self.days)
    
    def n_spectral(self):
        return self.clip_range[1] - self.clip_range[0]
    
    def map_ratio_zero_one(self, in_val, indx):
        clip_low = self.mean_ratio[indx] - self.sd_ratio[indx]
        clip_hi = self.mean_ratio[indx] + self.sd_ratio[indx]
        pad = 0.1
        split = 0.15
        mid = 1.0 - 2.0 * split - pad
        if in_val < clip_low:
            return pad + split * (in_val - self.min_ratio[indx]) / (clip_low - self.min_ratio[indx])
        elif in_val < clip_hi:
            return pad + mid * (in_val - clip_low) / (clip_hi - clip_low)
        else:
            return pad + split * (in_val - clip_hi) / (self.max_ratio[indx] - clip_hi)
        
    def get_pixel_lumin_mean(self, spectrum, b_clip=False):
        if b_clip:
            lum_mean = np.mean(spectrum[self.clip_range[0]:self.clip_range[1]])
        else:
            lum_mean = np.mean(spectrum)

        return lum_mean
    
    def get_pixel_data(self, in_spectrum, b_clip=False):
        spectrum = in_spectrum
        if b_clip:
            spectrum = in_spectrum[self.clip_range[0], self.clip_range[1]]
        
        if self.process_type == "raw":
            return spectrum
        
        if self.process_type == "normalized":
            sum_all = np.sum(spectrum)
            return spectrum / sum_all
        
        if self.process_type == "ratios":
            # feature_name = ["Lum", "UV-G", "B-G", "R-G", "NR-G", "G"]
            lum = self.get_pixel_lumin_mean(spectrum)
            uv_val = np.max(in_spectrum[self.clip_range[0]:self.red_channel])
            blue_val = np.min(in_spectrum[self.clip_range[0]:self.green_range[0]])
            green_val = np.max(in_spectrum[self.green_range[0]:self.green_range[1]])
            red_val = np.min(in_spectrum[self.green_range[1]:self.nir_plateau])
            nir_val = in_spectrum[self.nir_plateau]
            out_data = np.array([lum, 
                                 uv_val / green_val,
                                 blue_val / green_val, 
                                 red_val / green_val,
                                 nir_val / green_val, 
                                 green_val])
            return out_data
        
        if self.process_type == "luminance":
            return np.array([lum])
                               
