#!/usr/bin/env python3

# All the "magic" numbers used in the processing

class HyperSpectralCherryNumbers:
    def __init__(self):
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

        self.avg_lum_min = 0.075
        self.avg_lum_max = 0.5

        self.ratio_green_to_red = 1.2
        self.ratio_green_to_blue = 1.3
        self.ratio_nir_to_red = 3.0

        self.infected = {1:"P20", 2:"P33", 0:"P157", 7:"P163", 9:"P338"}
        self.not_infected = {4:"P401", 8:"P406", 5:"P418", 10:"P421", 6:"P430"}

    def n_spectral(self):
        return self.clip_range[1] - self.clip_range[0]
