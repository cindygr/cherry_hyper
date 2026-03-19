import numpy as np
import matplotlib.pyplot as plt
from os import listdir, mkdir, chdir, getcwd
from os.path import exists, isdir
import xmltodict
import json as json
import imageio as imageio


def example_process_files(source_dir, dest_dir):
    """ Loop through all files, do something to them, then write new data out
    @param source_dir - directory where source data is
    @param dest_dir - directory to put the results"""

    # Make the output director
    if not exists(dest_dir):
        mkdir(dest_dir)

    for fname in listdir(source_dir):
        # Use for reading/writing
        full_fname = source_dir + fname

        # Unique id - use for naming new files
        fname_parts = fname.split(".")[0].split('_')
        if len(fname_parts) < 4:
            print(f"Skipping {fname}, not a valid file name")
            continue
        unique_id = fname[0:20]

        # do something
        data = np.load(full_fname)
        out_fname = dest_dir + fname
        np.save(out_fname, data.astype(np.float16))
        output_fname = unique_id + "_blah.type"
        # write output_fname


if __name__ == '__main__':
    numpy_data_dir = "/Users/cindygrimm/VSCode/data/cherry/numpy/"
    numpy_data_small_dir = "/Users/cindygrimm/VSCode/data/cherry/numpy_small/"

    example_process_files(numpy_data_dir, numpy_data_small_dir)

