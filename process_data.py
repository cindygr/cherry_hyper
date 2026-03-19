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

        output_fname = unique_id + "_blah.type"
        # write output_fname


if __name__ == '__main__':

    extract_data_as_numpy()

    # Load the hyperspectral image
    img = spy.open_image(hdr_path)

    # Convert to a numpy array (optional, but useful)
    data = img.load()

    print("Image shape (rows, cols, bands):", data.shape)

    # Display a quick RGB composite
    # SPECIM cameras often use bands around:
    # R ≈ 60, G ≈ 30, B ≈ 10 (this varies by model!)
    rgb = spy.get_rgb(img, [60, 30, 10])

    plt.figure(figsize=(8, 6))
    plt.imshow(rgb)
    plt.title("SPECIM RGB Composite")
    plt.axis("off")
    plt.show()
