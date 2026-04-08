import spectral as spy
import numpy as np
import matplotlib.pyplot as plt
from os import listdir, mkdir, chdir, getcwd
from os.path import exists, isdir
import xmltodict
import json as json
import imageio as imageio


# Convert the hdr data 
#.  Copies the hdr view finder image to rgb_images (to ensure conversion is correct)
#   Converts the raw hdr data to a numpy file (in numpy)
#.  Creates an rgb image from the raw hdr data and saves it to rgb_imgages
#.  Issues with the conversion (a few missing files, an incorrect (?) plant number) are in Problems.txt (this folder)


def convert_to_numpy(hdr_path):
    # Load the hyperspectral image
    img = spy.open_image(hdr_path)

    # Convert to a numpy array (optional, but useful)
    data = img.load()

    print("Image shape (rows, cols, bands):", data.shape)
    return data, img


def show_as_image(img):
    # Display a quick RGB composite
    # SPECIM cameras often use bands around:
    # R ≈ 60, G ≈ 30, B ≈ 10 (this varies by model!)
    rgb = spy.get_rgb(img, [60, 30, 10])

    plt.figure(figsize=(8, 6))
    plt.imshow(rgb)
    plt.title("SPECIM RGB Composite")
    plt.axis("off")
    plt.show()
 

def extract_data_as_numpy():
    # Path to your data directory
    data_dir = "/Users/cindygrimm/VSCode/data/cherry/Cherry Hyperspectral Imaging/"
    numpy_data_dir = "/Users/cindygrimm/VSCode/data/cherry/numpy/"
    images_data_dir = "/Users/cindygrimm/VSCode/data/cherry/rgb_images/"

    plant_numbers = {}
    plant_names = []
    day_dates = []
    for _ in range(0, 60):
        day_dates.append('Not found')
    errors = []

    plant_count = 0
    # Number of plants, number of days, side
    plant_days = np.zeros((16, 60, 2), dtype=np.bool)

    if not exists(numpy_data_dir):
        mkdir(numpy_data_dir)

    if not exists(images_data_dir):
        mkdir(images_data_dir)

    chdir(data_dir)
    
    for fname in listdir(data_dir):
        fname_parts = fname.split(" ")
        if len(fname_parts) != 6:
            print(f"Skipping {fname}")
            continue

        day_dir = data_dir + fname
        for fname_plant in listdir(day_dir):
            fname_plant_parts = fname_plant.split('_')
            if len(fname_plant) < 14:
                print(f"Skipping sub folder {fname_plant} in {fname}")
                continue
            if len(fname_plant) != 14:
                fname_plant_parts = fname_plant[-14:].split('_')
            
            if len(fname_plant_parts) != 2:
                continue
            fname_meta_data_dir = day_dir + "/" + fname_plant + "/metadata/"
            for fname_xml in listdir(fname_meta_data_dir):
                if ".xml" in fname_xml:
                    with open(fname_meta_data_dir + fname_xml, "r") as f:
                        xmldata = f.read()
                    # Use xmltodict.parse() to convert the XML string to an OrderedDict
                    ordered_dict = xmltodict.parse(xmldata)
        
                    # Optionally convert the OrderedDict to a standard dictionary (if order doesn't matter)
                    # This is useful for easier data manipulation
                    plant_dict = json.loads(json.dumps(ordered_dict))
                    global_tag = plant_dict["properties"]["global_tag"]
                    if global_tag != None:
                        plant_id = global_tag["key"]['@field']
                    else: 
                        print(f"Bad folder {fname_meta_data_dir}")
                        continue
                        #plant_id = global_tag["key"]['@field']
                        #plant_id = "None"

                    plant_id_parts = plant_id.split('.')
                    b_inserted = False
                    if len(plant_id_parts) == 4:
                        print("Warning: Missing side, assuming 0")
                        plant_id_parts.insert(1, 0)
                        b_inserted = True
                    elif len(plant_id_parts) == 2:
                        if fname_parts[4] == 'Nov':
                            plant_id_parts.append(11)
                        elif fname_parts[4] == 'Dec':
                            plant_id_parts.append(12)
                        plant_id_parts.append(int(fname_parts[3]))
                    for indx in range(1, len(plant_id_parts)):
                        plant_id_parts[indx] = int(plant_id_parts[indx])
                    if int(fname_parts[3]) != plant_id_parts[3]:
                        print(f"Bad match day {int(fname_parts[3])} {plant_id_parts[3]}")
                        plant_id_parts[2], plant_id_parts[3] = plant_id_parts[3], plant_id_parts[2]
                    if fname_parts[4] == 'Nov' and plant_id_parts[2] != 11:
                        print(f"Bad match month Nov")
                    if fname_parts[4] == 'Dec' and plant_id_parts[2] != 12:
                        print(f"Bad match month Dec")
                    print(f"plant_id {plant_id} {fname_meta_data_dir}")

                    if not plant_id_parts[0] in plant_numbers:
                        plant_numbers[plant_id_parts[0]] = plant_count
                        plant_names.append(plant_id_parts[0])
                        which_plant = plant_count
                        plant_count = plant_count + 1
                    else:
                        which_plant = plant_numbers[plant_id_parts[0]]
                    side = int(plant_id_parts[1]) - 1
                    if plant_id_parts[2] == 12:
                        day = 30 + plant_id_parts[3]
                    elif plant_id_parts[2] == 11:
                        day = plant_id_parts[3]
                    day_dates[day] = str(plant_id_parts[2]) + "-" + str(plant_id_parts[3])
                    if plant_days[which_plant][day][side]:
                        if b_inserted:
                            if plant_days[which_plant][day][1]:
                                print(f"Error, triplicate with guessed side {plant_id} replacing")
                                errors.append(plant_id)
                                errors.append(fname_meta_data_dir)
                            else:
                                side = 1
                                plant_days[which_plant][day][side] = True
                        else:
                            side = (side + 1) % 2
                            if plant_days[which_plant][day][side]:
                                print(f"Error, triplicate {plant_id} replacing")
                                errors.append(plant_id)
                                errors.append(fname_meta_data_dir)
                            else:
                                print(f"Warning, mislabeled {plant_id}")
                                plant_days[which_plant][day][side] = True
                    else:
                        plant_days[which_plant][day][side] = True

                    plant_unique_id = f"P{which_plant}_D{day}_S{side}_{plant_id_parts[0]}_{day_dates[day]}"
                    print(f"Unique id {plant_unique_id}")
            # Now get the 
            fname_hyper_data_dir = day_dir + "/" + fname_plant + "/results/"
            for fname_results in listdir(fname_hyper_data_dir):
                if ".hdr" in fname_results:
                    numpy_fname = numpy_data_dir + plant_unique_id + ".npy"
                    if exists(numpy_fname):
                        print(f"Skipping {numpy_fname}, already processed")
                        #continue
                    try:
                        data, sp_im = convert_to_numpy(fname_hyper_data_dir + fname_results)
                    except:
                        print(f"Data file not found {fname_hyper_data_dir} {fname_results}")
                        errors.append(f"Missing data file {fname_hyper_data_dir} {fname_results}")
                        continue
                        
                    np.save(numpy_data_dir + plant_unique_id + ".npy", data.astype(np.float32))
                    # Gets 3 of the hyperspectral channels at nm 60, 30, and 10 and uses that as the red, green, and blue channels
                    rgb = spy.get_rgb(sp_im, [10, 30, 60]) * 256
                    # This nonsense is because spy must be using the OpenCV image format, which has
                    #.  y is in channel 0, x in channel 1, and y is flipped top/bottom
                    #.  uses blue, green, red instead of red, green blue
                    rgb = np.transpose(rgb, axes=[1,0,2])
                    rgb = np.flip(rgb, axis=1)
                    imageio.imwrite(images_data_dir + plant_unique_id + "_hdr.png", rgb.astype(np.uint8))
                if "VIEWFINDER" in fname_results:
                    rgb_fname = images_data_dir + plant_unique_id + "_viewfinder.png"
                    if exists(rgb_fname):
                        print(f"Skipping {rgb_fname}, already processed")
                        continue
                    rgb_view = imageio.imread(fname_hyper_data_dir + fname_results)
                    imageio.imwrite(rgb_fname, rgb_view)


    missing = []
    for pid in range(0, plant_count):
        for day in range(0, plant_days.shape[1]):
            if plant_days[pid][day][0] and not plant_days[pid][day][1]:
                missing.append([pid, day, 1, plant_names[pid], day_dates[day]])
                print(f"Plant {pid} {plant_names[pid]} day {day_dates[day]} only side 0" )
            if plant_days[pid][day][1] and not plant_days[pid][day][0]:
                missing.append([pid, day, 0, plant_names[pid], day_dates[day]])
                print(f"Plant {pid} {plant_names[pid]} day {day_dates[day]} only side 1" )
            if not plant_days[pid][day][0] and not plant_days[pid][day][1]:
                if day in day_dates:
                    missing.append([pid, day, 0, plant_names[pid], day_dates[day]])
                    missing.append([pid, day, 1, plant_names[pid], day_dates[day]])
                    print(f"Plant {pid} {plant_names[pid]} missing day {day_dates[day]} both sides" )

    print(f"Dates: {day_dates}")
    print(f"Plant numbers: {plant_numbers}")
    print(f"Errors {errors}")
    with open(data_dir + "mapping.json", "w") as f:
        all_dict = {"Days": day_dates, "Plants": plant_numbers, "Errors": errors, "Missing": missing}
        json.dump(all_dict, f, indent=4)
    return plant_days, day_dates         


if __name__ == '__main__':
    extract_data_as_numpy()

