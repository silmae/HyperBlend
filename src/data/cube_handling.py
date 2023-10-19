
import numpy as np
import spectral
import os
import matplotlib.pyplot as plt
import logging
import csv

from src.data import path_handling as PH
from src.utils import spectra_utils as SU
from src import constants as C


def construct_envi_cube(forest_id: str, light_max_power):
    """Constructs an ENVI style hyperspectral image cube out of rendered images.

    Can be used after the scene has been rendered (at least spectral and visibility maps).

    White reference for reflectance calculation is searched automatically from
    available visibility maps. Note that the maps must be named like "Reference 0.00 material...".

    Default RGB bands for ENVI metadata are inferred if in visible range.
    Otherwise first, middle, and last bands are used.

    :return:
    """

    p = PH.path_directory_forest_rend_spectral(forest_id=forest_id)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Rend directory for forest '{forest_id}' not found.")

    frame_name_list = os.listdir(p)
    if len(frame_name_list) < 1:
        raise FileNotFoundError(f"No rendered frames were found from '{p}'.")

    frame_list = []
    for thing in frame_name_list:
        file_path = PH.join(p, thing)
        image_as_array = plt.imread(file_path)
        frame_list.append(image_as_array)

    raw_cube = np.array(frame_list)

    # Burnt areas have values around 65535
    # Loop white references until the image is not burned
    max_burn = 65000.0

    p = PH.path_file_forest_sun_csv(forest_id=forest_id)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Sun csv file '{p}' not found. Try rerunning forest initialization.")


    # Retrieve band and wavelength info from the sun file.
    wls = []
    bands = []
    irradiances = []
    with open(p) as file:
        reader = csv.reader(file, delimiter=' ')
        for row in reader:

            if "band" in row:
                continue # header row

            bands.append(int(row[0]))
            wls.append(float(row[1]))
            irradiances.append(float(row[2]))

    # bands, wls, irradiances = FU.read_csv(p)
    white = np.array(irradiances) * light_max_power

    # white_mean = max_burn

    # Find available reflectance plate reflectivity based on visibility map file names.
    # reflectivities = []
    # map_names = PH.list_reference_visibility_maps(forest_id=forest_id)
    # for map_name in map_names:
    #     splitted = map_name.split(' ')
    #     reflectivity = float(splitted[1])
    #     if reflectivity > 0.0:
    #         reflectivities.append(reflectivity)
    #
    # reflectivities.sort(reverse=True)

    # logging.info(f"Searching for good white reference plate..")
    # accepted_reflectivity = None
    # for reflectivity in reflectivities:
    #     accepted_reflectivity = reflectivity
    #     mask_path = PH.find_reference_visibility_map(forest_id=forest_id, reflectivity=reflectivity)
    #     mask = plt.imread(mask_path)
    #     mask = mask > 0
    #     white_cube = raw_cube[:,mask] # Flattens the reference plate area pixels..
    #     white_mean = np.mean(white_cube, axis=(1)) #.. so we take the mean only on one axis.
    #     white_mean_max = white_mean.max()
    #     if white_mean_max < max_burn:
    #         break

    # logging.info(f"Accepted white reference with {accepted_reflectivity:.2f} reflectivity producing maximum mean reflectance {white_mean_max:.1f}.")

    # Originally loaded as z,x,y, where z is spectral dimension
    raw_cube = np.swapaxes(raw_cube, 0,2) # swap to y,x,z
    raw_cube = np.swapaxes(raw_cube, 0,1) # swap to x,y,z

    # white_mean = np.expand_dims(white_mean, axis=(1,2))
    reflectance_cube = np.divide(raw_cube, white, dtype=np.float32)
    # refl_max = np.max(reflectance_cube)

    # Swap axis to arrange the array as expected by spectral.envi
    # reflectance_cube = np.swapaxes(reflectance_cube, 0,2)
    # reflectance_cube = np.swapaxes(reflectance_cube, 0,1)

    # p = PH.path_file_forest_sun_csv(forest_id=forest_id)
    # if not os.path.exists(p):
    #     logging.warning(f"Could not find sun data for wavelength info. The image cube will be saved without it.")


    # Define default RGB bands.
    if SU.is_in_visible(wls=wls):
        nearest_R_idx = SU.find_nearest_idx(wls, C.default_R_wl)
        nearest_G_idx = SU.find_nearest_idx(wls, C.default_G_wl)
        nearest_B_idx = SU.find_nearest_idx(wls, C.default_B_wl)
        default_bands = [bands[nearest_R_idx], bands[nearest_G_idx], bands[nearest_B_idx]]
    else:
        default_bands = [bands[-1], bands[int(len(bands)/2)], bands[0]]

    header_dict = {
        "bands" : reflectance_cube.shape[0],
        "lines": reflectance_cube.shape[1],
        "samples": reflectance_cube.shape[2],
        "data_type": 4,
        # "reference reflectivity": accepted_reflectivity,
        "default bands": default_bands,
        "wavelength": wls,
        "wavelength units": "nm",
    }

    cube_dir_path = PH.path_directory_forest_cube(forest_id)
    if not os.path.exists(cube_dir_path):
        os.makedirs(cube_dir_path)

    p_hdr = PH.path_file_forest_reflectance_header(forest_id=forest_id)
    # SPy wants to know only the path to the header. It will find the image file automatically from the same dir.
    spectral.envi.save_image(hdr_file=p_hdr, image=reflectance_cube, dtype=np.float32, force=True, metadata=header_dict)


def show_cube(forest_id: str):
    """Shows the hyperspectral image cube.

    Use construct_envi_cube() to generate it.

    :param forest_id:
        Forest scene id.
    :return:
        None
    :raises
        FileNotFoundError if the cube does not exist.
    """

    p_cube = PH.path_file_forest_reflectance_header(forest_id=forest_id)
    if not os.path.exists(p_cube):
        raise FileNotFoundError(f"Cannot find spectral cube file from '{p_cube}'. "
                                f"Use construct_envi_cube() to generate the cube from rendered images.")
    data = spectral.open_image(p_cube)

    # Minus 1 because spectral is zero-based and ENVI standard one-based.. apparently.
    default_bands = [int(band) - 1 for band in data.metadata['default bands']]

    rgb = data.read_bands(bands=default_bands)
    plt.close('all') # Close all previous plots before showing this one.
    plt.figure(figsize=(10,10))
    plt.imshow(rgb)
    plt.show()

    # TODO Would be nice if this worked, but it just flashes on the screen
    # view = spectral.imshow(data, bands=default_bands)
