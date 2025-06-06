import numpy as np
import spectral
import os
import matplotlib.pyplot as plt
import logging
import csv

from src.data import path_handling as PH
from src.utils import spectra_utils as SU
from src import constants as C


def construct_envi_cube(system_sim_name: str):
    """Constructs an ENVI style hyperspectral image cube out of rendered images.

    Can be used after the scene has been rendered (at least spectral and visibility maps).

    White reference for reflectance calculation is searched automatically from
    available visibility maps. Note that the maps must be named like
    `Reference 0.00 material_0001.tif`.

    TODO how this should behave if there are no visibility maps?

    Default RGB bands for ENVI metadata are inferred if in visible range.
    Otherwise first, middle, and last bands are used.

    :raises FileNotFoundError: if the rendered frames directory does not exist or is
        empty. Also if the sun data file does not exist, which is needed for wavelength info.
    """

    p = PH.directory_system_rend_spectral(system_sim_name=system_sim_name)
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"Rend directory for system simulation '{system_sim_name}' not found."
        )

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
    white_mean = max_burn

    # Find available reflectance plate reflectivity based on visibility map file names.
    reflectivities = []
    map_names = PH.list_reference_visibility_maps(system_sim_name=system_sim_name)
    for map_name in map_names:
        splitted = map_name.split(" ")
        reflectivity = float(splitted[1])
        if reflectivity > 0.0:
            reflectivities.append(reflectivity)

    reflectivities.sort(reverse=True)

    logging.info(f"Searching for a good white reference plate..")
    accepted_reflectivity = None
    for reflectivity in reflectivities:
        accepted_reflectivity = reflectivity
        mask_path = PH.find_reference_visibility_map(
            system_sim_name=system_sim_name, reflectivity=reflectivity
        )
        mask = plt.imread(mask_path)
        mask = mask > 0
        # Flattens the reference plate area pixels
        white_cube = raw_cube[:, mask]
        # so we take the mean only on one axis.
        white_mean = np.mean(white_cube, axis=(1))
        white_mean_max = white_mean.max()
        if white_mean_max < max_burn:
            logging.info(
                f"Accepted white reference with {accepted_reflectivity:.2f} reflectivity "
                f"producing maximum mean reflectance {white_mean_max:.1f}."
            )
            break

    white_mean = np.expand_dims(white_mean, axis=(1, 2))
    reflectance_cube = np.divide(raw_cube, white_mean, dtype=np.float32)

    # Swap axis to arrange the array as expected by spectral.envi
    reflectance_cube = np.swapaxes(reflectance_cube, 0, 2)
    reflectance_cube = np.swapaxes(reflectance_cube, 0, 1)

    p = PH.file_system_sim_light_spectra_csv(
        system_sim_name=system_sim_name, light_file_name=C.file_blender_default_sun
    )
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"Could not find sun data for wavelength info from `{p}`. "
            f"There is no point saving the cube without bands and wavelengths."
        )

    # Retrieve band and wavelength info from the sun file.
    wls = []
    bands = []
    with open(p) as file:
        reader = csv.reader(file, delimiter=" ")
        for row in reader:

            if "band" in row:
                continue  # header row

            bands.append(int(row[0]))
            wls.append(float(row[1]))

    # Define default RGB bands.
    if SU.is_in_visible(wls=wls):
        default_bands = [
            bands[SU.find_nearest_idx(wls, C.default_R_wl)],
            bands[SU.find_nearest_idx(wls, C.default_G_wl)],
            bands[SU.find_nearest_idx(wls, C.default_B_wl)],
        ]
    else:
        default_bands = [bands[-1], bands[int(len(bands) / 2)], bands[0]]

    # TODO Perhaps the extra ENVI header keys should be defined in constants.py
    header_dict = {
        "bands": reflectance_cube.shape[0],
        "lines": reflectance_cube.shape[1],
        "samples": reflectance_cube.shape[2],
        "data_type": 4,
        "reference reflectivity": accepted_reflectivity,
        "default bands": default_bands,
        "wavelength": wls,
        "wavelength units": "nm",
    }

    cube_dir_path = PH.directory_system_spectral_cube(system_sim_name=system_sim_name)
    if not os.path.exists(cube_dir_path):
        os.makedirs(cube_dir_path)

    p_hdr = PH.file_spectral_cube(system_sim_name=system_sim_name, file_type="header")
    # SPy wants to know only the path to the header. It will find
    #   the image file automatically from the same directory.
    spectral.envi.save_image(
        hdr_file=p_hdr,
        image=reflectance_cube,
        dtype=np.float32,
        force=True,
        metadata=header_dict,
    )


def show_cube(system_sim_name: str):
    """Shows the hyperspectral image cube.

    This is mainly for debugging purposes, to quickly visualize the cube.
    For any real needs, use a dedicated ENVI viewer, such as the CubeInspector
    found at `github.com/silmae/CubeInspector <https://github.com/silmae/cubeinspector>`_.
    CubeInspector is a sister project of HyperBlend.

    :raises FileNotFoundError: if the cube does not exist.
    """

    p_cube = PH.file_spectral_cube(system_sim_name=system_sim_name, file_type="header")

    if not os.path.exists(p_cube):
        raise FileNotFoundError(f"Cannot find spectral cube file from '{p_cube}'. ")

    data = spectral.open_image(p_cube)

    # Minus 1 because spectral is zero-based and ENVI standard one-based.. apparently.
    default_bands = [int(band) - 1 for band in data.metadata["default bands"]]

    rgb = data.read_bands(bands=default_bands)
    plt.close("all")  # Close all previous plots before showing this one.
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb)
    plt.show()
