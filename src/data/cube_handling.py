import numpy as np
import spectral
import os
import matplotlib.pyplot as plt
import logging
import csv

from fontTools.feaLib.ast import asFea

from src.data import path_handling as PH, toml_handling as TH
from src.utils import spectra_utils as SU
from src import constants as C, plotter


def construct_envi_cube(system_sim_name: str, system_sim_name_for_white_signal=None):
    """Constructs an ENVI style hyperspectral image cube out of rendered images.

    Can be used after the scene has been rendered (at least spectral).

    White reference for reflectance calculation is searched automatically from
    available visibility maps if `system_sim_name_for_white_signal` is not given .
    Note that the maps must be named like `Reference 0.00 material_0001.tif`.

    Saves white signal used in reflectance calculation as a toml file.

    Default RGB bands for ENVI metadata are inferred if in visible range.
    Otherwise first, middle, and last bands are used.

    :param system_sim_name: Name of the system simulation to process.
    :param system_sim_name_for_white_signal: Optionally, give system simulation name
        from where to fetch the white signal. If None, it is inferred from the data.
        If there are no visibility maps in the current system simulation, this will fail
        and raise an error.
    :raises FileNotFoundError: if the rendered frames directory does not exist or is
        empty. Also if the sun data file does not exist, which is needed for wavelength info.
    """

    raw_cube = get_raw_cube(system_sim_name=system_sim_name)

    if system_sim_name_for_white_signal is None:
        logging.info(f"Inferring white signal from data in '{system_sim_name}'.")
        white_signal = infer_white_ref_from_data(system_sim_name=system_sim_name)
    else:
        logging.info(f"Using white signal from '{system_sim_name_for_white_signal}'.")
        white_signal = read_white_signal(
            system_sim_name=system_sim_name_for_white_signal
        )

    reflectance_cube = np.divide(raw_cube, white_signal, dtype=np.float32)

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
    logging.info(f"Reflectance cube saved for system simulation '{system_sim_name}'.")


def get_raw_cube(system_sim_name: str):
    """Reads rendered spectral frames into a raw hyperspectral image cube.

    :param system_sim_name: Name of the system simulation.
    :return: Raw hyperspectral image cube as a 3D numpy array with shape (bands, height, width).
    :raises FileNotFoundError: if the rendered frames directory does not exist or is empty.
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
    return raw_cube


def infer_white_ref_from_data(system_sim_name: str):
    """
    Infers a good white reference signal from available visibility maps.

    :param system_sim_name: Name of the system simulation.
    :return: White reference signal as a 3D numpy array with shape (bands, 1, 1).
    :raises FileNotFoundError: if no visibility maps are found.
    """

    # Burnt areas have values around 65535
    # Loop white references until the image is not burned
    max_burn = 65000.0
    white_mean = max_burn

    # Find available reflectance plate reflectivity based on visibility map file names.
    reflectivities = []
    map_names = PH.list_reference_visibility_maps(system_sim_name=system_sim_name)

    if len(map_names) < 1:
        raise FileNotFoundError(
            f"No reference visibility maps were found for system simulation '{system_sim_name}'. "
            f"Cannot construct reflectance cube without a white reference."
        )

    for map_name in map_names:
        splitted = map_name.split(" ")
        reflectivity = float(splitted[1])
        if reflectivity > 0.0:
            reflectivities.append(reflectivity)

    reflectivities.sort(reverse=True)

    logging.info(f"Searching for a good white reference plate..")
    for reflectivity in reflectivities:
        accepted_reference_plate_reflectivity = reflectivity
        mask_path = PH.find_reference_visibility_map(
            system_sim_name=system_sim_name, reflectivity=reflectivity
        )
        mask = plt.imread(mask_path)
        mask = mask > 0

        # Flattens the reference plate area pixels
        raw_cube = get_raw_cube(system_sim_name=system_sim_name)
        white_cube = raw_cube[:, mask]
        # so we take the mean only on one axis.
        white_mean = np.mean(white_cube, axis=(1))
        white_mean_max = white_mean.max()
        if white_mean_max < max_burn:
            logging.info(
                f"Accepted white reference with {accepted_reference_plate_reflectivity:.2f} reflectivity "
                f"producing maximum mean reflectance {white_mean_max:.1f}."
            )
            break

    save_white_signal(system_sim_name=system_sim_name, white_signal=white_mean)

    # Expand dimensions to match the raw cube shape for reflectance calculation.
    white_mean = np.expand_dims(white_mean, axis=(1, 2))
    return white_mean


def save_white_signal(system_sim_name: str, white_signal: np.ndarray):
    """Saves the white signal used in reflectance calculation as a toml file.

    :param system_sim_name: Name of the system simulation.
    :param white_signal: White signal as a 1D numpy array with shape (bands,).
    """

    white_dict = {"white_signal": white_signal}

    write_dir = PH.directory_system_simulation(system_sim_name=system_sim_name)
    TH.write_dict_as_toml(
        dictionary=white_dict, directory=write_dir, filename="white_signal"
    )
    logging.info(f"Saved white signal for system simulation '{system_sim_name}'.")


def read_white_signal(system_sim_name: str) -> np.ndarray:
    """Reads the white signal used in reflectance calculation from a toml file.

    :param system_sim_name: Name of the system simulation.
    :return: White signal as a 3D numpy array with shape (bands, 1, 1).
    :raises FileNotFoundError: if the white signal file does not exist.
    """

    logging.info(f"Reading white signal for system simulation '{system_sim_name}'.")
    read_dir = PH.directory_system_simulation(system_sim_name=system_sim_name)
    if not os.path.exists(read_dir):
        raise FileNotFoundError(
            f"Cannot find system simulation directory from '{read_dir}'. "
            f"Cannot read white signal."
        )
    white_dict = TH.read_toml_as_dict(directory=read_dir, filename="white_signal")
    white_signal = np.array(white_dict["white_signal"])
    white_signal = np.expand_dims(white_signal, axis=(1, 2))
    return white_signal


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
