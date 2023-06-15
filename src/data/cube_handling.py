
import numpy as np
import spectral
import os
import matplotlib.pyplot as plt

from src.data import path_handling as PH
from src.data import file_handling as FH


def rended_to_envi_cube(forest_id: str):

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
        print(thing)

    raw_cube = np.array(frame_list)
    # TODO below hack radiance
    reflectance_cube = np.divide(raw_cube, raw_cube.max(), dtype=np.float32)
    reflectance_cube = np.swapaxes(reflectance_cube, 0,2)
    reflectance_cube = np.swapaxes(reflectance_cube, 0,1)
    print(f"new shape: {reflectance_cube.shape}")

    header_dict = {
        "bands" : reflectance_cube.shape[0],
        "lines": reflectance_cube.shape[1],
        "samples": reflectance_cube.shape[2],
        "data_type": 4,
        "test value": 4,
        # TODO read wavelengths to metadata
    }

    p_hdr = PH.path_file_forest_reflectance_header(forest_id=forest_id)
    # SPy wants to know only the path to the header. It will find the image file automatically from the same dir.
    spectral.envi.save_image(hdr_file=p_hdr, image=reflectance_cube, dtype=np.float32, force=True, metadata=header_dict)


def show_cube(forest_id: str):

    # spectral.envi.open()
    p_cube = PH.path_file_forest_reflectance_header(forest_id=forest_id)
    data = spectral.open_image(p_cube)
    rgb = data.read_bands(bands=(1,2,3))
    plt.imshow(rgb)
    plt.show()
    # view = spectral.imshow(data, bands=(0,1,2))
