# from spectral.io import envi
# import yaml
import numpy as np
import os

# from scipy.io import savemat
# from scipy.io import loadmat
from PIL import Image
from matplotlib import pyplot as plt

"""
This script is created by Lassi Raivonen and edited by Kimmo Riihiaho.

Dir structure:
workdir
│ │ └── HySuPP.ipynb
│ └──── Forest unmixing
└─ HySUPP
    │ │ └── config/data/ dataset.yaml goes here
    │ └──── data/ dataset.mat goes here
    └────── logs/ output logs go here
"""

# Path to visibility map directory
# visibility_maps_dir = "D:\Koodi\Python\HyperBlend\HyperBlend\System simulation\scene_FDS1_1024/rend\Visibility maps/"

# Names for the visibility files. Remove reference materials if they're not needed.
# visibility_names = [
#     "Ground material_0001",
#     "Diffuse material 1_0001",
#     "Reference 0.00 material_0001",
#     "Reference 0.05 material_0001",
#     "Reference 0.10 material_0001",
#     "Reference 0.25 material_0001",
#     "Reference 0.50 material_0001",
#     "Slab material 1_0001",
#     "Slab material 2_0001",
#     "Slab material 3_0001",
#     "Slab material 4_0001",
#     "Slab material 5_0001",
#     "Slab material 6_0001",
# ]


def ground_truth_endmembers(spectral_cube, visibility_maps_dir: str, visibility_names):
    """
    Calculate ground truth endmembers from visibility maps.

    Parameters:
        spectral_cube
            spectral cube read with "cube = envi.open(path_to_hdr, path_to_img)".
            Cube that is loaded to memory does not work
        visibility_maps_dir:
            Path where visibility maps are located.
        visibility_names:
            List of visibility map file names without extensions.
    Returns:
        np.ndarray
            L x p -shaped numpy array with ground truth endmembers,
            where L is number of channels and p is number of endmembers.

    """

    # Create binary masks from visibility maps.
    visibility_mask_list = []
    for i,visibility_name in enumerate(visibility_names):
        if not visibility_name.endswith(".tif"):
            visibility_name = f"{visibility_name}.tif"
        im = Image.open(f"{visibility_maps_dir}/{visibility_name}")
        imarray = np.array(im)
        binary_arr = (imarray > 0).astype(bool)

        # Skip if all visibility map values are zero. This can happen if no
        #    objects of certain type were spawned on the scene.
        if np.all(binary_arr == False):
            print(f"Empty visibility map encountered in '{visibility_names[i]}'.")
            continue

        visibility_mask_list.append(binary_arr)

    material_means_array = np.zeros(shape=(len(visibility_mask_list), spectral_cube.shape[2]))
    # material_stds = []
    for i,vismask in enumerate(visibility_mask_list):
        # Select pixels by material. This is now a flat array.
        material_pixels = spectral_cube[vismask,:]
        material_means_array[i,:] = np.mean(material_pixels, axis=0, dtype=np.float64)
        # material_std = np.std(material_pixels, axis=0, dtype=np.float64)
        # material_means.append(material_mean)
        # material_stds.append(material_std)

    material_means_array = material_means_array / material_means_array.max()

    return np.swapaxes(material_means_array, 0, 1)


def ground_truth_abundances(factor, visibility_maps_dir: str, visibility_names):
    """
    Calculate ground truth abundances from visibility maps for given downsampling factor.

    Parameters:
        factor:
            Downsampling factor for abundances.
            Factor of 2 corresponds to 512x512.
            Factor of 4 corresponds to 256x256.
            ...
            Factor of 128 corresponds to 8x8.
            Factor of 256 corresponds to 4x4.
        visibility_maps_dir:
            Path where visibility maps are located.
        visibility_names:
            List of visibility map file names without extensions.
    Returns:
        np.ndarray: Ground truth abundances array downsampled by the factor.
    """

    # Image width/height
    n = 1024
    binary_mask_arr = []
    sum_array = np.zeros((n, n), dtype=np.float64)

    # Build binary masks
    for visibility_name in visibility_names:
        if not visibility_name.endswith(".tif"):
            visibility_name = f"{visibility_name}.tif"
        im = Image.open(f"{visibility_maps_dir}/{visibility_name}")
        imarray = np.array(im, dtype=np.float64)
        binary_arr = (imarray > 0).astype(np.float64)
        binary_mask_arr.append(binary_arr)
        sum_array += binary_arr

    """
    Here is where ChatGPT took over and hopefully fixed my code rather than breaking it.
    I tested the function for factors 1 and 512 and they seemed to produce the right results.
    Downsampling is done by averaging rather than being calculated from individual pixels, so
    the values might not be correct to the n:th decimal place.
    """

    # Normalize
    sum_array[sum_array == 0] = 1  # prevent division by zero
    ground_truth_abundances_arr = [arr / sum_array for arr in binary_mask_arr]

    # Back into 3D array for downsampling
    abundances = np.stack(ground_truth_abundances_arr, axis=-1)  # (1024, 1024, p)

    # Downsample
    if factor > 1:
        # Check that downsampling factor is valid
        if n % factor != 0:
            raise ValueError(f"Factor {factor} does not evenly divide {n}.")

        # Downsample by averaging
        abundances = abundances.reshape(
            n // factor, factor, n // factor, factor, -1
        ).mean(axis=(1, 3))

    # Normalize arrays to [0,1]
    sum_downsampled = np.sum(abundances, axis=-1, keepdims=True)
    sum_downsampled[sum_downsampled == 0] = 1
    normalized_abundances = abundances / sum_downsampled

    return normalized_abundances


if __name__ == "__main__":

    #########################################
    ## create dataset.mat and dataset.yaml ##
    #########################################

    """
    Create required dataset files from spectral cube.

        Y: original hyperspectral image (dimension L x N)
        E: ground truth endmembers (dimension L x p)
        A: ground truth abundances (dimension p x N)
        H: HSI number of rows
        W: HSI number of columns
        p: number of endmembers
        L: number of channels
        N: number of pixels (N == H*W)
    """

    # Paths to img and hdr files
    path_to_img = "Forest unmixing/scene_dataset_1024_1/Spectral cube/spectral_cube_dataset_1024_1.img"
    path_to_hdr = "Forest unmixing/scene_dataset_1024_1/Spectral cube/spectral_cube_dataset_1024_1.hdr"

    # Open cube. This is needed for ground_truth_endmembers function
    cube = envi.open(path_to_hdr, path_to_img)

    # Load cube to memory
    Y_cube = cube.load()

    # Remove bands related to water absorption.
    # I manually checked what band images looked bad, a professional should also check these.
    remove = np.concatenate((np.arange(192, 202), np.arange(284, 307)))
    Y_cube = np.delete(Y_cube, remove, axis=2)

    # Divide by max val to get values from 0 to 1
    max_val = np.max(Y_cube)
    Y_cube = Y_cube / max_val

    # Get ground truths and abundances
    E = ground_truth_endmembers(cube, max_val, remove)
    A = ground_truth_abundances(factor=1)

    # Transform the data to right dimensions
    # Checked the right dimensions from DC1.mat data.
    H, W, p = A.shape
    A = A.reshape(H * W, p).T

    H, W, L = Y_cube.shape
    N = H * W
    Y = Y_cube.reshape(H * W, L).T
    print("Y shape:", Y.shape, ", E shape:", E.shape, ", A shape:", A.shape)

    dataset = {
        "Y": Y,
        "H": H,
        "W": W,
        "L": L,
        "N": N,
        "E": E,  # Ground truth endmembers
        "A": A,  # Ground truth abundances
        "p": p,  # Number of endmembers
    }

    data_class_name = "src.data.base.RealHSI"
    dataset_name = "Forest"

    # Create and save mat and yaml files to right subdirectories.
    save_path_mat = "./HySUPP/data/" + dataset_name + ".mat"
    savemat(save_path_mat, dataset)
    print(f"Saved mat to {dataset_name}.mat")

    save_path_yaml = "./HySUPP/config/data/" + dataset_name + ".yaml"
    dataset_yaml = {
        "name": data_class_name,
        "dataset": dataset_name,
        "p": p,
        "data_dir": "${DATA_dir}",
        "figs_dir": "${FIGS_dir}",
    }

    with open(f"{save_path_yaml}", "w") as file:
        yaml.dump(dataset_yaml, file, sort_keys=False)

    print(f"Saved yaml to {dataset_name}.yaml")

    """
    These are for running HySUPP in notebook and just for your inspiration.
    Might work out of the box, might not. It will be an adventure to find out!
    
    ##########################
    ## Unsupervised methods ##
    ##########################
    
    # Running the unmixing analysis for different number of endmembers
    num_of_ems = [9, 10, 11, 12]
    
    for e in num_of_ems:
        !cd HySUPP && python unmixing.py mode=blind data='Forest' data.p={e} model=CNNAEU
    
    # Running the unmixing analysis for single number of endmembers
    # num_of_ems = [10]
    # !cd HySUPP && python unmixing.py mode=blind data='Forest' data.p={e} model=CNNAEU
    
    ########################
    ## Supervised methods ##
    ########################
    
    !cd HySUPP && python unmixing.py mode=supervised data='Forest' model=FCLS

"""
