"""

Temporarily store forest scene related stuff here, so they don't
mess with 0.2.0 release.

These are mostly hacks and should be rewritten into proper code.

"""


import numpy as np
import matplotlib.pyplot as plt

from src.data import path_handling as PH
from src.data import toml_handling as TH


######### Stuff for scenes testing #############

"""
This used to be in main for testing forest scenes. 
Just copypasted here out of the way.
"""

# set_name = 'surface_test_predict_2'

# FH.clear_folder(PH.path_directory_subresult(set_name, 0))
# o = Optimization(set_name)
# wls,r,t = prospect.get_default_prospect_leaf()
# diff = r-t
# max_diff = diff.max()
# print(f'max difference of r and t = {max_diff}')
# SU._make_target(set_name, wls=wls, r_m=r, t_m=t)
# o.run_optimization(resolution=5, use_threads=True, prediction_method='surface')

# # Test the software with hard coded data.
# presets.optimize_default_target(spectral_resolution=50)
#
# # Example using "real" data
# data = [[400, 0.21435, 0.26547], [401, 0.21431, 0.26540]]
# set_name = 'test_set'
# o = Optimization(set_name)
# TH.write_target(set_name, data, sample_id=0)
# o.run_optimization()

# scene_id = '2704221302'
# ndvi(scene_id)

# scene_id = FH.duplicate_scene_from_template()
# # scene_id = "0123456789" # id for debugging
# forest.generate_some_leaf_stuff(scene_id, resolution=50)
# BC.setup_forest(scene_id, leaf_id_list=[1,2,3])
# BC.render_forest_previews(scene_id)
# BC.render_forest_spectral(scene_id)
# show_forest_rend(11, scene_id)

# for i in range(1,44):
#     show_forest_rend(i, scene_id)
# show_forest_rend(2, scene_id)
# show_forest_rend(3, scene_id)
# show_forest_rend(15, scene_id)

# load_into_cube(scene_id=scene_id)


def show_forest_rend(band, scene_id):

    p = PH.join(PH.path_directory_forest_rend_spectral(scene_id), f"band_{band:04}.tiff")
    band1 = plt.imread(p)
    plt.imshow(band1)
    plt.colorbar()
    plt.show()


def white_ref(frame, percent=2):
    ref_half_width = int((frame.shape[0] / 100) * percent / 2)
    ref_half_height = int((frame.shape[1] / 100) * percent / 2)
    x_mid = frame.shape[0] / 2
    y_mid = frame.shape[1] / 2
    white_area = frame[int(x_mid-ref_half_width):int(x_mid+ref_half_width), int(y_mid-ref_half_height):int(y_mid+ref_half_height)]
    white_mean = np.mean(white_area)
    return white_mean


def ndvi(scene_id):
    red = plt.imread(PH.join(PH.path_directory_forest_rend_spectral(scene_id), f"band_{11:04}.tiff"))
    nir = plt.imread(PH.join(PH.path_directory_forest_rend_spectral(scene_id), f"band_{21:04}.tiff"))
    red = red / white_ref(red)
    nir = nir / white_ref(nir)

    # red = np.ndarray.astype(red, dtype=np.float32)
    # nir = np.ndarray.astype(nir, dtype=np.float32)
    # nir = nir / nir.max()
    np.nan_to_num(red, copy=False)
    np.nan_to_num(nir, copy=False)
    vi = (nir - red) / (nir + red)
    np.nan_to_num(vi, copy=False)
    plt.imshow(vi)
    plt.colorbar()
    plt.show()


def load_into_cube(scene_id):
    p = PH.path_directory_forest_rend_spectral(scene_id)
    for filename in os.listdir(p):
        print(filename)


def maxdiff(set_name="prospect_randoms"):

    # Find maximum r t difference of sample.
    res = TH.read_sample_result(set_name, 0)
    r = np.array(res['refls_modeled'])
    t = np.array(res['trans_modeled'])
    diff = np.abs(r-t)
    print(f"max diff : {diff.max()}")
