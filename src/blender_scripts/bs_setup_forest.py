# bpy stands for Blender Python, which is included Blender's own Python environment.
# It exists inside Blender, so don't worry if your IDE flags it as not found.
import bpy
import os
import sys  # to get command line args
import argparse  # to parse options for us and print a nice help message
import logging
import math
import importlib
import csv
import random
import numpy as np

blend_dir = os.path.dirname(os.path.abspath(bpy.data.filepath))

if 'scenes' in blend_dir:
    # We are in a copied blend file in HyperBlend/scenes/scene_12345
    script_dir = os.path.abspath(blend_dir + '../../../src/blender_scripts')
    data_dir = os.path.abspath(blend_dir + '../../../src/data')
    forest_dir = os.path.abspath(blend_dir + '../../../src/forest')
else:
    # We are in the template forest blend file
    script_dir = os.path.abspath(blend_dir + '/src/blender_scripts')
    data_dir = os.path.abspath(blend_dir + '/src/data')
    forest_dir = os.path.abspath(blend_dir + '/src/forest')

# After this is set, any script in /blender_scripts can be imported
if script_dir not in sys.path:
    sys.path.append(script_dir)
if data_dir not in sys.path:
    sys.path.append(data_dir)
if forest_dir not in sys.path:
    sys.path.append(forest_dir)

import forest_constants as FC
import forest_utils as FU
import file_names as FN
import path_handling as PH
import sun
importlib.reload(FC)
importlib.reload(FU)
importlib.reload(FN)
importlib.reload(PH)
importlib.reload(sun)

b_context = bpy.context
b_data = bpy.data
b_ops = bpy.ops
b_scene = b_data.scenes['Forest']

cameras = b_data.collections['Cameras'].all_objects
lights = b_data.collections['Lighting'].all_objects
trees = b_data.collections['Trees'].all_objects
markers = b_data.collections['Marker objects'].all_objects
forest = b_data.collections['Ground'].all_objects.get('Ground')

forest_geometry_node = forest.modifiers['GeometryNodes'].node_group.nodes.get('Group.004')


def random_sun(rand_state):
    """Set sun parameters to random values to create unique scenes.

    Parameter ranges are currently hard-coded.

    TODO allow changing parameter ranges when the script is called.
    """

    random.setstate(rand_state)

    set_sun_angle(random.uniform(0,40), random.uniform(0,360))

    return random.getstate()


def set_sun_angle(elevation_deg=0, azimuth=90):
    """Set the sun elevation and azimuth angles in degrees.

    :param elevation_deg:
        Elevation angle between [0,90] degrees. Value 0 is sun zenith.
    :param azimuth:
        Azimuth angle between [0,360] degrees. With default value (90), the sun is shining directly from right.
    :return:
    """
    if elevation_deg < 0:
        elevation_deg = 0
    if elevation_deg > 90:
        elevation_deg = 90
    if azimuth < 0:
        azimuth = 0
    if azimuth > 360:
        azimuth = 360

    sun = lights.get(FC.key_obj_sun)
    sun.rotation_euler = (math.radians(elevation_deg), 0, math.radians(azimuth))


def set_sun_power_for_all(bands, irradiances):
    for i,band in enumerate(bands):
        set_sun_power(irradiances[i], band)


def set_sun_power(value, frame):
    """Set sun power in W/m2/Xnm, where X is the bandwith in nanometers. """

    # sun = lights.get(FC.key_obj_sun)
    bpy.data.lights["Sun"].energy = value
    dp = f'energy'
    bpy.data.lights["Sun"].keyframe_insert(dp, frame=frame)
    # dp = f'nodes["Emission"].inputs["Strength"].default_value'
    # sun.node_tree.nodes["Emission"].inputs['Strength'].default_value = value
    # sun.node_tree.keyframe_insert(dp, frame=frame)


def set_leaf_material(leaf_index, band_list, ad_list, sd_list, ai_list, mf_list):
    """Set leaf material for all frames in band list.

    :param leaf_index:
    :param band_list:
    :param ad_list:
    :param sd_list:
    :param ai_list:
    :param mf_list:
    :return:
    """

    def set_leaf_material_parameter_per_frame(parameter, value, frame):
        material = bpy.data.materials[f"Leaf material {leaf_index}"]
        dp = f'nodes["Group"].inputs["{parameter}"].default_value'
        material.node_tree.nodes["Group"].inputs[f"{parameter}"].default_value = value
        material.node_tree.keyframe_insert(dp, frame=frame)

    for i,band in enumerate(band_list):
        set_leaf_material_parameter_per_frame('Absorption density', ad_list[i], band)
        set_leaf_material_parameter_per_frame('Scattering density', sd_list[i], band)
        set_leaf_material_parameter_per_frame('Scattering anisotropy', ai_list[i], band)
        set_leaf_material_parameter_per_frame('Mix factor', mf_list[i], band)
        set_leaf_material_parameter_per_frame('Density scale', 300, band)  # TODO what should we do with this


def read_leaf_material_csv(leaf_index=1):
    p = PH.join(PH.path_directory_forest_scene(scene_id), FN.filename_leaf_material_csv(leaf_index))

    if not os.path.exists(p):
        raise FileNotFoundError(f"Leaf csv file '{p}' not found. Check your files and indexes given to setup script.")

    leaf_index
    band_list = []
    wl_list = []
    ad_list = []
    sd_list = []
    ai_list = []
    mf_list = []
    with open(p) as file:
        reader = csv.reader(file, delimiter=' ')
        for row in reader:
            try:
                band_list.append(int(row[0]))
                wl_list.append(float(row[1]))
                ad_list.append(float(row[2]))
                sd_list.append(float(row[3]))
                ai_list.append(float(row[4]))
                mf_list.append(float(row[5]))
            except ValueError:
                # this is ok
                # print(f"Material headers: {row}")
                pass

    return band_list, wl_list, ad_list, sd_list, ai_list, mf_list


def set_animation_frames(band_count):
    for scene in b_data.scenes:
        scene.render.fps = 5 # does not really matter
        scene.frame_start = 1
        scene.frame_end = band_count # can be safely set to zero as Blender will fix it into one


def get_leaf_bandwith_and_bandcount():

    if leaf_ids is None or len(leaf_ids) < 1:
        raise RuntimeWarning(f"No leaf indices to check.")

    previous_band_list, previous_wl_list, _, _, _, _ = read_leaf_material_csv(leaf_ids[0])

    if len(leaf_ids) == 1:
        bandwith =  previous_wl_list[1] - previous_wl_list[0]
        return bandwith, previous_band_list

    all_ok = True
    for i in range(1,len(leaf_ids)):
        band_list, wl_list, _, _, _, _ = read_leaf_material_csv(leaf_ids[i])
        close_enough_band = np.allclose(previous_band_list, band_list)
        close_enough_wl = np.allclose(previous_wl_list, wl_list)
        all_ok = all_ok and close_enough_band and close_enough_wl
        previous_band_list = band_list
        previous_wl_list = wl_list

    if all_ok:
        bandwith = wl_list[1] - wl_list[0]
        return bandwith, band_list
    else:
        raise RuntimeError(f"Bands and wavelengths of one or more leaves do not match each other.")



if __name__ == '__main__':


    # Store arguments passed from blender_control.py
    argv = sys.argv

    if "--" not in argv:
        argv = []  # no arguments for the script
    else:
        argv = argv[argv.index("--") + 1:]  # get all args after "--"

    # Argument names
    key_scene_id = ['-id', '--scene_id']
    key_sun_filename = ['-sun', '--sun_filename']
    key_leaf_ids = ['-l_ids', '--leaf_ids']
    # TODO resampling

    parser = argparse.ArgumentParser()

    parser.add_argument(key_scene_id[0], key_scene_id[1], dest=key_scene_id[1], action="store",
                        required=True, help="Directory containing the Blend file to be operated on.")
    parser.add_argument(key_sun_filename[0], key_sun_filename[1], dest=key_sun_filename[1], action="store",
                        required=False, help=".")
    parser.add_argument(key_leaf_ids[0], key_leaf_ids[1], dest=key_leaf_ids[1], action="store",
                        required=False, type=str, help="List of available leaf indexes as a string.")

    args = parser.parse_args(argv)

    scene_id = vars(args)[key_scene_id[1]]
    sun_filename = vars(args)[key_sun_filename[1]]
    leaf_ids = vars(args)[key_leaf_ids[1]]
    if leaf_ids is not None:
        leaf_ids = [int(id) for id in (leaf_ids.lstrip('[').rstrip(']')).split(', ')]

    try:

        bandwidth, band_list = get_leaf_bandwith_and_bandcount() # TODO resampling

        logging.error(f"Automatically detected bandwidth {bandwidth} nm and band count {len(band_list)}.")
        set_animation_frames(len(band_list))

        wls, irradiances = sun.load_light(file_name=sun_filename)  # TODO load Blender-ready sun csv
        logging.error(f"Spectral range from {wls[0]:.1f} nm to {wls[-1]:.1f} nm")

        # "Exposure": Scale values with magical constant to avoid overexposure. Tested with 10% white reflectance panel.
        max_irr = np.max(irradiances)
        factor = 35 / max_irr
        irradiances = irradiances * factor

        set_sun_power_for_all(band_list, irradiances)
    except RuntimeWarning as e:
        logging.error(f"Could not automatically detect bandwidth and band count from leaf data.")
        logging.error(e)

    bpy.data.scenes["Forest"].cycles.use_denoising = False

    logging.error(f"Hello, I am forest setup script in '{PH.path_directory_forest_scene(scene_id)}'")

    # FU.print_collection_items('Cameras')
    # FU.print_collection_items('Lighting')
    # FU.print_collection_items('Trees')
    # FU.print_collection_items('Ground')
    # FU.print_materials()
    # FU.list_tree_parameter_names()
    # FU.set_tree_parameter(1, 'Tree length', 11.0)
    # FU.list_forest_parameters()
    # FU.set_forest_parameter('Grid density', 5)
    # FU.set_rendering_parameters()
    # set_sun_angle(60)
    # framing_material()
    # FU.list_forest_parameters()

    # random.seed()
    # rand_state = random.getstate()
    # rand_state = random_sun(rand_state)
    #
    # rand_state = FU.random_ground(rand_state)
    #
    # rand_state = FU.random_tree(1, rand_state)
    # rand_state = FU.random_tree(2, rand_state)
    # FU.random_tree(3, rand_state)
    #
    # read_leaf_material_csv(1)
    # read_leaf_material_csv(2)
    # read_leaf_material_csv(3)

    # set_sun_power(1,1)
    # set_sun_power(2,100)
    #
    bpy.ops.wm.save_as_mainfile(filepath=PH.path_file_forest_scene(scene_id))

    # TODO set Cycles
    # TODO set rendering parameters (image size, sample count...)
    # TODO set GPU compute
    # TODO set camera position and rotation
    # TODO set sun angle
    # TODO set material parameters
    # TODO set tree parameters
    # TODO set color space and screen parameters
    # TODO how to disable using User preferences?
