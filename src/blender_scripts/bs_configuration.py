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
import forest_control

importlib.reload(FC)
importlib.reload(FU)
importlib.reload(forest_control)

b_context = bpy.context
b_data = bpy.data
b_ops = bpy.ops
b_scene = b_data.scenes[FC.key_scene_name]

if __name__ == '__main__':

    # Store arguments passed from blender_control.py
    argv = sys.argv

    if "--" not in argv:
        argv = []  # no arguments for the script
    else:
        argv = argv[argv.index("--") + 1:]  # get all args after "--"

    # Argument names
    key_scene_id = ['-id', '--scene_id']
    key_global_master = ['-g', '--global_master']

    parser = argparse.ArgumentParser()

    parser.add_argument(key_scene_id[0], key_scene_id[1], dest=key_scene_id[1], action="store",
                        required=False, help="Name of the scene for which the scene control file will be generated.")
    parser.add_argument(key_global_master[0], key_global_master[1], dest=key_global_master[1], action="store_true",
                        required=False,
                        help="If True, a global scene configuration file is generated to "
                        "project root. This will also ignore the scene id parameter.")

    args = parser.parse_args(argv)

    scene_id = vars(args)[key_scene_id[1]]
    global_master = vars(args)[key_global_master[1]]

    if global_master:
        logging.error(f"Generating global master scene control file.")
    else:
        logging.error(f"Generating ordinary master scene control file.")

    scene_dict = FU.get_scene_parameters(as_master=True)

    forest_control.write_forest_control(forest_id=scene_id, control_dict=scene_dict, global_master=global_master)
