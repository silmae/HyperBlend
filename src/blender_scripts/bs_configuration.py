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

if __name__ == "__main__":

    blend_dir = os.path.dirname(os.path.abspath(bpy.data.filepath))

    if "System simulation" in blend_dir:
        # We are in a copied blend file in HyperBlend/System simulation/scene_12345
        script_dir = os.path.abspath(blend_dir + "/../../../src/blender_scripts")
        data_dir = os.path.abspath(blend_dir + "/../../../src/data")
        forest_dir = os.path.abspath(blend_dir + "/../../../src/system_simulation")
    else:
        # We are in the template system_simulation blend file
        script_dir = os.path.abspath(blend_dir + "/../src/blender_scripts")
        data_dir = os.path.abspath(blend_dir + "/../src/data")
        forest_dir = os.path.abspath(blend_dir + "/../src/system_simulation")

    # After this is set, any script in /blender_scripts can be imported
    if script_dir not in sys.path:
        sys.path.append(script_dir)
    if data_dir not in sys.path:
        sys.path.append(data_dir)
    if forest_dir not in sys.path:
        sys.path.append(forest_dir)

    from src.blender_scripts import forest_constants as FC
    from src.blender_scripts import forest_utils as FU
    from src.blender_scripts import forest_control

    importlib.reload(FC)
    importlib.reload(FU)
    importlib.reload(forest_control)

    b_context = bpy.context
    b_data = bpy.data
    b_ops = bpy.ops
    b_scene = b_data.scenes[FC.key_scene_name]

    # Store arguments passed from blender_control.py
    argv = sys.argv

    if "--" not in argv:
        argv = []  # no arguments for the script
    else:
        argv = argv[argv.index("--") + 1 :]  # get all args after "--"

    # Argument names
    key_scene_id = ["-id", "--scene_id"]
    key_global_master = ["-g", "--global_master"]
    key_generate = ["-e", "--generate"]

    parser = argparse.ArgumentParser()

    parser.add_argument(
        key_scene_id[0],
        key_scene_id[1],
        dest=key_scene_id[1],
        action="store",
        required=False,
        help="Name of the scene for which the scene control file will be generated.",
    )
    parser.add_argument(
        key_global_master[0],
        key_global_master[1],
        dest=key_global_master[1],
        action="store_true",
        required=False,
        help="If True, a global scene configuration file is generated to "
        "project root/Internal/. This will also ignore the scene id parameter.",
    )
    parser.add_argument(
        key_generate[0],
        key_generate[1],
        dest=key_generate[1],
        action="store_true",
        required=False,
        help="If True (default), generate system simulation control file. If False, apply the control file to the scene.",
    )

    args = parser.parse_args(argv)

    scene_id = vars(args)[key_scene_id[1]]
    global_master = vars(args)[key_global_master[1]]
    generate = vars(args)[key_generate[1]]

    if generate:
        if global_master:
            logging.error(f"Generating global master scene control file.")
        else:
            logging.error(f"Generating ordinary master scene control file.")

        scene_dict = FU.get_scene_parameters(as_master=True)

        forest_control.write_forest_control(
            system_sim_name=scene_id,
            control_dict=scene_dict,
            global_master=global_master,
        )
    else:
        if global_master:
            logging.error(f"Applying global master scene control file.")
        else:
            logging.error(f"Applying ordinary master scene control file.")

        FU.apply_forest_control(system_sim_name=scene_id, global_master=global_master)

        from src.data import path_handling as PH

        importlib.reload(PH)

        if global_master:
            filepath = PH.file_blend_system_simulation_template()
        else:
            filepath = PH.file_blend_system_simulation(simulation_name=scene_id)

        # Save changes to the Blender file
        bpy.ops.wm.save_as_mainfile(filepath=filepath)
