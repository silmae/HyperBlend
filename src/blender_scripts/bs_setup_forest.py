# bpy stands for Blender Python, which is included Blender's own Python environment.
# It exists inside Blender, so don't worry if your IDE flags it as not found.
import bpy
import os
import sys  # to get command line args
import argparse  # to parse options for us and print a nice help message
import logging

b_context = bpy.context
b_data = bpy.data
b_ops = bpy.ops
b_scene = b_data.scenes['Scene']
b_cameras = b_data.collections['Cameras'].all_objects
b_lights = b_data.collections['Lighting'].all_objects
b_trees = b_data.collections['Trees'].all_objects


def print_collection_items(collection_name):
    print(f'{collection_name} collection contains:')
    for item in b_data.collections[collection_name].all_objects:
        print('\t' + item.name)


if __name__ == '__main__':

    # Store arguments passed from blender_control.py
    argv = sys.argv

    if "--" not in argv:
        argv = []  # no arguments for the script
    else:
        argv = argv[argv.index("--") + 1:]  # get all args after "--"

    # Argument names
    key_base_path = ['-p', '--base_path']

    parser = argparse.ArgumentParser()

    parser.add_argument(key_base_path[0], key_base_path[1], dest=key_base_path[1], action="store",
                        required=True, help="Blend file's path")

    args = parser.parse_args(argv)

    base_path = vars(args)[key_base_path[1]]

    logging.error(f"Hello, I am forest setup script in '{base_path}'")
    print_collection_items('Cameras')
    print_collection_items('Lighting')
    print_collection_items('Trees')
    print_collection_items('Ground')
    # bpy.ops.wm.save_as_mainfile(filepath=base_path)
