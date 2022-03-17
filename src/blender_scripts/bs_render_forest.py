import bpy
import os
import sys  # to get command line args
import argparse  # to parse options for us and print a nice help message
import logging
import math

"""
This is how one can import stuff to Blender

blend_dir = os.path.dirname(os.path.abspath(bpy.data.filepath))
script_dir = os.path.abspath(blend_dir + '/src/blender_scripts')
if script_dir not in sys.path:
    print(script_dir)
    sys.path.append(script_dir)
    

import bs_render_forest
import imp
imp.reload(bs_render_forest)
print(bs_render_forest.key_obj_sun)
"""

b_context = bpy.context
b_data = bpy.data
b_ops = bpy.ops
b_scene = b_data.scenes['Forest']

cameras = b_data.collections['Cameras'].all_objects
lights = b_data.collections['Lighting'].all_objects
trees = b_data.collections['Trees'].all_objects
markers = b_data.collections['Marker objects'].all_objects
ground = b_data.collections['Ground'].all_objects

forest = b_data.collections['Ground'].all_objects.get('Ground')

forest_geometry_node = forest.modifiers['GeometryNodes'].node_group.nodes.get('Group.004')

########### Object names ###########

key_obj_sun = 'Sun'

key_obj_ground = 'Ground'
key_obj_ground_test = 'Test ground'

key_obj_tree_1 = 'Tree 1'
key_obj_tree_2 = 'Tree 2'
key_obj_tree_3 = 'Tree 3'

key_obj_marker_1 = 'Marker 1'
key_obj_marker_2 = 'Marker 2'
key_obj_marker_3 = 'Marker 3'

key_cam_drone_hsi = 'Drone HSI'
key_cam_drone_rgb = 'Drone RGB'
key_cam_walker_rgb = 'Walker RGB'
key_cam_sleeper_rgb = 'Sleeper RGB'

########################################


def set_render_parameters(render_mode: str='spectral', camera: str='Drone RGB', res_x=512, res_y=512, res_percent=100):
    """Jau

    :param render_mode:
        Either 'spectral' or 'rgb'.
    :param res_x:
    :param res_y:
    :param res_percent:
    :return:
    """

    # just in case we have multiple scenes at some point loop them over
    for scene in b_data.scenes:

        scene.sequencer_colorspace_settings.name = 'Raw'
        # Video sequenser can be always set to Raw as it only affects video editing

        if render_mode.lower() == 'spectral':

            scene.render.image_settings.file_format = 'TIFF'  # OK
            scene.render.image_settings.tiff_codec = 'NONE'
            scene.render.image_settings.color_mode = 'BW'  # 'BW', 'RGB', 'RGBA'

            scene.camera = cameras.get('Drone HSI')

            scene.display_settings.display_device = 'None'
            scene.view_settings.view_transform = 'Raw'
            scene.view_settings.look = 'None'
            scene.view_settings.exposure = 0
            scene.view_settings.gamma = 1

        elif render_mode.lower() == 'rgb':

            scene.render.image_settings.file_format = 'PNG'  # OK
            scene.render.image_settings.compression = 15 # percent packing
            scene.render.image_settings.color_mode = 'RGB'  # 'BW', 'RGB', 'RGBA'

            if camera in cameras:
                scene.camera = cameras.get(camera)
            else:
                raise AttributeError(f"Camera '{camera} is not in camera list {cameras}.")

            scene.display_settings.display_device = 'sRGB'
            scene.view_settings.view_transform = 'Filmic Log'
            scene.view_settings.look = 'None'
            scene.view_settings.exposure = 0
            scene.view_settings.gamma = 1
        else:
            raise AttributeError(f"Parameter render_mode in set_render_parameters() must be either 'spectral' or 'rgb'. Was '{render_mode}'.")

        scene.render.image_settings.color_depth = '16'


        scene.render.resolution_x = res_x # OK
        scene.render.resolution_y = res_y # OK

        if res_percent > 100:
            res_percent = 100
        if res_percent < 1:
            res_percent = 1
        scene.render.resolution_percentage = res_percent # OK




        scene.render.filepath = rend_path

        scene.render.use_persistent_data = True
        # Keep render data around for faster re-renders and animation renders, at the cost of increased memory usage

        scene.render.engine = 'CYCLES' # do not use 'BLENDER_EEVEE'  # OK
        scene.cycles.samples = 16

        # Set the device_type
        b_context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA" # or "OPENCL"

        # Set the device and feature set
        b_context.scene.cycles.device = "GPU"

        # get_devices() to let Blender detects GPU device
        b_context.preferences.addons["cycles"].preferences.get_devices()
        print(bpy.context.preferences.addons["cycles"].preferences.compute_device_type)
        for d in bpy.context.preferences.addons["cycles"].preferences.devices:
            d["use"] = 1 # Using all devices, include GPU and CPU
            print(d["name"], d["use"])


def set_visibility(mode: str):

    def hide(obj):
        print(f"Hiding object '{obj.name}'.")
        obj.hide_render = True
        obj.hide_set(True)

    def unhide(obj):
        print(f"Unhiding object '{obj.name}'.")
        obj.hide_render = False
        obj.hide_set(False)

    if mode != key_cam_sleeper_rgb and mode != key_cam_walker_rgb and mode != key_cam_drone_rgb and mode != 'Map' and mode != key_cam_drone_hsi and mode != 'Trees':
        raise AttributeError(f"Visibility for mode '{mode}' not recognised.")

    # First hide everything
    for object in markers:
        hide(object)
    for object in trees:
        hide(object)
    for object in ground:
        hide(object)

    unhide(lights.get(key_obj_sun)) # always show sun

    if mode == key_cam_sleeper_rgb or mode == key_cam_walker_rgb or mode == key_cam_drone_rgb or mode == 'Map':
        unhide(ground.get(key_obj_ground))
    elif mode == 'Trees':
        unhide(ground.get(key_obj_ground_test))
        for tree in trees:
            unhide(tree)


def set_input(node, input_name, value):
    input = node.inputs.get(input_name)
    if input == None:
        raise AttributeError(f"Parameter called '{input_name}' seems not to exist. Check the name.")
    old_val = input.default_value
    input.default_value = value
    print(f"{node.name}: parameter {input.name} value changed from {old_val} to {value}.")


def set_forest_parameter(parameter_name, value):
    """
    Side length (VALUE)
    Grid density (INT)
    Use real object (BOOLEAN)
    Spawn point density (INT)
    Tree 1 density (INT)
    Tree 2 density (INT)
    Tree 3 density (INT)
    Seed (INT)
    Hill height (VALUE)
    Hill scale (VALUE)

    :param parameter_name:
    :param value:
    :return:
    """

    set_input(forest_geometry_node, parameter_name, value)


def render_sleeper_rgb():

    set_render_parameters(render_mode='rgb', camera='Sleeper RGB', res_x=1028, res_y=512, res_percent=100)
    set_visibility(mode='Sleeper RGB')
    set_forest_parameter('Use real object', True)
    image_name = f'sleeper_rgb.png'
    image_path = os.path.normpath(f'{rend_path}/{image_name}')
    logging.info(f"Trying to render '{image_path}'.")
    b_scene.render.filepath = image_path
    b_ops.render.render(write_still=True)


def render_walker_rgb():

    set_render_parameters(render_mode='rgb', camera='Walker RGB', res_x=1028, res_y=512, res_percent=100)
    set_visibility(mode='Walker RGB')
    set_forest_parameter('Use real object', True)
    image_name = f'walker_rgb.png'
    image_path = os.path.normpath(f'{rend_path}/{image_name}')
    logging.info(f"Trying to render '{image_path}'.")
    b_scene.render.filepath = image_path
    b_ops.render.render(write_still=True)


def render_drone_rgb():

    set_render_parameters(render_mode='rgb', camera='Drone RGB', res_x=1028, res_y=512, res_percent=100)
    set_visibility(mode='Drone RGB')
    set_forest_parameter('Use real object', True)
    image_name = f'drone_rgb.png'
    image_path = os.path.normpath(f'{rend_path}/{image_name}')
    logging.info(f"Trying to render '{image_path}'.")
    b_scene.render.filepath = image_path
    b_ops.render.render(write_still=True)


def render_map_rgb():

    set_render_parameters(render_mode='rgb', camera='Drone RGB', res_x=1028, res_y=512, res_percent=100)
    set_visibility(mode='Drone RGB')
    set_forest_parameter('Use real object', False)
    image_name = f'map_rgb.png'
    image_path = os.path.normpath(f'{rend_path}/{image_name}')
    logging.info(f"Trying to render '{image_path}'.")
    b_scene.render.filepath = image_path
    b_ops.render.render(write_still=True)


if __name__ == '__main__':

    # Store arguments passed from blender_control.py
    argv = sys.argv

    if "--" not in argv:
        argv = []  # no arguments for the script
    else:
        argv = argv[argv.index("--") + 1:]  # get all args after "--"

    # Argument names
    key_base_path = ['-p', '--base_path']
    key_blend_file_name = ['-fn', '--blend_file_name']
    key_render_mode = ['-rm', '--render_mode']

    parser = argparse.ArgumentParser()

    parser.add_argument(key_base_path[0], key_base_path[1], dest=key_base_path[1], action="store",
                        required=True, help="Directory containing the Blend file to be operated on.")
    parser.add_argument(key_blend_file_name[0], key_blend_file_name[1], dest=key_blend_file_name[1], action="store",
                        required=True, help="Blender file name that is found from the Base path.")
    parser.add_argument(key_render_mode[0], key_render_mode[1], dest=key_render_mode[1], action="store",
                        required=True, help="")

    args = parser.parse_args(argv)

    base_path = vars(args)[key_base_path[1]]
    print(f"base_path = {base_path}")
    blend_file_name = vars(args)[key_blend_file_name[1]]
    print(f"blend_file_name = {blend_file_name}")
    rend_path = os.path.abspath(base_path + '/rend') + os.path.sep
    print(f"rend_path = {rend_path}")
    file_path = os.path.abspath(base_path + '/' + blend_file_name)
    print(f"file_path = {file_path}")
    render_mode = vars(args)[key_render_mode[1]]

    if render_mode == 'Sleeper':
        render_sleeper_rgb()
        render_walker_rgb()
        render_drone_rgb()
        render_map_rgb()
    else:
        logging.error(f"Render mode '{render_mode}' not recognised.")

    logging.error(f"Hello, I am forest render script in '{base_path}'")
