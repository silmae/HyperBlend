import bpy
import os
import sys  # to get command line args
import argparse  # to parse options for us and print a nice help message
import logging
import importlib

blend_dir = os.path.dirname(os.path.abspath(bpy.data.filepath))

if 'scenes' in blend_dir:
    # We are in a copied blend file in HyperBlend/scenes/scene_12345
    script_dir = os.path.abspath(blend_dir + '../../../src/blender_scripts')
    data_dir = os.path.abspath(blend_dir + '../../../src/data')
else:
    # We are in the template forest blend file
    script_dir = os.path.abspath(blend_dir + '/src/blender_scripts')
    data_dir = os.path.abspath(blend_dir + '/src/data')

# After this is set, any script in /blender_scripts can be imported
if script_dir not in sys.path:
    sys.path.append(script_dir)
    sys.path.append(data_dir)

import forest_constants as FC
import forest_utils as FU
import file_names as FN
import path_handling as PH
importlib.reload(FC)
importlib.reload(FU)
importlib.reload(FN)
importlib.reload(PH)


b_context = bpy.context
b_data = bpy.data
b_ops = bpy.ops
b_scene = b_data.scenes[FC.key_scene_name]

cameras = b_data.collections[FC.key_collection_cameras].all_objects
lights = b_data.collections[FC.key_collection_lights].all_objects
trees = b_data.collections[FC.key_collection_trees].all_objects
markers = b_data.collections[FC.key_collection_markers].all_objects
ground = b_data.collections[FC.key_collection_ground].all_objects


def set_materials_use_spectral(use_spectral):

    bpy.data.materials["Leaf material 1"].node_tree.nodes["Group"].inputs["Use spectral"].default_value = use_spectral
    bpy.data.materials["Leaf material 2"].node_tree.nodes["Group"].inputs["Use spectral"].default_value = use_spectral
    bpy.data.materials["Leaf material 3"].node_tree.nodes["Group"].inputs["Use spectral"].default_value = use_spectral

    bpy.data.materials["Trunk material 1"].node_tree.nodes["Group"].inputs["Use spectral"].default_value = use_spectral
    bpy.data.materials["Trunk material 2"].node_tree.nodes["Group"].inputs["Use spectral"].default_value = use_spectral
    bpy.data.materials["Trunk material 3"].node_tree.nodes["Group"].inputs["Use spectral"].default_value = use_spectral

    bpy.data.materials["Ground material"].node_tree.nodes["Group"].inputs["Use spectral"].default_value = use_spectral

    if use_spectral:
        bpy.data.worlds["World"].node_tree.nodes["Group"].inputs['Strength'].default_value = 0
    else:
        bpy.data.worlds["World"].node_tree.nodes["Group"].inputs['Strength'].default_value = 2


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
            # scene.view_settings.view_transform = 'Raw' # cannot be used when display device is None as it already turns off the display color transformations
            scene.view_settings.look = 'None'
            scene.view_settings.exposure = 0
            scene.view_settings.gamma = 1

            set_materials_use_spectral(True)

            # disable sky for spectral images

            bpy.data.node_groups["Ground geometry"].nodes["Group.004"].inputs['Show white reference'].default_value = True

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

            set_materials_use_spectral(False)

            bpy.data.node_groups["Ground geometry"].nodes["Group.004"].inputs['Show white reference'].default_value = False

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

    if mode != FC.key_cam_sleeper_rgb and mode != FC.key_cam_walker_rgb and mode != FC.key_cam_drone_rgb and mode != 'Map' and mode != FC.key_cam_drone_hsi and mode != FC.key_cam_tree_rgb:
        raise AttributeError(f"Visibility for mode '{mode}' not recognised.")

    # First hide everything
    for object in markers:
        hide(object)
    for object in trees:
        hide(object)
    for object in ground:
        hide(object)

    unhide(lights.get(FC.key_obj_sun)) # always show sun

    if mode == FC.key_cam_sleeper_rgb or mode == FC.key_cam_walker_rgb or mode == FC.key_cam_drone_rgb or mode == 'Map' or mode == FC.key_cam_drone_hsi:
        unhide(ground.get(FC.key_obj_ground))
    elif mode == FC.key_cam_tree_rgb:
        unhide(ground.get(FC.key_obj_ground_test))
        for tree in trees:
            unhide(tree)


def render_sleeper_rgb():

    set_render_parameters(render_mode='rgb', camera='Sleeper RGB', res_x=1028, res_y=512, res_percent=100)
    set_visibility(mode='Sleeper RGB')
    FU.set_forest_parameter('Use real object', True)
    image_name = f'sleeper_rgb.png'
    image_path = PH.join(PH.path_directory_forest_rend(scene_id), image_name)
    # image_path = os.path.normpath(f'{rend_path}/{image_name}')
    logging.info(f"Trying to render '{image_path}'.")
    b_scene.render.filepath = image_path
    b_ops.render.render(write_still=True)


def render_walker_rgb():

    set_render_parameters(render_mode='rgb', camera='Walker RGB', res_x=1028, res_y=512, res_percent=100)
    set_visibility(mode='Walker RGB')
    FU.set_forest_parameter('Use real object', True)
    image_name = f'walker_rgb.png'
    image_path = PH.join(PH.path_directory_forest_rend(scene_id), image_name)
    # image_path = os.path.normpath(f'{rend_path}/{image_name}')
    logging.info(f"Trying to render '{image_path}'.")
    b_scene.render.filepath = image_path
    b_ops.render.render(write_still=True)


def render_drone_rgb():

    set_render_parameters(render_mode='rgb', camera='Drone RGB', res_x=1028, res_y=512, res_percent=100)
    set_visibility(mode='Drone RGB')
    FU.set_forest_parameter('Use real object', True)
    image_name = f'drone_rgb.png'
    image_path = PH.join(PH.path_directory_forest_rend(scene_id), image_name)
    # image_path = os.path.normpath(f'{rend_path}/{image_name}')
    logging.info(f"Trying to render '{image_path}'.")
    b_scene.render.filepath = image_path
    b_ops.render.render(write_still=True)


def render_tree_rgb():

    set_render_parameters(render_mode='rgb', camera='Tree RGB', res_x=1028, res_y=512, res_percent=100)
    set_visibility(mode='Tree RGB')
    # FU.set_forest_parameter('Use real object', True)
    image_name = f'tree_rgb.png'
    image_path = PH.join(PH.path_directory_forest_rend(scene_id), image_name)
    # image_path = os.path.normpath(f'{rend_path}/{image_name}')
    logging.info(f"Trying to render '{image_path}'.")
    b_scene.render.filepath = image_path
    b_ops.render.render(write_still=True)


def render_map_rgb():

    set_render_parameters(render_mode='rgb', camera='Drone RGB', res_x=512, res_y=512, res_percent=100)
    set_visibility(mode='Drone RGB')
    FU.set_forest_parameter('Use real object', False)
    image_name = f'map_rgb.png'
    image_path = PH.join(PH.path_directory_forest_rend(scene_id), image_name)
    # image_path = os.path.normpath(f'{rend_path}/{image_name}')
    logging.info(f"Trying to render '{image_path}'.")
    b_scene.render.filepath = image_path
    b_ops.render.render(write_still=True)


def render_drone_hsi():
    set_render_parameters(render_mode='spectral', camera='Drone HSI', res_x=512, res_y=512, res_percent=100)
    set_visibility(mode='Drone HSI')
    FU.set_forest_parameter('Use real object', True)
    b_scene.render.filepath = PH.join(PH. path_directory_forest_rend_spectral(scene_id), "band_####.tiff")
    b_ops.render.render(write_still=True, animation=True)


if __name__ == '__main__':

    # Store arguments passed from blender_control.py
    argv = sys.argv

    if "--" not in argv:
        argv = []  # no arguments for the script
    else:
        argv = argv[argv.index("--") + 1:]  # get all args after "--"

    # Argument names
    key_scene_id = ['-id', '--scene_id']
    key_render_mode = ['-rm', '--render_mode']

    parser = argparse.ArgumentParser()

    parser.add_argument(key_scene_id[0], key_scene_id[1], dest=key_scene_id[1], action="store",
                        required=True, help="Scene id.")
    parser.add_argument(key_render_mode[0], key_render_mode[1], dest=key_render_mode[1], action="store",
                        required=True, help="")

    args = parser.parse_args(argv)

    scene_id = vars(args)[key_scene_id[1]]


    logging.error(f"Hello, I am forest render script in '{PH.path_directory_forest_scene(scene_id)}'")

    render_mode = vars(args)[key_render_mode[1]]

    if render_mode.lower() == 'preview':
        render_sleeper_rgb()
        render_walker_rgb()
        render_drone_rgb()
        # render_map_rgb()
        render_tree_rgb()
    elif render_mode.lower() == 'spectral':
        render_drone_hsi()
    else:
        logging.error(f"Render mode '{render_mode}' not recognised.")
