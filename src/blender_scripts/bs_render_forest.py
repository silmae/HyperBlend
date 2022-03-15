import bpy
import os
import sys  # to get command line args
import argparse  # to parse options for us and print a nice help message
import logging
import math

b_context = bpy.context
b_data = bpy.data
b_ops = bpy.ops
b_scene = b_data.scenes['Scene']

cameras = b_data.collections['Cameras'].all_objects
lights = b_data.collections['Lighting'].all_objects
trees = b_data.collections['Trees'].all_objects
markers = b_data.collections['Marker objects'].all_objects
forest = b_data.collections['Ground'].all_objects.get('Ground')

forest_geometry_node = forest.modifiers['GeometryNodes'].node_group.nodes.get('Group.004')


def set_render_parameters(render_mode: str='spectral', camera: str='Drone RGB', res_x=512, res_y=512, res_percent=100):
    """Jau

    :param render_mode:
        Either 'spectral' or 'rgb'.
    :param res_x:
    :param res_y:
    :param res_percent:
    :return:
    """
    bpy.data.materials["Leaf material 1"].node_tree.nodes["ColorRamp"].color_ramp.elements[1].position

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

    parser = argparse.ArgumentParser()

    parser.add_argument(key_base_path[0], key_base_path[1], dest=key_base_path[1], action="store",
                        required=True, help="Directory containing the Blend file to be operated on.")
    parser.add_argument(key_blend_file_name[0], key_blend_file_name[1], dest=key_blend_file_name[1], action="store",
                        required=True, help="Blender file name that is found from the Base path.")

    args = parser.parse_args(argv)

    base_path = vars(args)[key_base_path[1]]
    print(f"base_path = {base_path}")
    blend_file_name = vars(args)[key_blend_file_name[1]]
    print(f"blend_file_name = {blend_file_name}")
    rend_path = os.path.abspath(base_path + '/rend') + os.path.sep
    print(f"rend_path = {rend_path}")
    file_path = os.path.abspath(base_path + '/' + blend_file_name)
    print(f"file_path = {file_path}")

    logging.error(f"Hello, I am forest render script in '{base_path}'")
