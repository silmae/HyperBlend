"""

"""

# bpy stands for Blender Python, which is included Blender's own Python environment.
# It exists inside Blender, so don't worry if your IDE flags it as not found.
import bpy
import os
import sys  # to get command line args
import argparse  # to parse options for us and print a nice help message
import logging

# import settings as s

C = bpy.context
D = bpy.data
O = bpy.ops
S = D.scenes['Scene']
cameraList = D.collections['Cameras'].all_objects
# project_path = 'C:/Users/kiauriih/Code/Python/LeafBlend'

# Target names in the blender file
target_leaf = 'Leaf plate'
target_refl_ref = 'Reflectance white'
target_tran_ref = 'Transmittance white'

# Camera names
cam_name_refl = 'refl'
cam_name_tran = 'tran'

imaging_type_refl = 'refl'
imaging_type_tran = 'tran'


def set_camera():
    """Set reflectance camera active."""
    for cam in cameraList:
        if cam.name == cam_name_refl:
            C.scene.camera = cam


def render_reflectance(wl, value):

    set_camera()

    bpy.data.materials["Reflectance white"].node_tree.nodes["Combine RGB"].mode = 'HSV'
    bpy.data.materials["Reflectance white"].node_tree.nodes["Combine RGB"].inputs[0].default_value = 0
    bpy.data.materials["Reflectance white"].node_tree.nodes["Combine RGB"].inputs[1].default_value = 0
    bpy.data.materials["Reflectance white"].node_tree.nodes["Combine RGB"].inputs[2].default_value = value

    image_name = f'refl_wl_{wl:.2f}.tif'
    file_path = os.path.normpath(f'{RENDER_PATH}/{image_name}')
    S.render.filepath = file_path

    # This loop will break in Blender 2.8
    # for obj in D.collections['Targets'].all_objects:
    #     logging.warning(f"Rendering target: {obj.name} ({obj.visible_get()})")
        # obj.hide_render = True
        # obj.hide_viewport = True
    # Hide individually instead
    D.collections['Targets'].all_objects['Leaf plate'].hide_render = True
    D.collections['Targets'].all_objects['Reflectance white'].hide_render = True
    D.collections['Targets'].all_objects['Transmittance white'].hide_render = True

    target_obj = D.objects[target_refl_ref]
    target_obj.hide_render = False
    target_obj.hide_viewport = False

    if not DRY_RUN:
        # logging.warning(f'Saving render to "{file_path}"')
        O.render.render(write_still=True)
    else:
        logging.warning(f'Faking to save render to "{file_path}"')


if __name__ == '__main__':

    # Store arguments passed from blender_control.py
    argv = sys.argv

    if "--" not in argv:
        argv = []  # no arguments for the script
    else:
        argv = argv[argv.index("--") + 1:]  # get all args after "--"

    # Argument names
    key_base_path = ['-p', '--base_path']
    key_dry_run = ['-y', '--dry_run']
    key_sun_power = ['-s', '--sun_power']

    parser = argparse.ArgumentParser()

    parser.add_argument(key_base_path[0], key_base_path[1], dest=key_base_path[1], action="store",
                        required=True, help="Base path to render folders. Under it have to be folders for "
                                            "/rend, /rend_ref_refl, and rend_ref_tran")

    parser.add_argument(key_dry_run[0], key_dry_run[1], dest=key_dry_run[1], action="store_true",
                        required=False, help="If given, nothing is saved but the prints will come out.")

    parser.add_argument(key_sun_power[0], key_sun_power[1], dest=key_sun_power[1], action="store", type=float,
                        required=True,
                        help="Sun power in W/m^2.")

    args = parser.parse_args(argv)

    DRY_RUN = vars(args)[key_dry_run[1]]
    base_path = vars(args)[key_base_path[1]]
    sun_power = vars(args)[key_sun_power[1]]

    bpy.data.lights["Light"].energy = sun_power

    RENDER_PATH = base_path + '/rend'

    n = 101
    d_val = 1/n
    for i in range(n):
        render_reflectance(wl=i, value=i*d_val)
