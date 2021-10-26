"""
This is the rendering script that Blender runs when evoked by blender_control.py.

Object names in this file and the Blender scene file must match. We cannot use the
constants.py file here as this script is run in Blender's own Python environment,
so all object names and such are hard coded.

Debug prints do not show if Blender output is directed to dev.null in blender_control.py.

TODO: explicitly set up render engine and such so that Blender's default setting that
are likely to change in future do not break the script.
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

class Range(object):
    """
    Range class for automatically checking input values by argparser.
    """

    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __eq__(self, other):
        return self.start <= other <= self.end

    def __contains__(self, item):
        return self.__eq__(item)

    def __iter__(self):
        yield self

    def __repr__(self):
        return '[{0},{1}]'.format(self.start, self.end)


def set_active_camera(cam_name):
    """Set rendering camera to be used. Must be either 'refl' or 'tran'"""

    for cam in cameraList:
        if cam.name == cam_name:
            C.scene.camera = cam


def get_active_camera():
    """Returns the camera that is currently in use."""

    return C.scene.camera


def toggle_cam():
    """Toggle from reflectance to transmittance camera and vice versa."""

    if get_active_camera().name == cam_name_refl:
        set_active_camera(cam_name_tran)
    else:
        set_active_camera(cam_name_refl)

    # print(f'Active camera is {C.scene.camera.name}')


def render_leaf(imaging_type, wl, abs_dens, scat_dens, scat_ai, mix_fac, dry_run=True):
    """Renders the leaf target with given arguments."""

    # Debug prints
    # print("Received parameters for render_leaf()")
    # print(f"wl: {wl}")
    # print(f"abs_dens: {abs_dens}")
    # print(f"scat_dens: {scat_dens}")
    # print(f"scat_ai: {scat_ai}")
    # print(f"mix_fac: {mix_fac}")
    # print(f"dry_run: {dry_run}")

    mat = D.materials['leaf_material']
    nodes = mat.node_tree.nodes
    volume_absoption_node = nodes.get('Volume Absorption')
    volume_scatter_node = nodes.get('Volume Scatter')
    volume_mix_node = nodes.get('Mix Volume')
    volume_absoption_node.inputs['Density'].default_value = abs_dens
    volume_scatter_node.inputs['Density'].default_value = scat_dens
    volume_scatter_node.inputs['Anisotropy'].default_value = scat_ai
    volume_mix_node.inputs['Fac'].default_value = mix_fac

    render_target(imaging_type, wl, target_leaf, render_path_leaf, dry_run=dry_run)


def render_reference(imaging_type, wl, dry_run=True):
    """Renders white reference in either reflectance or transmittance mode."""

    if imaging_type == imaging_type_refl:
        render_target(imaging_type, wl, target_refl_ref, render_path_refl_ref, dry_run=dry_run)
    elif imaging_type == imaging_type_tran:
        render_target(imaging_type, wl, target_tran_ref, render_path_tran_ref, dry_run=dry_run)


def render_target(imaging_type, wl, target_name, render_path, dry_run=True):
    """General rendering function."""

    if imaging_type == imaging_type_refl:
        set_active_camera(cam_name_refl)
    elif imaging_type == imaging_type_tran:
        set_active_camera(cam_name_tran)

    image_name = f'{imaging_type}_wl{wl:.2f}.tif'
    file_path = os.path.normpath(f'{render_path}/{image_name}')
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

    target_obj = D.objects[target_name]
    target_obj.hide_render = False
    target_obj.hide_viewport = False

    if not dry_run:
        # logging.warning(f'Saving render to "{file_path}"')
        O.render.render(write_still=True)
    else:
        logging.warning(f'Faking to save render to "{file_path}"')


def make_folders():
    """Create default folder if not exist."""

    if not os.path.exists(os.path.normpath(render_path_leaf)):
        os.mkdir(os.path.normpath(render_path_leaf))
    if not os.path.exists(os.path.normpath(render_path_refl_ref)):
        os.mkdir(os.path.normpath(render_path_refl_ref))
    if not os.path.exists(os.path.normpath(render_path_tran_ref)):
        os.mkdir(os.path.normpath(render_path_tran_ref))


def clear_folders(clear_reference=False):
    """Clear render folders. Reference folder are also cleared if clear_reference=True.

    Creates the folders if not exist, so it should be safe to call. This is mainly for
    debugging if run straight from Blender. Primarily, use clearing code in file_handling.py.
    """

    make_folders()
    print("Clearing old data")
    list(map(os.unlink, (os.path.join(render_path_leaf, f) for f in os.listdir(render_path_leaf))))
    if clear_reference:
        print("Clearing reflectance and transmittance reference data")
        list(map(os.unlink, (os.path.join(render_path_refl_ref, f) for f in os.listdir(render_path_refl_ref))))
        list(map(os.unlink, (os.path.join(render_path_tran_ref, f) for f in os.listdir(render_path_tran_ref))))


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
    key_clear = ['-c', '--clear']
    key_clear_refs = ['-cr', '--clear_references']
    key_render_references = ['-r', '--render_references']
    key_wavelength = ['-wl', '--wavelength']
    key_abs_dens = ['-da', '--abs_dens']
    key_scat_dens = ['-ds', '--scat_dens']
    key_scat_ai = ['-ai', '--scat_ai']
    key_mix_fac = ['-mf', '--mix_fac']

    parser = argparse.ArgumentParser()

    parser.add_argument(key_base_path[0], key_base_path[1], dest=key_base_path[1], action="store",
                        required=True, help="Base path to render folders. Under it have to be folders for "
                                            "/rend, /rend_ref_refl, and rend_ref_tran")

    parser.add_argument(key_dry_run[0], key_dry_run[1], dest=key_dry_run[1], action="store_true",
                        required=False, help="If given, nothing is saved but the prints will come out.")

    parser.add_argument(key_clear[0], key_clear[1], dest=key_clear[1], action="store_true",
                        required=False, help="Clear leaf render folder before starting new renders.")

    parser.add_argument(key_clear_refs[0], key_clear_refs[1], dest=key_clear_refs[1], action="store_true",
                        required=False, help="Clear all reference folders before starting new renders.")

    parser.add_argument(key_render_references[0], key_render_references[1], dest=key_render_references[1], action="store_true",
                        required=False, help="Render new references if illumination changes.")

    parser.add_argument(key_wavelength[0], key_wavelength[1], dest=key_wavelength[1], action="store", type=float,
                        required=True,
                        help="Wavelength to be rendered. Affects only the naming of resulting image file.")

    parser.add_argument(key_abs_dens[0], key_abs_dens[1], dest=key_abs_dens[1], action="store", type=float,
                        choices=Range(0.0, 1000.0),
                        required=True, help="Shader volume absorption node's density input.")

    parser.add_argument(key_scat_dens[0], key_scat_dens[1], dest=key_scat_dens[1], action="store", type=float,
                        choices=Range(0.0, 1000.0),
                        required=True, help="Shader volume scattering node's density input.")

    parser.add_argument(key_scat_ai[0], key_scat_ai[1], dest=key_scat_ai[1], action="store", type=float, choices=Range(-1.0, 1.0),
                        required=True, help="Shader volume scattering node's anisotropy input.")

    parser.add_argument(key_mix_fac[0], key_mix_fac[1], dest=key_mix_fac[1], action="store", type=float, choices=Range(0.0, 1.0),
                        required=True,
                        help="Mixing factor of absorption and scattering (0 for full absorption, 1 for scatter).")

    args = parser.parse_args(argv)

    dry_run = vars(args)[key_dry_run[1]]
    clear = vars(args)[key_clear[1]]
    clear_refs = vars(args)[key_clear_refs[1]]
    render_references = vars(args)[key_render_references[1]]
    wavelength = vars(args)[key_wavelength[1]]
    abs_dens = vars(args)[key_abs_dens[1]]
    scat_dens = vars(args)[key_scat_dens[1]]
    scat_ai = vars(args)[key_scat_ai[1]]
    mix_fac = vars(args)[key_mix_fac[1]]
    base_path = vars(args)[key_base_path[1]]

    render_path_leaf = base_path + '/rend'
    # logging.warning(f"Rendering to '{render_path_leaf}'.")
    render_path_refl_ref = base_path + '/rend_refl_ref'
    render_path_tran_ref = base_path + '/rend_tran_ref'

    ## Tested setting up rendering with GPU ###
    # from: https://blender.stackexchange.com/questions/104651/selecting-gpu-with-python-script
    #
    # NOTE this is slower on small rends than CPU
    #
    # bpy.data.scenes[0].render.engine = "CYCLES"
    #
    # # Set the device_type
    # bpy.context.preferences.addons[
    #     "cycles"
    # ].preferences.compute_device_type = "CUDA" # or "OPENCL"
    #
    # # Set the device and feature set
    # bpy.context.scene.cycles.device = "GPU"
    #
    # # get_devices() to let Blender detects GPU device
    # bpy.context.preferences.addons["cycles"].preferences.get_devices()
    # print(bpy.context.preferences.addons["cycles"].preferences.compute_device_type)
    # for d in bpy.context.preferences.addons["cycles"].preferences.devices:
    #     d["use"] = 1 # Using all devices, include GPU and CPU
    #     print(d["name"], d["use"])

    #################################################

    make_folders()

    if clear and not dry_run:
        clear_folders(clear_reference=clear_refs)

    render_leaf(imaging_type_refl, wavelength, abs_dens, scat_dens, scat_ai, mix_fac, dry_run=dry_run)
    render_leaf(imaging_type_tran, wavelength, abs_dens, scat_dens, scat_ai, mix_fac, dry_run=dry_run)

    if render_references:
        render_reference(imaging_type_refl, wavelength, dry_run=dry_run)
        render_reference(imaging_type_tran, wavelength, dry_run=dry_run)
