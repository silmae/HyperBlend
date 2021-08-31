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

## Test setting up rendering with GPU ###
# from: https://blender.stackexchange.com/questions/104651/selecting-gpu-with-python-script
# NOTE this is slower on small rends than CPU
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

class Range(object):
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


argv = sys.argv

if "--" not in argv:
    argv = []  # no arguments for the script
else:
    argv = argv[argv.index("--") + 1:]  # get all args after "--"

# print(argv)

key_base_path = 'base_path'
key_dry_run = 'dry_run'
# key_single_wavelength = 'single_wavelength'
key_clear = 'clear'
key_clear_refs = 'clear_references'
key_render_references = 'render_references'
key_wavelength = 'wavelength'
key_abs_dens = 'abs_dens'
key_scat_dens = 'scat_dens'
key_scat_ai = 'scat_ai'
key_mix_fac = 'mix_fac'

parser = argparse.ArgumentParser()

parser.add_argument("-p", f"--{key_base_path}", dest=key_base_path, action="store",
                    required=True, help="Base path to render folders. Under it have to be folders for "
                                        "/rend, /rend_ref_refl, and rend_ref_tran")

parser.add_argument("-y", f"--{key_dry_run}", dest=key_dry_run, action="store_true",
                    required=False, help="If given, nothing is saved but the prints will come out.")

parser.add_argument("-c", f"--{key_clear}", dest=key_clear, action="store_true",
                    required=False, help="Clear leaf render folder before starting new renders.")

parser.add_argument("-cr", f"--{key_clear_refs}", dest=key_clear_refs, action="store_true",
                    required=False, help="Clear all reference folders before starting new renders.")

parser.add_argument("-r", f"--{key_render_references}", dest=key_render_references, action="store_true",
                    required=False, help="Render new references if illumination changes.")

parser.add_argument("-wl", f"--{key_wavelength}", dest=key_wavelength, action="store", type=float,
                    required=True,
                    help="Number of wavelengths. Wavelength ('-wl') must be provided if '-s' switch is used.")

parser.add_argument("-da", f"--{key_abs_dens}", dest=key_abs_dens, action="store", type=float,
                    choices=Range(0.0, 1000.0),
                    required=True, help="Shader volume absorption node's density input.")

parser.add_argument("-ds", f"--{key_scat_dens}", dest=key_scat_dens, action="store", type=float,
                    choices=Range(0.0, 1000.0),
                    required=True, help="Shader volume scattering node's density input.")

parser.add_argument("-ai", f"--{key_scat_ai}", dest=key_scat_ai, action="store", type=float, choices=Range(-1.0, 1.0),
                    required=True, help="Shader volume scattering node's anisotropy input.")

parser.add_argument("-mf", f"--{key_mix_fac}", dest=key_mix_fac, action="store", type=float, choices=Range(0.0, 1.0),
                    required=True,
                    help="Mixing factor of absorption and scattering (0 for full absorption, 1 for scatter).")

args = parser.parse_args(argv)
# print(vars(args))
# print(f"{s.key_wl}={vars(args)[s.key_wl]}")

dry_run = vars(args)[key_dry_run]
# single_wavelength = vars(args)[key_single_wavelength]
clear = vars(args)[key_clear]
clear_refs = vars(args)[key_clear_refs]
render_references = vars(args)[key_render_references]
wavelength = vars(args)[key_wavelength]
abs_dens = vars(args)[key_abs_dens]
scat_dens = vars(args)[key_scat_dens]
scat_ai = vars(args)[key_scat_ai]
mix_fac = vars(args)[key_mix_fac]
base_path = vars(args)[key_base_path]

render_path_leaf = base_path + '/rend'
# logging.warning(f"Rendering to '{render_path_leaf}'.")
render_path_refl_ref = base_path + '/rend_refl_ref'
render_path_tran_ref = base_path + '/rend_tran_ref'

# if single_wavelength and not wavelength:
#     raise Exception(f"Wavelength must be provided if '-s' switch is used.")
# if single_wavelength and abs_dens is None:
#     raise Exception(f"Absorption density must be provided if '-s' switch is used.")
# if single_wavelength and scat_dens is None:
#     raise Exception(f"Scattering density must be provided if '-s' switch is used.")
# if single_wavelength and scat_ai is None:
#     raise Exception(f"Scattering anisotropy must be provided if '-s' switch is used.")
# if single_wavelength and mix_fac is None:
#     raise Exception(f"Mixing factor must be provided if '-s' switch is used.")


def set_active_camera(cam_name):
    for cam in cameraList:
        if cam.name == cam_name:
            C.scene.camera = cam


def get_active_camera():
    return C.scene.camera


def toggle_cam():
    if get_active_camera().name == cam_name_refl:
        set_active_camera(cam_name_tran)
    else:
        set_active_camera(cam_name_refl)

    # print(f'Active camera is {C.scene.camera.name}')


def render_leaf(imaging_type, wl, abs_dens, scat_dens, scat_ai, mix_fac, dry_run=True):
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


def render_reference(imaging_type, wl, dry_run=dry_run):
    if imaging_type == imaging_type_refl:
        render_target(imaging_type, wl, target_refl_ref, render_path_refl_ref, dry_run=dry_run)
    elif imaging_type == imaging_type_tran:
        render_target(imaging_type, wl, target_tran_ref, render_path_tran_ref, dry_run=dry_run)


def render_target(imaging_type, wl, target_name, render_path, dry_run=True):
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

    # logging.warning(f"target_name: {target_name}")
    target_obj = D.objects[target_name]
    # logging.warning("PIIIIIdsafIPd")
    target_obj.hide_render = False
    target_obj.hide_viewport = False

    if not dry_run:
        # logging.warning(f'Saving render to "{file_path}"')
        O.render.render(write_still=True)
    else:
        logging.warning(f'Faking to save render to "{file_path}"')

# Calling script must handle the series
# def render_image_series(wl_list, absorption_list, scatter_list, mix_factor_list, scattering_anisotropy_list,
#                         do_reference=False, dry_run=True):
#     print('\n#################################\nRendering images\n############################')
#
#     if not dry_run:
#         clear_folders()
#
#     for j, wl in enumerate(wl_list):
#
#         render_leaf(imaging_type_refl, wl, absorption_list[j], scatter_list[j], scattering_anisotropy_list[j],
#                     mix_factor_list[j], dry_run=dry_run)
#         render_leaf(imaging_type_tran, wl, absorption_list[j], scatter_list[j], scattering_anisotropy_list[j],
#                     mix_factor_list[j], dry_run=dry_run)
#
#         if do_reference:
#             render_reference(imaging_type_refl, wl, dry_run=dry_run)
#             render_reference(imaging_type_tran, wl, dry_run=dry_run)


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

    Creates missing folder, so it should be safe to call.
    """

    make_folders()
    print("Clearing old data")
    list(map(os.unlink, (os.path.join(render_path_leaf, f) for f in os.listdir(render_path_leaf))))
    if clear_reference:
        print("Clearing reflectance and transmittance reference data")
        list(map(os.unlink, (os.path.join(render_path_refl_ref, f) for f in os.listdir(render_path_refl_ref))))
        list(map(os.unlink, (os.path.join(render_path_tran_ref, f) for f in os.listdir(render_path_tran_ref))))


########### "MAIN" ##################

make_folders()

if clear and not dry_run:
    clear_folders(clear_reference=clear_refs)

render_leaf(imaging_type_refl, wavelength, abs_dens, scat_dens, scat_ai, mix_fac, dry_run=dry_run)
render_leaf(imaging_type_tran, wavelength, abs_dens, scat_dens, scat_ai, mix_fac, dry_run=dry_run)

if render_references:
    render_reference(imaging_type_refl, wavelength, dry_run=dry_run)
    render_reference(imaging_type_tran, wavelength, dry_run=dry_run)

    # Do not use this! It should be updated later to make the final renders
    # after parameter optimization.
    # n = 5
    # wl_list = np.linspace(400, 1500, num=n)
    # absorption_list = np.linspace(100, 100, num=n)
    # scatter_list = np.linspace(77, 77, num=n)
    # mix_factor_list = np.linspace(1.0, 0.0, num=n)
    # scattering_anisotropy_list = np.linspace(0.0, 0.5, num=n)
    #
    # render_image_series(wl_list, absorption_list, scatter_list, mix_factor_list, scattering_anisotropy_list,
    #                     do_reference=render_references, dry_run=dry_run)
