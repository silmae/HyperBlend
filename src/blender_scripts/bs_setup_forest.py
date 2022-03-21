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
b_scene = b_data.scenes['Forest']

cameras = b_data.collections['Cameras'].all_objects
lights = b_data.collections['Lighting'].all_objects
trees = b_data.collections['Trees'].all_objects
markers = b_data.collections['Marker objects'].all_objects
forest = b_data.collections['Ground'].all_objects.get('Ground')

forest_geometry_node = forest.modifiers['GeometryNodes'].node_group.nodes.get('Group.004')


def set_sun_angle(angle_deg):
    sun = lights.get(FC.key_obj_sun)
    sun.rotation_euler = (math.radians(angle_deg), 0, math.radians(90))


def framing_material(leaf_index, band, absorption_density, scattering_density, scattering_anisotropy, mix_factor):

    logging.warning(f"Setting framed material for band {band} of leaf {leaf_index}: [{absorption_density}, {scattering_density}, {scattering_anisotropy}, {mix_factor}]")

    # mat = b_data.materials.get(material_name)
    # print(f"Touching material '{mat.name}'.")
    # data_path = 'nodes["Diffuse BSDF"].inputs["Color"].default_value'
    # bpy.data.materials["Leaf material 2"].node_tree.nodes["Group"].inputs[0].default_value

    # mat.node_tree.nodes["Group"].inputs['Absorption density'].default_value

    def set_stuff(material, parameter, value, frame):
        dp = f'nodes["Group"].inputs["{parameter}"].default_value'
        material.node_tree.nodes["Group"].inputs[f"{parameter}"].default_value = value
        material.node_tree.keyframe_insert(dp, frame=frame)

    material = bpy.data.materials[f"Leaf material {leaf_index}"]
    set_stuff(material, 'Absorption density', absorption_density, band)
    set_stuff(material, 'Scattering density', scattering_density, band)
    set_stuff(material, 'Scattering anisotropy', scattering_anisotropy, band)
    set_stuff(material, 'Mix factor', mix_factor, band)
    set_stuff(material, 'Density scale', 300, band) # TODO what should we do with this

    # mat.node_tree.nodes['Diffuse BSDF'].inputs['Color'].default_value = (1, 0, 0, 1) # RGBA
    # mat.node_tree.nodes['Diffuse BSDF'].inputs['Color'].default_value = (0, 1, 0, 1)  # RGBA
    # mat.node_tree.keyframe_insert(data_path, frame=2)
    # mat.node_tree.nodes['Diffuse BSDF'].inputs['Color'].default_value = (0, 0, 1, 1)  # RGBA
    # mat.node_tree.keyframe_insert(data_path, frame=3)


def read_leaf_material_csv(leaf_index=1):
    p = os.path.abspath(base_path + f"/leaf_material_{leaf_index}.csv")
    band_count = 0
    with open(p) as file:
        reader = csv.reader(file, delimiter=' ')
        for row in reader:
            try:
                band = int(row[0])
                wavelength = float(row[1])
                absorption_density = float(row[2])
                scattering_density = float(row[3])
                scattering_anisotropy = float(row[4])
                mix_factor = float(row[5])
                framing_material(leaf_index, band, absorption_density, scattering_density, scattering_anisotropy, mix_factor)
                band_count += 1
            except ValueError:
                print(f"Material headers: {row}")

    for scene in b_data.scenes:
        scene.render.fps = 5 # does not really matter
        scene.frame_start = 1
        scene.frame_end = band_count


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

    logging.error(f"Hello, I am forest setup script in '{base_path}'")

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

    read_leaf_material_csv(1)
    read_leaf_material_csv(2)
    read_leaf_material_csv(3)

    bpy.ops.wm.save_as_mainfile(filepath=file_path)

    # TODO set Cycles
    # TODO set rendering parameters (image size, sample count...)
    # TODO set GPU compute
    # TODO set camera position and rotation
    # TODO set sun angle
    # TODO set material parameters
    # TODO set tree parameters
    # TODO set color space and screen parameters
    # TODO how to disable using User preferences?
