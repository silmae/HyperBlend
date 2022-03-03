# bpy stands for Blender Python, which is included Blender's own Python environment.
# It exists inside Blender, so don't worry if your IDE flags it as not found.
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


def print_collection_items(collection_name):
    print(f'{collection_name} collection contains:')
    for item in b_data.collections[collection_name].all_objects:
        print(f"\t{item.name}: location {item.location}")


def print_materials():
    print(f'Available materials:')
    for item in b_data.materials:
        print(f"\t{item.name}")


def set_sun_angle(angle_deg):
    sun = lights.get('Sun')
    sun.rotation_euler = (math.radians(angle_deg), 0, math.radians(90))


def list_tree_parameter_names():
    gn = trees[0].modifiers['GeometryNodes'].node_group
    # print(f"Accessing {gn.name}")
    tree_geometry_node = gn.nodes.get('Group')
    for input in tree_geometry_node.inputs:
        print(f"{input.name} ({input.type})")


def set_tree_parameter(tree_nr, parameter_name, value):
    """

    Available parameters:
        Tree length (VALUE)
        Main branch length (VALUE)
        Secondary branch length (VALUE)
        Twig lenght (VALUE)
        Branches per m (INT)
        Secondary branches per m (INT)
        Twigs per m (INT)
        Leaves per m (INT)
        Branch inclination deg (INT)
        Trunk pruning (VALUE)
        Clear start trunk percent (INT)
        Clear start Main brach percent (INT)
        Clear start Secondary branch percent (INT)
        Trunk thickness (VALUE)
        Main branch thickness (VALUE)
        Secondary branch thickness (VALUE)
        Branch base thickness (VALUE)
        Main branch curvature (VALUE)
        Secondary branch opening (VALUE)
        Secondary branch curvature (VALUE)
        Distortion scale (VALUE)
        Leaf thickness cm (VALUE)
        Leaf  side length cm (VALUE)
        Trunk material (MATERIAL)
        Branch material (MATERIAL)
        Leaf material (MATERIAL)

    :param tree_nr:
    :param parameter_name:
    :param value:
    :return:
    """

    gn = trees[tree_nr-1].modifiers['GeometryNodes'].node_group
    tree_geometry_node = gn.nodes.get('Group')
    set_input(tree_geometry_node, parameter_name, value)


def set_input(node, input_name, value):
    input = node.inputs.get(input_name)
    if input == None:
        raise AttributeError(f"Parameter called '{input_name}' seems not to exist. Check the name.")
    old_val = input.default_value
    input.default_value = value
    print(f"{node.name}: parameter {input.name} value changed from {old_val} to {value}.")


def list_forest_parameters():
    for input in forest_geometry_node.inputs:
        print(f"{input.name} ({input.type})")


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


def set_rendering_parameters():
    resolution_x = 128
    resolution_y = 128
    for scene in b_data.scenes:
        scene.render.resolution_x = resolution_x # OK
        scene.render.resolution_y = resolution_y # OK
        scene.render.resolution_percentage = 100 # OK

        scene.render.image_settings.file_format = 'TIFF' # OK
        scene.render.image_settings.tiff_codec = 'NONE'
        scene.render.image_settings.color_mode = 'BW' # 'RGB', 'RGBA'
        scene.render.image_settings.color_depth = '16'

        scene.render.fps = 5 # OK
        scene.frame_start = 1 # OK
        scene.frame_end = 3 # OK


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


def framing_material(material_name):
    mat = b_data.materials.get(material_name)
    print(f"Touching material '{mat.name}'.")
    data_path = 'nodes["Diffuse BSDF"].inputs["Color"].default_value'
    mat.node_tree.nodes['Diffuse BSDF'].inputs['Color'].default_value = (1, 0, 0, 1) # RGBA
    mat.node_tree.keyframe_insert(data_path, frame=1)
    mat.node_tree.nodes['Diffuse BSDF'].inputs['Color'].default_value = (0, 1, 0, 1)  # RGBA
    mat.node_tree.keyframe_insert(data_path, frame=2)
    mat.node_tree.nodes['Diffuse BSDF'].inputs['Color'].default_value = (0, 0, 1, 1)  # RGBA
    mat.node_tree.keyframe_insert(data_path, frame=3)


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

    # print_collection_items('Cameras')
    # print_collection_items('Lighting')
    # print_collection_items('Trees')
    # print_collection_items('Ground')
    # print_materials()
    # list_tree_parameter_names()
    # set_tree_parameter(1, 'Tree length', 11.0)
    # list_forest_parameters()
    # set_forest_parameter('Ground resolution', 5)
    # set_rendering_parameters()
    # set_sun_angle(60)
    framing_material()

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
