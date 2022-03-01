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
cameras = b_data.collections['Cameras'].all_objects
lights = b_data.collections['Lighting'].all_objects
trees = b_data.collections['Trees'].all_objects
forest = b_data.collections['Ground'].all_objects.get('Ground')


def print_collection_items(collection_name):
    print(f'{collection_name} collection contains:')
    for item in b_data.collections[collection_name].all_objects:
        print(f"\t{item.name}: location {item.location}")


def print_materials():
    print(f'Available materials:')
    for item in b_data.materials:
        print(f"\t{item.name}")


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
    success = False
    # for input in tree_geometry_node.inputs:
    #     if input.name == parameter_name:
    #         old_val = input.default_value
    #         input.default_value = value
    #         print(f"Parameter {input.name} value changed from {old_val} to {value}")
    #         success = True
    # if not success:
    #     logging.warning(f"Parameter '{parameter_name}' value change failed. Check parameter name and data type.")

    input = tree_geometry_node.inputs.get(parameter_name)
    if input == None:
        raise AttributeError(f"Parameter called '{parameter_name}' seems not to exist. Check the name.")
    old_val = input.default_value
    input.default_value = value
    print(f"Parameter {input.name} value changed from {old_val} to {value}")


def list_forest_parameters():
    gn = forest.modifiers['GeometryNodes'].node_group
    print(f"Accessing {gn.name}")
    for input in gn.inputs:
        print(f"\t{input.name} ({input.type})")


def set_forest_parameter(parameter_name, value):
    input = forest.modifiers['GeometryNodes'].node_group.inputs.get(parameter_name)
    if input == None:
        raise AttributeError(f"Parameter called '{parameter_name}' seems not to exist. Check the name.")
    old_val = input.default_value
    input.default_value = value
    print(f"Parameter {input.name} value changed from {old_val} to {value}")

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

    # print_collection_items('Cameras')
    # print_collection_items('Lighting')
    # print_collection_items('Trees')
    # print_collection_items('Ground')
    # print_materials()
    # list_tree_parameter_names()
    # set_tree_parameter(1, 'Tree lengt', 4.0)
    list_forest_parameters()
    set_forest_parameter('Side length', 34.0)
    # bpy.ops.wm.save_as_mainfile(filepath=base_path)

    # TODO set Cycles
    # TODO set rendering parameters (image size, sample count...)
    # TODO set GPU compute
    # TODO set camera position and rotation
    # TODO set sun angle
    # TODO set material parameters
    # TODO set tree parameters
    # TODO set color space and screen parameters
