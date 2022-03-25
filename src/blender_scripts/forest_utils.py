import bpy
import os
import sys  # to get command line args
import argparse  # to parse options for us and print a nice help message
import logging
import random

blend_dir = os.path.dirname(os.path.abspath(bpy.data.filepath))

if 'scenes' in blend_dir:
    # We are in a copied blend file in HyperBlend/scenes/scene_12345
    script_dir = os.path.abspath(blend_dir + '../../../src/blender_scripts')
else:
    # We are in the template forest blend file
    script_dir = os.path.abspath(blend_dir + '/src/blender_scripts')

# After this is set, any script in /blender_scripts can be imported
if script_dir not in sys.path:
    sys.path.append(script_dir)

import forest_constants as FC
import importlib
importlib.reload(FC)

test_val = "I am test value"

b_context = bpy.context
b_data = bpy.data
b_ops = bpy.ops
b_scene = b_data.scenes[FC.key_scene_name]

cameras = b_data.collections[FC.key_collection_cameras].all_objects
lights = b_data.collections[FC.key_collection_lights].all_objects
trees = b_data.collections[FC.key_collection_trees].all_objects
markers = b_data.collections[FC.key_collection_markers].all_objects
ground = b_data.collections[FC.key_collection_ground].all_objects

forest = b_data.collections[FC.key_collection_ground].all_objects.get(FC.key_obj_ground)
forest_geometry_node = forest.modifiers['GeometryNodes'].node_group.nodes.get('Group.004')


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
    Show white reference (BOOLEAN)

    :param parameter_name:
    :param value:
    :return:
    """

    set_input(forest_geometry_node, parameter_name, value)


def random_ground(rand_state):
    """Set ground parameters to random values to create unique scenes.

    Parameter ranges are currently hard-coded.

    TODO allow changing parameter ranges when the script is called.
    """

    random.setstate(rand_state)

    hill_scale = random.uniform(5,15)
    set_forest_parameter(parameter_name='Hill height', value=hill_scale*0.2)
    set_forest_parameter(parameter_name='Hill scale', value=hill_scale)
    set_forest_parameter(parameter_name='Seed', value=random.randint(0,1000000))

    return random.getstate()


def random_tree(tree_number, rand_state):
    """Set tree parameters to random values to create unique scenes.

    Parameter ranges are currently hard-coded.

    TODO allow changing parameter ranges when the script is called.
    """

    random.setstate(rand_state)

    tree_length_rand = random.uniform(5, 15)
    set_tree_parameter(tree_nr=tree_number, parameter_name='Tree length', value=tree_length_rand)
    set_tree_parameter(tree_nr=tree_number, parameter_name='Main branch length', value=tree_length_rand/5)
    set_tree_parameter(tree_nr=tree_number, parameter_name='Secondary branch length', value=tree_length_rand/25)
    set_tree_parameter(tree_nr=tree_number, parameter_name='Twig length', value=random.uniform(0.2, 0.4))

    set_tree_parameter(tree_nr=tree_number, parameter_name='Branches per m', value=random.randint(10, 20))
    set_tree_parameter(tree_nr=tree_number, parameter_name='Secondary branches per m', value=random.randint(5, 10))
    set_tree_parameter(tree_nr=tree_number, parameter_name='Twigs per m', value=random.randint(15, 30))
    set_tree_parameter(tree_nr=tree_number, parameter_name='Leaves per m', value=random.randint(10, 20))

    set_tree_parameter(tree_nr=tree_number, parameter_name='Branch inclination deg', value=random.randint(0, 180))

    set_tree_parameter(tree_nr=tree_number, parameter_name='Clear start trunk percent', value=random.randint(0, 40))
    set_tree_parameter(tree_nr=tree_number, parameter_name='Clear start Main branch percent', value=random.randint(0, 40))
    set_tree_parameter(tree_nr=tree_number, parameter_name='Clear start Secondary branch percent', value=random.randint(0, 40))

    set_tree_parameter(tree_nr=tree_number, parameter_name='Trunk pruning', value=random.uniform(0, 4))

    # TODO thicknesses should have meaningful units, e.g., cm

    trunk_thickness_rand = random.uniform(tree_length_rand-2, tree_length_rand+2)
    set_tree_parameter(tree_nr=tree_number, parameter_name='Trunk thickness', value=trunk_thickness_rand)
    # set_tree_parameter(tree_nr=1, parameter_name='Main branch thickness', value=random.uniform(trunk_thickness_rand, tree_length_rand+2))
    # set_tree_parameter(tree_nr=1, parameter_name='Secondary branch thickness', value=random.uniform(tree_length_rand-2, tree_length_rand+2))

    set_tree_parameter(tree_nr=tree_number, parameter_name='Leaf  side length cm', value=random.uniform(2, 8))

    set_tree_parameter(tree_nr=tree_number, parameter_name='Trunk material', value=b_data.materials[f'Trunk material {tree_number}'])
    set_tree_parameter(tree_nr=tree_number, parameter_name='Branch material', value=b_data.materials[f'Trunk material {tree_number}'])
    set_tree_parameter(tree_nr=tree_number, parameter_name='Leaf material', value=b_data.materials[f'Leaf material {tree_number}'])

    return random.getstate()


def set_tree_parameter(tree_nr, parameter_name, value):
    """

    Available parameters:
        Tree length (VALUE)
        Main branch length (VALUE)
        Secondary branch length (VALUE)
        Twig length (VALUE)
        Branches per m (INT)
        Secondary branches per m (INT)
        Twigs per m (INT)
        Leaves per m (INT)
        Branch inclination deg (INT)
        Trunk pruning (VALUE)
        Clear start trunk percent (INT)
        Clear start Main branch percent (INT)
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
        Currently 1, 2 or 3.
    :param parameter_name:
    :param value:
    :return:
    """

    gn = trees[tree_nr-1].modifiers['GeometryNodes'].node_group
    tree_geometry_node = gn.nodes.get('Group')
    set_input(tree_geometry_node, parameter_name, value)


def list_forest_parameters():
    for input in forest_geometry_node.inputs:
        print(f"{input.name} ({input.type})")


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
