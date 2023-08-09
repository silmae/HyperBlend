import bpy
import os
import sys  # to get command line args
import argparse  # to parse options for us and print a nice help message
import logging
import random
import toml
import math

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
from src.data import path_handling as PH
import importlib

importlib.reload(FC)
importlib.reload(PH)

b_context = bpy.context
b_data = bpy.data
b_ops = bpy.ops
b_scene = b_data.scenes[FC.key_scene_name]

cameras = b_data.collections[FC.key_collection_cameras].all_objects
lights = b_data.collections[FC.key_collection_lights].all_objects
trees = b_data.collections[FC.key_collection_trees].all_objects
ground = b_data.collections[FC.key_collection_ground].all_objects

forest = b_data.collections[FC.key_collection_ground].all_objects.get(FC.key_obj_ground)
forest_geometry_node = forest.modifiers['GeometryNodes'].node_group.nodes.get('Group.004')


def write_forest_control(forest_id: str, control_dict: dict, global_master: bool = False):
    """Writes forest control file.

    :param forest_id:
        Forest id for which the control file is written to.
    :param control_dict:
        Dictionary to be written.
    :param global_master:
        If True, global master file is written to project root. Needs to be done if there are
        changes made to the forest template file. This will be kept safe in the Git repository.
        Default is False.
    """

    if global_master:
        write_dict_as_toml(dictionary=control_dict, directory=PH.path_directory_project_root(), filename='forest_control')
    else:
        write_dict_as_toml(dictionary=control_dict, directory=PH.path_directory_forest_scene(forest_id=forest_id), filename='forest_control')



def read_forest_control(forest_id: str) -> dict:
    return read_toml_as_dict(directory=PH.path_directory_forest_scene(forest_id=forest_id), filename='forest_control')


def write_dict_as_toml(dictionary: dict, directory: str, filename: str):
    """General purpose dictionary saving method.

    :param dictionary:
        Dictionary to be written as toml.
    :param directory:
        Path to the directory where the toml should be written.
    :param filename:
        Name of the file to be written. Postfix '.toml' will be added if necessary.
    """

    if not os.path.exists(os.path.abspath(directory)):
        raise RuntimeError(f"Cannot write given dictionary to path '{os.path.abspath(directory)}' "
                           f"because it does not exist.")

    if not filename.endswith('.toml'):
        filename = filename + '.toml'

    p = PH.join(directory, filename)
    with open(p, 'w+') as file:
        toml.dump(dictionary, file, encoder=toml.encoder.TomlNumpyEncoder())


def read_toml_as_dict(directory: str, filename: str):
    """General purpose toml reading method.

    :param directory:
        Path to the directory where the toml file is.
    :param filename:
        Name of the file to be read. Postfix '.toml' will be added if necessary.
    :return dictionary:
        Returns read toml file as a dictionary.
    """

    if not filename.endswith('.toml'):
        filename = filename + '.toml'

    p = PH.join(directory, filename)

    if not os.path.exists(os.path.abspath(p)):
        raise RuntimeError(f"Cannot read from file '{os.path.abspath(p)}' "
                           f"because it does not exist.")

    with open(p, 'r') as file:
        result = toml.load(file)
    return result


def set_input(node, input_name, value):
    input = node.inputs.get(input_name)
    if input == None:
        raise AttributeError(f"Parameter called '{input_name}' seems not to exist. Check the name.")
    old_val = input.default_value
    input.default_value = value
    print(f"{node.name}: parameter {input.name} value changed from {old_val} to {value}.")


def dictify_input_socket(gn, socket, is_master):

    space = "    "
    socket_id = socket.identifier
    socket_id_numeric = int(socket_id.split('_')[1])
    socket_value = gn[socket.identifier]
    socket_type = socket.type

    socket_dict = {f"{space}Value": socket_value,}

    # Only add standard deviation to master file
    if is_master:
        if socket_type == "VALUE":
            socket_dict[f"{space}Standard deviation"] = socket_value / 10
        if socket_type == "INT":
            socket_dict[f"{space}Standard deviation"] = int(socket_value / 10)

    socket_dict[f"{space}Type"] = socket_type
    socket_dict[f"{space}ID"] = socket_id_numeric

    return socket_dict


def get_tree_as_dict(tree_object, is_master=False):

    tree_dict = {"Name": tree_object.name}

    # print(f"Tree in tree list: {tree_object.name}")
    tree_gn = tree_object.modifiers["GeometryNodes"]

    for tree_input_socket in tree_gn.node_group.inputs:

        tree_socket_id = tree_input_socket.identifier
        tree_socket_id_numeric = int(tree_socket_id.split('_')[1])
        socket_type = tree_input_socket.type

        # Skip some parameters that do not need to be randomised.
        if socket_type == "MATERIAL": # Any materials
            continue
        elif tree_socket_id_numeric == 34: # Splines only
            continue
        elif tree_socket_id_numeric == 30: # Leaf object
            continue
        elif tree_socket_id_numeric == 29: # Hide leafs
            continue
        elif tree_socket_id_numeric == 15: # Top spawn
            continue
        else:
            tree_socket_name = tree_input_socket.name
            tree_dict[tree_socket_name] = dictify_input_socket(tree_gn, tree_input_socket, is_master=is_master)

    return tree_dict


def get_scene_parameters(as_master=False) -> dict:
    """
    TODO clean this up!!

     get forest parameters
          - get active trees and understory parameters
              - get active leaves
     get world parameters
    """

    logging.error(f"Reading scene definition from Blender file.")


    scene_dict = {"Note": "This file controls the setup of the Blender scene file. ",
                  "is_master_control": as_master,
                  }
    #TODO drones and cameras
    #       - altitude, orientation, resolution, FOV, § sample count

    sun = lights[FC.key_obj_sun]
    sun_dict = {
        "Note": "When sun azimuth angle is 0 degrees, the sun points to positive y-axis direction in Blender "
                "that is thought as north in HyperBlend. 90 degrees would be pointing west, 180 to south "
                "and 270 to east, respectively. Zenith angle is the Sun's angle from zenith.",
        FC.key_ctrl_sun_angle_zenith_deg: math.degrees(sun.rotation_euler[0]),
        FC.key_ctrl_sun_angle_azimuth_deg: math.degrees(sun.rotation_euler[2]),
        FC.key_ctrl_sun_base_power_hsi: FC.max_sun_power_spectral,
        FC.key_ctrl_sun_base_power_rgb: FC.max_sun_power_rgb,
    }
    scene_dict['Sun'] = sun_dict

    drone_dict = {
        "Note": "Unit of drone location and altitude is meter.",
        FC.key_ctrl_drone_location_x: bpy.data.objects[FC.key_drone].location[0],
        FC.key_ctrl_drone_location_y: bpy.data.objects[FC.key_drone].location[1],
        FC.key_ctrl_drone_altitude: bpy.data.objects[FC.key_drone].location[2],
    }
    scene_dict['Drone'] = drone_dict

    camera_dict = {
        "Note": "Camera angles are stored in degrees in this file. They must be "
                "converted to radians before passing to Blender file.",
        FC.key_ctrl_drone_hsi_fow: math.degrees(cameras.get(FC.key_cam_drone_hsi).data.angle),
        FC.key_ctrl_drone_rgb_fow: math.degrees(cameras.get(FC.key_cam_drone_rgb).data.angle),
    }
    scene_dict['Cameras'] = camera_dict

    rendering_dict = {
        "Note": "Sample count controls how many samples (light rays) are cast through each pixel."
                "More samples result in smoother image but require more time to render. Try values "
                "between 16 and 512, for example. The RGB sampling is for preview images so it can "
                "be higher as not many images are rendered with that sampling.",
        FC.key_ctrl_sample_count_rbg: b_scene.cycles.samples,
        FC.key_ctrl_sample_count_hsi: b_scene.cycles.samples,
    }
    scene_dict['Rendering'] = rendering_dict

    # Blender only has one global resolution setting that is not bound to different cameras.
    # So we take the one there is and set it as resolution for all cameras.
    resolution_x = b_scene.render.resolution_x
    resolution_y = b_scene.render.resolution_y
    image_dict = {
        FC.key_ctrl_hsi_resolution_x: resolution_x,
        FC.key_ctrl_hsi_resolution_y: resolution_y,
        FC.key_ctrl_rgb_resolution_x: resolution_x,
        FC.key_ctrl_rgb_resolution_y: resolution_y,
        FC.key_ctrl_walker_resolution_x: resolution_x,
        FC.key_ctrl_walker_resolution_y: resolution_y,
        FC.key_ctrl_sleeper_resolution_x: resolution_x,
        FC.key_ctrl_sleeper_resolution_y: resolution_y,
        FC.key_ctrl_tree_preview_resolution_x: resolution_x,
        FC.key_ctrl_tree_preview_resolution_y: resolution_y,
    }
    scene_dict['Images'] = image_dict

    ground_dict = {}
    ground_gn = bpy.data.objects["Ground"].modifiers["GeometryNodes"]

    for input_socket in ground_gn.node_group.inputs:

        socket_id = input_socket.identifier
        socket_id_numeric = int(socket_id.split('_')[1])
        socket_name = input_socket.name
        socket_value = ground_gn[input_socket.identifier]

        if socket_id_numeric == 10: # Simplified trees
            continue
        elif socket_id_numeric == 34: # Simplified understory
            continue
        elif socket_id_numeric == 18: # Reference object
            continue
        elif socket_id_numeric == 35: # Reference controller
            continue
        elif socket_id_numeric == 19: # Reference height
            continue
        elif socket_id_numeric in [25, 26, 27]: # Trees
            ground_dict[socket_name] = get_tree_as_dict(socket_value, is_master=as_master)
        elif socket_id_numeric in [30, 31]: # Understory
            ground_dict[socket_name] = get_tree_as_dict(socket_value, is_master=as_master)
        else:
            ground_dict[socket_name] = dictify_input_socket(ground_gn, input_socket, is_master=as_master)

    scene_dict['Forest'] = ground_dict

    return scene_dict

def set_forest_parameter(parameter_name, value):
    """
    Input Input_7 is named Seed
    Input Input_8 is named Size X [m]
    Input Input_9 is named Size Y [m]
    Input Input_10 is named Simplified trees
    Input Input_34 is named Simplified understory
    Input Input_12 is named Minimum tree separation [m]
    Input Input_13 is named Spawn probability [%]
    Input Input_14 is named Tree 1 probability [%]
    Input Input_15 is named Tree 2 probability [%]
    Input Input_16 is named Tree 3 probability [%]
    Input Input_25 is named Tree 1
    Input Input_26 is named Tree 2
    Input Input_27 is named Tree 3
    Input Input_18 is named Reference object
    Input Input_35 is named Reference controller
    Input Input_19 is named Reference height [m]
    Input Input_20 is named Reference safe distance [m]
    Input Input_22 is named Height map strength
    Input Input_24 is named Height point separation [m]
    Input Input_28 is named Max tree tilt [Deg]
    Input Input_29 is named Max tree scale [%]
    Input Input_30 is named Understory object 1
    Input Input_32 is named Understory 1 min separation [m]
    Input Input_31 is named Understory object 2
    Input Input_33 is named Understory 2 min separation [m]
    Input Input_36 is named Ground material

    :param parameter_name:
    :param value:
    :return:
    """

    mod = bpy.data.objects["Ground"].modifiers["GeometryNodes"]

    for input_socket in mod.node_group.inputs:

        # Uncomment to print input socets names.
        # print(f"Input {input_socket.identifier} is named {input_socket.name}")
        # print(f"Input socket name: '{input_socket.name}', parameter_name: '{parameter_name}'")

        if input_socket.name == parameter_name:
            old_val = mod[input_socket.identifier]
            mod[input_socket.identifier] = value
            logging.error(f"Forest parameter {input_socket.name} value changed from {old_val} to {value}.")


def get_visibility_mapping_material_names():
    """Return the names of materials that should be made into an abundance map."""

    ground_gn = bpy.data.objects["Ground"].modifiers["GeometryNodes"]
    tree_like_objects = ["Tree 1", "Tree 2", "Tree 3", "Understory object 1", "Understory object 2"]
    material_socket_names = ["Trunk material", "Leaf material"]
    res = ["Ground material"]

    for ground_socket in ground_gn.node_group.inputs:

        obj = ground_gn[ground_socket.identifier]
            # print(f"Object: {obj.name}")

        if ground_socket.name == "Reference object":
            for material in obj.data.materials:
                if material.name not in res:
                    res.append(material.name)
                # print(f"Reference material: {material.name}")

        if ground_socket.name in tree_like_objects:
            tree_gn = obj.modifiers["GeometryNodes"]
            for tree_socket in tree_gn.node_group.inputs:
                if tree_socket.name in material_socket_names:
                    material = tree_gn[tree_socket.identifier]
                    if material.name not in res:
                        res.append(material.name)
                    # print(f"Tree like material: {material.name}")

    # print(f"Unique active materials: {res}")
    return res


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
    mod = bpy.data.objects["Ground"].modifiers["GeometryNodes"]
    for input_socket in mod.node_group.inputs:
        print(f"Input {input_socket.identifier} is named {input_socket.name}")


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
