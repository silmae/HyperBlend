import bpy
import os
import sys
import logging
import csv

import numpy as np
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
import forest_control as control

import importlib

importlib.reload(FC)
importlib.reload(PH)
importlib.reload(control)

context = bpy.context
data = bpy.data
scene = data.scenes["Scene"]

# cameras = data.collections[FC.key_collection_cameras].all_objects
# lights = data.collections[FC.key_collection_lights].all_objects
# trees = data.collections[FC.key_collection_trees].all_objects


def get_scene_parameters(as_master=False) -> dict:
    """Fetches scene parameters from a Blender file and returns them
        as a dict that can be saved as a forest control file.

    :param as_master:
        If True, returned dict will be treated as a master control meaning that there
        will be default standard deviation added to some parameters that are to be
        randomized.
    :return:
        Forest control dict.
    """

    logging.error(f"Reading scene definition from Blender file.")

    scene_dict = {"Note": "This file controls the setup of the Blender scene file. ",
                  FC.key_ctrl_is_master_control: as_master,
                  }

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
        FC.key_ctrl_sample_count_rbg: scene.cycles.samples,
        FC.key_ctrl_sample_count_hsi: scene.cycles.samples,
    }
    scene_dict['Rendering'] = rendering_dict

    # Blender only has one global resolution setting that is not bound to different cameras.
    # So we take the one there is and set it as resolution for all cameras.
    resolution_x = scene.render.resolution_x
    resolution_y = scene.render.resolution_y
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
            ground_dict[socket_name] = _get_tree_as_dict(socket_value, is_master=as_master)
        elif socket_id_numeric in [30, 31]: # Understory
            ground_dict[socket_name] = _get_tree_as_dict(socket_value, is_master=as_master)
        else:
            ground_dict[socket_name] = _dictify_input_socket(ground_gn, input_socket, is_master=as_master)

    scene_dict['Forest'] = ground_dict

    return scene_dict


def _dictify_input_socket(gn, socket, is_master) -> dict:
    """Turns input socket data of a HyperBlend geometry nodes into a dict.

    Resulting dict will have following structure:

    {
    'Value': <int, float, string,...>,
    ('Standard deviation': <float>,)
    'Type': <either 'INT', 'VALUE' or 'MATERIAL'>,
    'ID': <int>,
    }

    Standard deviation is added only if is_master = True.

    Type refers to Blender's internal datatypes defined by uppercase strings. Type VALUE means float.

    :param gn:
        Geometry nodes node group where the input can be found.
    :param socket:
        Input socket object of the node group.
    :param is_master:
        If True, default standard deviation is added for later use.
    :return:
        Input socket as a dict.
    """

    socket_id = socket.identifier
    socket_id_numeric = int(socket_id.split('_')[1])
    socket_value = gn[socket.identifier]
    socket_type = socket.type

    socket_dict = {f"Value": socket_value, }

    # Only add standard deviation numerical parameters in master file, but ignore seeds.
    if is_master and 'Seed' not in socket.name:
        if socket_type == "VALUE":
            socket_dict[f"Standard deviation"] = socket_value * FC.ctrl_default_std_of_value
        if socket_type == "INT":
            socket_dict[f"Standard deviation"] = int(socket_value * FC.ctrl_default_std_of_value)

    socket_dict[f"Type"] = socket_type
    socket_dict[f"ID"] = socket_id_numeric

    return socket_dict


def _get_tree_as_dict(tree_object, is_master=False) -> dict:
    """Parses tree parameters from a tree object into a dict.

    :param tree_object:
        Tree object to be used.
    :param is_master:
        If True, default standard deviations will be added to parameters
        that will be randomized.
    :return:
        Tree parameters as a dictionary.
    """

    tree_dict = {"Name": tree_object.name}

    # print(f"Tree in tree list: {tree_object.name}")
    tree_gn = tree_object.modifiers["GeometryNodes"]

    for tree_input_socket in tree_gn.node_group.inputs:

        tree_socket_id = tree_input_socket.identifier
        tree_socket_id_numeric = int(tree_socket_id.split('_')[1])
        socket_type = tree_input_socket.type

        # Skip some parameters that do not need to be randomised.
        if socket_type == "MATERIAL":  # Any materials
            continue
        elif tree_socket_id_numeric == 34:  # Splines only
            continue
        elif tree_socket_id_numeric == 30:  # Leaf object
            continue
        elif tree_socket_id_numeric == 29:  # Hide leafs
            continue
        elif tree_socket_id_numeric == 15:  # Top spawn
            continue
        else:
            tree_socket_name = tree_input_socket.name
            tree_dict[tree_socket_name] = _dictify_input_socket(tree_gn, tree_input_socket, is_master=is_master)

    return tree_dict


def apply_forest_control(forest_id):
    """Reads forest control file and applies it to the forest scene.

    Note that some values, such as sun power, must be reset when rendering because
    proper values depend on are we rendering an RGB image or a hyperspectral image.

    :param forest_id:
        Forest id to be set.
    """

    logging.error(f"Applying scene control.")

    control_dict = control.read_forest_control(forest_id=forest_id)

    for key, dict_item in control_dict.items():

        # First we set all top-level scene parameters
        if key == "Sun":
            sun_dict = control_dict["Sun"]
            # sun = lights[FC.key_obj_sun]
            # sun.rotation_euler[0] = math.radians(sun_dict[FC.key_ctrl_sun_angle_zenith_deg])
            # sun.rotation_euler[2] = math.radians(sun_dict[FC.key_ctrl_sun_angle_azimuth_deg])
            # Only set RGB sun at this point. Rendering calls will change this anyway.
            set_sun_power(power=sun_dict[FC.key_ctrl_sun_base_power_rgb], frame=1)
        # elif key == "Drone":
        #     drone_dict = control_dict["Drone"]
        #     bpy.data.objects[FC.key_drone].location[0] = drone_dict[FC.key_ctrl_drone_location_x]
        #     bpy.data.objects[FC.key_drone].location[1] = drone_dict[FC.key_ctrl_drone_location_y]
        #     bpy.data.objects[FC.key_drone].location[2] = drone_dict[FC.key_ctrl_drone_altitude]
        # elif key == "Cameras":
        #     cameras_dict = control_dict["Cameras"]
        #     cameras.get(FC.key_cam_drone_hsi).data.angle = math.radians(cameras_dict[FC.key_ctrl_drone_hsi_fow])
        #     cameras.get(FC.key_cam_drone_rgb).data.angle = math.radians(cameras_dict[FC.key_ctrl_drone_rgb_fow])
        elif key == "Rendering":
            rendeering_dict = control_dict["Rendering"]
            scene.cycles.samples = rendeering_dict[FC.key_ctrl_sample_count_hsi]
        elif key == "Images":
            # images_dict = control_dict["Images"]
            # scene.render.resolution_x = images_dict[FC.key_ctrl_hsi_resolution_x]
            # scene.render.resolution_y = images_dict[FC.key_ctrl_hsi_resolution_y]
            pass
        # elif key == "Forest":
        #
        #     """
        #     Loop through forest (ground object) parameters. Almost all of these are dicts (even single valued
        #     parameters because we store the name and data type also. Then there are other objects (trees) that
        #     must be looped through separately.
        #     """
        #
        #     forest_dict = control_dict[key]
        #     for forest_key, forest_dict_item in forest_dict.items():
        #
        #         # Normal forest parameters
        #         if isinstance(forest_dict_item, dict) and 'Type' in forest_dict_item:
        #
        #             forst_item_type = forest_dict_item['Type']
        #
        #             if forst_item_type == 'VALUE':
        #                 set_forest_parameter(value=float(forest_dict_item['Value']), parameter_id=forest_dict_item['ID'])
        #             elif forst_item_type == 'INT':
        #                 set_forest_parameter(value=int(forest_dict_item['Value']), parameter_id=forest_dict_item['ID'])
        #             elif forst_item_type == 'MATERIAL':
        #                 pass # we do nothing for materials
        #             else:
        #                 logging.warning(f"Unexpected parameter type '{forst_item_type}'. Parameter '{forest_key}' : {forest_dict_item}.")
        #
        #         # Tree sub-dictionaries
        #         # TODO get rid of hard-coded names at some point
        #         elif forest_key == 'Tree 1' or forest_key == 'Tree 2' or forest_key == 'Tree 3' \
        #                 or forest_key == 'Understory object 1' or forest_key == 'Understory object 2':
        #
        #             tree_dict = forest_dict[forest_key]
        #
        #             for tree_key, tree_dict_item in tree_dict.items():
        #
        #                 tree_name = tree_dict['Name']
        #
        #                 if isinstance(tree_dict_item, dict) and 'Type' in tree_dict_item:
        #
        #                     tree_item_type = tree_dict_item['Type']
        #
        #                     if tree_item_type == 'VALUE':
        #                         set_tree_parameter(tree_name=tree_name, parameter_name=tree_key, value=float(tree_dict_item['Value']))
        #                     elif tree_item_type == 'INT':
        #                         set_tree_parameter(tree_name=tree_name, parameter_name=tree_key, value=int(tree_dict_item['Value']))
        #
        #                 elif tree_key == 'Name':
        #                     pass # The name is stored already.
        #                 else:
        #                     logging.warning(f"Unhandled tree parameter '{tree_key}' : {tree_dict_item}")
        #         elif forest_key == 'Note':
        #             pass # just comments
        #         else:
        #             logging.warning(f"Unhandled forest parameter '{forest_key}' : {forest_dict_item}")
        elif key == 'Note' or key == FC.key_ctrl_is_master_control:
            # No need to handle this key because we can set up a scene based on either master or slave control.
            pass
        else:
            logging.warning(f"Unhandled control dictionary key '{key}' : {dict_item}.")


def set_forest_parameter(value, parameter_name: str = None, parameter_id: int = None):
    """Sets forest parameter (ground object) to given value.

    Parameter can be identified either by name (parameter_name) or ID (parameter_id).
    One of these must be given.

    Available parameters are:

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

    :param parameter_id:
        Parameter ID as an integer.
    :param parameter_name:
        Name of the parameter.
    :param value:
        Value to be set. No checks on correct data type are
        made, so the caller must take care of that.
    """

    if parameter_id is None and parameter_name is None:
        raise AttributeError(f"You must provide either parameter name or parameter id. Both were None.")

    mod = bpy.data.objects["Ground"].modifiers["GeometryNodes"]

    for input_socket in mod.node_group.inputs:

        # Uncomment to print input socets names.
        # print(f"Input {input_socket.identifier} is named {input_socket.name}")
        # print(f"Input socket name: '{input_socket.name}', parameter_name: '{parameter_name}'")

        if parameter_name is not None and input_socket.name == parameter_name:
            old_val = mod[input_socket.identifier]
            mod[input_socket.identifier] = value
            if old_val != value:
                logging.error(f"Forest parameter '{input_socket.name}' value changed from {old_val} to {value}.")

        elif parameter_id is not None and f"Input_{parameter_id}" == input_socket.identifier:
            old_val = mod[input_socket.identifier]
            mod[input_socket.identifier] = value
            if old_val != value:
                logging.error(f"Forest parameter '{input_socket.name}' value changed from {old_val} to {value}.")


def get_visibility_mapping_material_names():
    """Return the names of materials that need a visibility map.

    Namely: ground, reference panels, trees, and understory materials. Only materials
    that are in active use (i.e. objects using the materials are spawned by ground).
    """

    ground_gn = bpy.data.objects["Ground"].modifiers["GeometryNodes"]
    tree_like_objects = ["Tree 1", "Tree 2", "Tree 3", "Understory object 1", "Understory object 2"]
    material_socket_names = ["Trunk material", "Leaf material"]

    # Ground material is always in use. The rest we must figure out by looping through
    #   the objects that are spawned by ground.
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


def set_tree_parameter(tree_name: str, parameter_name: str, value):
    """Sets tree parameter to given value.

    :param tree_name:
        Name of the tree to be set ('Tree 1', 'Tree 2', etc.).
    :param parameter_name:
        Name of the parameter to be set.
    :param value:
        New value for the parameter. No checks on correct data type are
        made, so the caller must take care of that.
    """

    tree = trees[tree_name]
    tree_mod = tree.modifiers["GeometryNodes"]
    inputs = tree_mod.node_group.inputs
    socket = inputs.get(parameter_name)
    socket_id = socket.identifier
    old_val = tree_mod[socket.identifier]
    tree_mod[socket.identifier] = value
    if old_val != value:
        logging.error(f"Socket {socket.name} ({socket_id}) changed from {old_val} to {value}.")


def set_sun_power(power, frame):
    """Sets sun power for a single frame.

     For RGB images, frame 1 should be used. For hyperspectral
     rendering, frames are considered to be spectral bands.

    :param power:
        Power in W/m2.
    :param frame:
        Frame (spectral band for HSI images).
    """

    # bpy.data.materials["Lamp material"].node_tree.nodes["Emission"].inputs[1].default_value = power

    material = bpy.data.materials["Lamp material"]
    dp = f'nodes["Emission"].inputs[1].default_value'
    material.node_tree.nodes["Emission"].inputs[1].default_value = power
    material.node_tree.keyframe_insert(dp, frame=frame)

    # bpy.data.lights["Sun"].energy = power
    # dp = 'energy'
    # bpy.data.lights["Sun"].keyframe_insert(dp, frame=frame)


def set_sun_power_hsi(forest_id: str):
    """Set hyperspectral sun power based on control file.

    Sun power is set for each "animation" frame that represent
    different spectral bands separately
    """

    p = PH.path_file_forest_sun_csv(forest_id=forest_id)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Sun csv file '{p}' not found. Try rerunning forest initialization.")

    bands, _, irradiances = read_csv(p)

    control_dict = control.read_forest_control(forest_id=forest_id)
    sun_power = control_dict['Sun'][FC.key_ctrl_sun_base_power_hsi]
    irradiances = np.array(irradiances) * sun_power

    for i,band in enumerate(bands):
        set_sun_power(irradiances[i], band)


def read_csv(path):

    bands = []
    wavelengths = []
    values = []

    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot read csv file from '{path}'. File not found.")

    with open(path) as file:
        reader = csv.reader(file, delimiter=' ')
        for row in reader:
            try:
                bands.append(int(row[0]))
                wavelengths.append(float(row[1]))
                values.append(float(row[2]))
            except ValueError:
                # this is ok
                # print(f"Material headers: {row}")
                pass

    return bands, wavelengths, values


def list_forest_parameters():
    mod = bpy.data.objects["Ground"].modifiers["GeometryNodes"]
    for input_socket in mod.node_group.inputs:
        print(f"Input {input_socket.identifier} is named {input_socket.name}")


def list_collection_items(collection_name):
    print(f'{collection_name} collection contains:')
    for item in data.collections[collection_name].all_objects:
        print(f"\t{item.name}: location {item.location}")


def list_materials():
    print(f'Available materials:')
    for item in data.materials:
        print(f"\t{item.name}")


def list_tree_parameter_names():
    gn = trees[0].modifiers['GeometryNodes'].node_group
    # print(f"Accessing {gn.name}")
    tree_geometry_node = gn.nodes.get('Group')
    for input in tree_geometry_node.inputs:
        print(f"{input.name} ({input.type})")
