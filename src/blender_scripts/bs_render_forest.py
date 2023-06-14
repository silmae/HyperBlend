import bpy
import os
import sys  # to get command line args
import argparse  # to parse options for us and print a nice help message
import logging
import importlib

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
b_scene = b_data.scenes[FC.key_scene_name]
# TODO should this be taken as : scene = bpy.context.scene ?


cameras = b_data.collections[FC.key_collection_cameras].all_objects
lights = b_data.collections[FC.key_collection_lights].all_objects
trees = b_data.collections[FC.key_collection_trees].all_objects
tree_collection = b_data.collections[FC.key_collection_trees]
leaf_collection = b_data.collections[FC.key_collection_leaves]
leaves = b_data.collections[FC.key_collection_leaves].all_objects
# markers = b_data.collections[FC.key_collection_markers].all_objects
ground = b_data.collections[FC.key_collection_ground].all_objects
ground_collection = b_data.collections[FC.key_collection_ground]


def get_material_names_and_indices():
    # for material in bpy.data.materials:
    #     material_indices.append(material.pass_index)
    #     material_names.append(material.name)

    materials = bpy.data.materials
    material_names = list(m.name for m in materials)
    material_indices = list(m.pass_index for m in materials)
    res_dict = dict(zip(material_indices, material_names))
    return res_dict


def set_materials_use_spectral(use_spectral: bool):

    materials = bpy.data.materials
    materials_to_set = []
    for material in materials:
        name = material.name
        if "Leaf" in name or "Trunk" in name or "Ground" in name or "World" in name:
            materials_to_set.append(name)

    for material_name in materials_to_set:
        bpy.data.materials[material_name].node_tree.nodes["Group"].inputs["Use spectral"].default_value = use_spectral

    # bpy.data.worlds["World"].node_tree.nodes["Group"].inputs[7].default_value

    # bpy.data.materials["Leaf material 1"].node_tree.nodes["Group"].inputs["Use spectral"].default_value = use_spectral
    # bpy.data.materials["Leaf material 2"].node_tree.nodes["Group"].inputs["Use spectral"].default_value = use_spectral
    # bpy.data.materials["Leaf material 3"].node_tree.nodes["Group"].inputs["Use spectral"].default_value = use_spectral
    #
    # bpy.data.materials["Trunk material 1"].node_tree.nodes["Group"].inputs["Use spectral"].default_value = use_spectral
    # bpy.data.materials["Trunk material 2"].node_tree.nodes["Group"].inputs["Use spectral"].default_value = use_spectral
    # bpy.data.materials["Trunk material 3"].node_tree.nodes["Group"].inputs["Use spectral"].default_value = use_spectral
    #
    # bpy.data.materials["Ground material"].node_tree.nodes["Group"].inputs["Use spectral"].default_value = use_spectral
    #
    # if use_spectral:
    #     bpy.data.worlds["World"].node_tree.nodes["Group"].inputs['Strength'].default_value = 0
    # else:
    #     bpy.data.worlds["World"].node_tree.nodes["Group"].inputs['Strength'].default_value = 2


def set_render_parameters(render_mode: str='spectral', camera: str='Drone RGB', res_x=512, res_y=512, res_percent=100):
    """Jau

    :param render_mode:
        Either 'spectral' 'abundances' or 'rgb'.
    :param res_x:
    :param res_y:
    :param res_percent:
    :return:
    """

    # Always render with real objects
    FU.set_forest_parameter('Simplified trees', False)
    FU.set_forest_parameter('Simplified understory', False)

    # just in case we have multiple scenes at some point loop them over
    for scene in b_data.scenes:

        scene.sequencer_colorspace_settings.name = 'Raw'
        # Video sequenser can be always set to Raw as it only affects video editing

        # Compositing setup
        composite_raw()
        if render_mode == 'abundances':
            composite_material_mask()
        else:
            composite_delete_masking_setup()

        if render_mode.lower() == 'spectral' or render_mode.lower() == 'abundances':

            scene.render.image_settings.file_format = 'TIFF'  # OK
            scene.render.image_settings.tiff_codec = 'NONE'
            scene.render.image_settings.color_mode = 'BW'  # 'BW', 'RGB', 'RGBA'

            scene.camera = cameras.get('Drone HSI')

            scene.display_settings.display_device = 'None'
            # scene.view_settings.view_transform = 'Raw' # cannot be used when display device is None as it already turns off the display color transformations
            scene.view_settings.look = 'None'
            scene.view_settings.exposure = 0
            scene.view_settings.gamma = 1

            set_materials_use_spectral(True)

            # disable sky for spectral images

            # bpy.data.node_groups["Ground geometry"].nodes["Group.004"].inputs['Show white reference'].default_value = True

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

            set_materials_use_spectral(False)

            # bpy.data.node_groups["Ground geometry"].nodes["Group.004"].inputs['Show white reference'].default_value = False

        else:
            raise AttributeError(f"Parameter render_mode in set_render_parameters() must be either 'spectral', 'abundances' or 'rgb'. Was '{render_mode}'.")

        scene.render.image_settings.color_depth = '16'


        scene.render.resolution_x = res_x # OK
        scene.render.resolution_y = res_y # OK

        if res_percent > 100:
            res_percent = 100
        if res_percent < 1:
            res_percent = 1
        scene.render.resolution_percentage = res_percent # OK

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


def set_visibility(mode: str):

    def hide(obj):
        print(f"Hiding object '{obj.name}'.")
        obj.hide_render = True
        obj.hide_set(True)

    def unhide(obj):
        print(f"Unhiding object '{obj.name}'.")
        obj.hide_render = False
        obj.hide_set(False)

    if mode != FC.key_cam_sleeper_rgb and mode != FC.key_cam_walker_rgb and mode != FC.key_cam_drone_rgb and mode != 'Map' and mode != FC.key_cam_drone_hsi and mode != FC.key_cam_tree_rgb:
        raise AttributeError(f"Visibility for mode '{mode}' not recognised.")

    """
    Per documentation https://docs.blender.org/api/master/info_gotcha.html#unfortunate-corner-cases,
    we have to make a separate copy of the iterator to change object attributes without crashing, thus the [:]
    """

    # First hide everything
    tree_collection.hide_render = True
    for obj in trees[:]:
        hide(obj)
    leaf_collection.hide_render = True
    for obj in leaves[:]:
        hide(obj)
    ground_collection.hide_render = False
    for obj in ground[:]:
        hide(obj)

    unhide(lights.get(FC.key_obj_sun)) # always show sun
    # TODO sky?

    if mode == FC.key_cam_sleeper_rgb or mode == FC.key_cam_walker_rgb or mode == FC.key_cam_drone_rgb or mode == 'Map' or mode == FC.key_cam_drone_hsi:
        unhide(ground.get(FC.key_obj_ground))
    elif mode == FC.key_cam_tree_rgb:
        unhide(ground.get(FC.key_obj_ground_test))
        tree_collection.hide_render = False
        ground_collection.hide_render = False
        for tree in trees[:]:
            unhide(tree)


############## Compositing #########################################


def composite_raw():
    """Set up render output (compositing) to produce raw render result.

    This should used for any other renders but abundance maps. If this is not called, there
    is no connected sockets to output and the rendered image is completely black.
    """

    node_tree = b_scene.node_tree
    src = node_tree.nodes["Render Layers"]
    dst = node_tree.nodes["Composite"]
    node_tree.links.new(src.outputs['Image'], dst.inputs['Image'])


def composite_material_mask():
    """Assign pass indices to materials for abundance maps.

    Adapted from
    https://blenderartists.org/t/simple-script-to-create-a-unique-material-index-for-all-cycles-materials-in-the-scene/581087

    Deletes any existing masking setup before building a new one.
    """

    node_tree = b_scene.node_tree
    src = node_tree.nodes["Render Layers"]

    b_scene.view_layers["ViewLayer"].use_pass_material_index = True
    # enable material pass layer in case it was off

    composite_delete_masking_setup()
    # Delete possible old setup before building it again.

    f_output = node_tree.nodes.new('CompositorNodeOutputFile') # Create new File Output node

    # Set saving path and image settings
    f_output.base_path = PH.path_directory_forest_rend_abundances(SCENE_ID)
    f_output.format.file_format = 'TIFF'
    f_output.format.color_mode = 'BW' # no colors needed
    f_output.format.tiff_codec = 'NONE' # no packing of images
    f_output.format.color_depth = '16' # For some reason, 8 bit images look horrible so let's stick with 16 bits
    f_output.width = 400 # node width in Blender Compositing view

    # For positioning nodes in readable fashion in Blender Compositing view
    x_offset = 300
    y_offset = 100
    x, y = src.location

    f_output.location = (x + 2 * x_offset, y - y_offset)

    # Loop through materials assigning material indices and creating ID Mask
    # nodes for each.
    for idx, material in enumerate(bpy.data.materials):
        socet_name = f"{material.name}_"

        ID = node_tree.nodes.new('CompositorNodeIDMask')
        ID.label = f"{ID.name}_mat_{material.name}"
        ID.index = idx
        ID.location = (x + x_offset, y - idx * y_offset)

        # Create an input socket for each material pass
        if not material.name in (slot.path for slot in f_output.file_slots[:]):
            f_output.file_slots.new(socet_name)

        # Link ID Mask nodes to File Output node
        node_tree.links.new(src.outputs['IndexMA'], ID.inputs[0])
        node_tree.links.new(ID.outputs[0], f_output.inputs[socet_name])


def composite_delete_masking_setup():
    """Material masking setup must be deleted for any other renders than abundance rendering.

    Otherwise, abundance masks are rewritten for every rendered frame. NOTE: this deletes any
    existing ID Mask and File Output nodes from Compositing.
    """

    node_tree = b_scene.node_tree

    # Cannot delete while iterating, so just collect nodes to be deleted
    to_delete = []
    for node in node_tree.nodes:
        if node.bl_idname == 'CompositorNodeIDMask' or node.bl_idname == 'CompositorNodeOutputFile':
            to_delete.append(node)

    # And then delete them all from the node tree
    for node in to_delete:
        node_tree.nodes.remove(node)


############## Render calls #########################################


def call_blender_render(write_still=True, animation=False):
    """Saves current scene before calling Blender rendering.

    If the rendering crashes, one can inspect the scene file to find out what went wrong.
    """

    bpy.ops.wm.save_as_mainfile(filepath=PH.path_file_forest_scene(SCENE_ID))
    b_ops.render.render(write_still=write_still, animation=animation)


def render_sleeper_rgb():

    set_render_parameters(render_mode='rgb', camera='Sleeper RGB', res_x=1028, res_y=512, res_percent=100)
    set_visibility(mode='Sleeper RGB')
    FU.set_forest_parameter('Use real object', True)
    image_name = f'sleeper_rgb.png'
    image_path = PH.join(PH.path_directory_forest_rend(SCENE_ID), image_name)
    logging.info(f"Trying to render '{image_path}'.")
    b_scene.render.filepath = image_path
    call_blender_render(write_still=True)


def render_walker_rgb():

    set_render_parameters(render_mode='rgb', camera='Walker RGB', res_x=1028, res_y=512, res_percent=100)
    set_visibility(mode='Walker RGB')
    FU.set_forest_parameter('Use real object', True)
    image_name = f'walker_rgb.png'
    image_path = PH.join(PH.path_directory_forest_rend(SCENE_ID), image_name)
    logging.info(f"Trying to render '{image_path}'.")
    b_scene.render.filepath = image_path
    call_blender_render(write_still=True)


def render_drone_rgb():

    set_render_parameters(render_mode='rgb', camera='Drone RGB', res_x=1028, res_y=512, res_percent=100)
    set_visibility(mode='Drone RGB')
    image_name = f'drone_rgb.png'
    image_path = PH.join(PH.path_directory_forest_rend(SCENE_ID), image_name)
    logging.info(f"Trying to render '{image_path}'.")
    b_scene.render.filepath = image_path
    call_blender_render(write_still=True)


def render_tree_rgb():

    set_render_parameters(render_mode='rgb', camera='Tree RGB', res_x=1028, res_y=512, res_percent=100)
    set_visibility(mode='Tree RGB')
    image_name = f'tree_rgb.png'
    image_path = PH.join(PH.path_directory_forest_rend(SCENE_ID), image_name)
    logging.info(f"Trying to render '{image_path}'.")
    b_scene.render.filepath = image_path
    call_blender_render(write_still=True)


def render_map_rgb():

    set_render_parameters(render_mode='rgb', camera='Drone RGB', res_x=512, res_y=512, res_percent=100)
    set_visibility(mode='Drone RGB')
    FU.set_forest_parameter('Use real object', False)
    image_name = f'map_rgb.png'
    image_path = PH.join(PH.path_directory_forest_rend(SCENE_ID), image_name)
    logging.info(f"Trying to render '{image_path}'.")
    b_scene.render.filepath = image_path
    call_blender_render(write_still=True)


def render_drone_hsi():
    set_render_parameters(render_mode='spectral', camera='Drone HSI', res_x=512, res_y=512, res_percent=100)
    set_visibility(mode='Drone HSI')
    b_scene.render.filepath = PH.join(PH.path_directory_forest_rend_spectral(SCENE_ID), "band_####.tiff")
    call_blender_render(write_still=True, animation=True)


def render_abundances():

    set_render_parameters(render_mode='abundances', camera='Drone HSI', res_x=512, res_y=512, res_percent=100)
    set_visibility(mode='Drone HSI')
    FU.set_forest_parameter('Use real object', True)
    image_name = f'abundance_rgb_preview.png'
    image_path = PH.join(PH.path_directory_forest_rend_abundances(SCENE_ID), image_name)
    logging.info(f"Trying to render '{image_path}'.")
    b_scene.render.filepath = image_path
    call_blender_render(write_still=True)


if __name__ == '__main__':

    # Store arguments passed from blender_control.py
    argv = sys.argv

    if "--" not in argv:
        argv = []  # no arguments for the script
    else:
        argv = argv[argv.index("--") + 1:]  # get all args after "--"

    # Argument names
    key_scene_id = ['-id', '--scene_id']
    key_render_mode = ['-rm', '--render_mode']

    parser = argparse.ArgumentParser()

    parser.add_argument(key_scene_id[0], key_scene_id[1], dest=key_scene_id[1], action="store",
                        required=True, help="Scene id.")
    parser.add_argument(key_render_mode[0], key_render_mode[1], dest=key_render_mode[1], action="store",
                        required=True, help="Rendering mode")

    args = parser.parse_args(argv)

    SCENE_ID = vars(args)[key_scene_id[1]]

    logging.error(f"Hello, I am forest render script in '{PH.path_directory_forest_scene(SCENE_ID)}'")

    RENDER_MODE = vars(args)[key_render_mode[1]]

    # FU.list_forest_parameters()

    # TODO set sun power for rgb and spectra separately. Must set
    #  the active frame also for rgb renders. Or just set sun
    #  power the same for all frames.

    if RENDER_MODE.lower() == 'preview':
        render_sleeper_rgb()
        render_walker_rgb()
        render_drone_rgb()
        render_map_rgb()
        render_tree_rgb()
    elif RENDER_MODE.lower() == 'spectral':
        render_drone_hsi()
    elif RENDER_MODE.lower() == 'abundances':
        render_abundances()
    else:
        logging.error(f"Render mode '{RENDER_MODE}' not recognised.")

    bpy.ops.wm.save_as_mainfile(filepath=PH.path_file_forest_scene(SCENE_ID))
