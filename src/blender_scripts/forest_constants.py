"""
Names of things in forest blend file. All of them are case-sensitive.

"""

key_scene_name = 'Forest'

########### Collection names ###########

key_collection_cameras = 'Cameras'
key_collection_lights = 'Lighting'
key_collection_trees = 'Trees'
key_collection_leaves = 'Leaves'
key_collection_markers = 'Marker objects'
key_collection_ground = 'Ground'


########### Object names ###########

key_obj_sun = 'Sun'

key_obj_ground = 'Ground'
key_obj_ground_test = 'Test ground'

key_obj_tree_1 = 'Tree 1'
key_obj_tree_2 = 'Tree 2'
key_obj_tree_3 = 'Tree 3'

key_drone = 'Drone'
"""Drone object that has Drone HSI and Drone RGB cameras attached to it."""

key_cam_drone_hsi = 'Drone HSI'
key_cam_drone_rgb = 'Drone RGB'
key_cam_walker_rgb = 'Walker RGB'
key_cam_sleeper_rgb = 'Sleeper RGB'
key_cam_tree_rgb = 'Tree RGB'

#######################################

max_sun_power_spectral = 40
max_sun_power_rgb = 400

########### Scene control file keys ###########

key_ctrl_drone_location_x = "drone_location_x"
key_ctrl_drone_location_y = "drone_location_y"
key_ctrl_drone_altitude = "drone_altitude"

key_ctrl_drone_hsi_fow = "drone_hsi_fow"
"""Drone hyperspectral camera's field of view."""
key_ctrl_drone_rgb_fow = "drone_rgb_fow"
"""Drone RGB camera's field of view."""

key_ctrl_hsi_resolution_x = "hsi_resolution_x"
key_ctrl_hsi_resolution_y = "hsi_resolution_y"
key_ctrl_rgb_resolution_x = "rgb_resolution_x"
key_ctrl_rgb_resolution_y = "rgb_resolution_y"
key_ctrl_walker_resolution_x = "walker_resolution_x"
key_ctrl_walker_resolution_y = "walker_resolution_y"
key_ctrl_sleeper_resolution_x = "sleeper_resolution_x"
key_ctrl_sleeper_resolution_y = "sleeper_resolution_y"
key_ctrl_tree_preview_resolution_x = "tree_preview_resolution_x"
key_ctrl_tree_preview_resolution_y = "tree_preview_resolution_y"

key_ctrl_sun_angle_zenith_deg = "sun_angle_zenith_deg"
key_ctrl_sun_angle_azimuth_deg = "sun_angle_azimuth_deg"
key_ctrl_sun_base_power_hsi = "sun_base_power_hsi"
key_ctrl_sun_base_power_rgb = "sun_base_power_rgb"
