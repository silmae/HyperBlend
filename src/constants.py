
imaging_type_refl = 'refl'
imaging_type_tran = 'tran'
target_type_leaf = 'leaf'
target_type_ref = 'reference'

postfix_image_format = '.tif'
postfix_text_data_format = '.toml'

blender_executable_path_win = 'C:\Program Files\Blender Foundation\Blender 2.92/blender.exe' # change this to where your Blender is installed
blender_executable_path_linux = '/snap/bin/blender' # default location on Ubuntu

# location of the ramdisk
ramdisk = '/media/ramdisk'

path_project_root = '../'

# folder names
# NOTE do not change render folder names as they are used by Blender script
folder_rend = 'rend'
folder_rend_ref_refl = 'rend_refl_ref'
folder_rend_ref_tran = 'rend_tran_ref'
folder_opt = 'optimization'
folder_opt_sample_targets = 'sample_targets'
folder_opt_sample_results = 'sample_results'
folder_sample_prefix = 'sample'
folder_opt_work = 'working_temp'
folder_opt_result = 'result'
folder_opt_subresult = 'sub_results'
folder_opt_plot = 'plot'
folder_set_result = 'set_result'

# file names
file_opt_target = 'target'
file_opt_res = 'final_result'
file_sample_result = 'sample_result'

# subresult toml keys
subres_key_wl = 'wl'
subres_key_reflectance_measured = 'reflectance_measured'
subres_key_transmittance_measured = 'transmittance_measured'
subres_key_reflectance_modeled = 'reflectance_modeled'
subres_key_transmittance_modeled = 'transmittance_modeled'
subres_key_reflectance_error = 'reflectance_error'
subres_key_transmittance_error = 'transmittance_error'
subres_key_iterations = 'iterations'
subres_key_elapsed_time_s = 'elapsed_time_s'
subres_key_history_reflectance = 'history_reflectance'
subres_key_history_transmittance = 'history_transmittance'
subres_key_history_absorption_density = 'history_absorption_density'
subres_key_history_scattering_density = 'history_scattering_density'
subres_key_history_scattering_anisotropy = 'history_scattering_anisotropy'
subres_key_history_mix_factor = 'history_mix_factor'
subres_key_optimizer = 'optmizer'
"""The content of the optimizer result depends on the 
optimizer used. """
subres_key_optimizer_result = 'optimiser_result'
subres_key_optimizer_ftol = 'ftol'
subres_key_optimizer_xtol = 'xtol'
subres_key_optimizer_diffstep = 'diffstep'

# final result toml keys
result_key_wall_clock_elapsed_min = 'wall_clock_time_elapsed_min'
result_key_process_elapsed_min = 'process_elapsed_min'
result_key_r_RMSE = 'r_RMSE'
result_key_t_RMSE = 't_RMSE'
result_key_wls = 'wls'
result_key_refls_modeled = 'refls_modeled'
result_key_refls_measured = 'refls_measured'
result_key_refls_error = 'refls_error'
result_key_trans_modeled = 'trans_modeled'
result_key_trans_measured = 'trans_measured'
result_key_trans_error = 'trans_error'
result_key_absorption_density = 'absorption_density'
result_key_scattering_density = 'scattering_density'
result_key_scattering_anisotropy = 'scattering_anisotropy'
result_key_mix_factor = 'mix_factor'
