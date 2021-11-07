"""
Constants used throughout the program. Mostly file names and dictionary keys.

Do not change these, it may break reading of existing optimization results and rendering.

"""


imaging_type_refl = 'refl'
"""Imaging type reflectance (for rendering). """

imaging_type_tran = 'tran'
"""Imaging type transmittance (for rendering). """

target_type_leaf = 'leaf'
"""Rendering target leaf object."""

target_type_ref = 'reference'
"""Rendering target (white) reference."""

postfix_image_format = '.tif'
"""Image format of images rendered with Blender."""

postfix_text_data_format = '.toml'
"""Text file format with witch the results are saved."""

blender_executable_path_win = 'C:\Program Files\Blender Foundation\Blender 2.92/blender.exe'
"""change this to where your Blender is installed."""

blender_executable_path_linux = '/snap/bin/blender'
"""Default location of Blender on Ubuntu."""

blender_scene_name = 'scene_leaf_material.blend'
"""Name of the Blender scene."""

blender_script_name = 'bs_render_single.py'
"""Name of the Blender script to be run."""

ramdisk = '/media/ramdisk'
"""Location of the ramdisk if used."""

"""Project root relative to working folder that is assumed to be project_root/src/."""
path_project_root = '../'

# folder names
# NOTE do not change render folder names as they are used by Blender script
folder_rend = 'rend'
"""Folder name for leaf target renders."""

folder_rend_ref_refl = 'rend_refl_ref'
"""Folder name for reflectance white reference renders."""

folder_rend_ref_tran = 'rend_tran_ref'
"""Folder name for transmittancewhite reference renders."""

folder_opt = 'optimization'
"""Top level optimization folder name."""

folder_opt_sample_targets = 'sample_targets'
"""Target files are stored to this folder."""

folder_opt_sample_results = 'sample_results'
"""Sample result numerical data is stored in here. """

folder_sample_prefix = 'sample'
"""Prefix for sample folder names as in 'sample_0'. """

folder_opt_work = 'working_temp'
"""Top level folder where rendered images are saved in their own subfolders."""

folder_opt_result = 'result'
"""Top level folder for optimization results."""

folder_opt_subresult = 'sub_results'
"""Wavelength-wise results for each sample are stored here (numerical and png images)."""

folder_set_result = 'set_result'
"""Set result is saved here."""

# file names

file_opt_target = 'target'
"""Target file name prefix as in 'target_0.toml'."""

file_opt_res = 'final_result'
"""Set result file name."""

file_sample_result = 'sample_result'
"""Sample result file name."""

# Wavelength result's dictionary keys

key_wl_result_wl = 'wl'
"""Wavelength."""

key_wl_result_x0 = 'x0'
"""Starting guess used for the wavelength optimization."""

key_wl_result_x_best = 'x_best'
"""Optimized leaf material variables for the wavelength."""

key_wl_result_refl_measured = 'reflectance_measured'
"""Measured (target) reflectance."""

key_wl_result_tran_measured = 'transmittance_measured'
"""Measured (target) transmittance."""

key_wl_result_refl_modeled = 'reflectance_modeled'
"""Modeled reflectance."""

key_wl_result_tran_modeled = 'transmittance_modeled'
"""Modeled transmittance."""

key_wl_result_refl_error = 'reflectance_error'
"""Error of modeled reflectance."""

key_wl_result_tran_error = 'transmittance_error'
"""Error of modeled transmittance."""

key_wl_result_render_calls = 'render_calls'
"""Render calls. Used in optimization history. """

key_wl_result_elapsed_time_s = 'elapsed_time_s'
"""Time in seconds to optimize a single wavelength."""

key_wl_result_history_r = 'history_reflectance'
"""History of reflectance listed for each render run."""

key_wl_result_history_t = 'history_transmittance'
"""History of transmittance  listed for each render run."""

key_wl_result_history_ad = 'history_absorption_density'
"""History of absorption density listed for each render run."""

key_wl_result_history_sd = 'history_scattering_density'
"""History of scattering density listed for each render run."""

key_wl_result_history_ai = 'history_scattering_anisotropy'
"""History of scattering anisotropy listed for each render run."""

key_wl_result_history_mf = 'history_mix_factor'
"""History of mix factor listed for each render run."""

key_wl_result_optimizer = 'optimizer'
"""Optimizer name used in optimization."""

key_wl_result_optimizer_result = 'optimizer_result'
"""The content of the optimizer result depends on the optimizer used. """

key_wl_result_optimizer_ftol = 'ftol'
"""Used function value (error to measured) tolerance to stop optimization."""

key_wl_result_optimizer_xtol = 'xtol'
"""Used variable change tolerance (between iterations) to stop optimization."""

key_wl_result_optimizer_diffstep = 'diffstep'
"""Used relative step size when approximating the Jacobian in optimizer."""


# Sample result dictionary keys

key_sample_result_wall_clock_elapsed_min = 'wall_clock_time_elapsed_min'
"""Wall clock time used to optimize one sample."""

key_sample_result_process_elapsed_min = 'process_elapsed_min'
"""Processor time used to optimize one sample."""

key_sample_result_r_RMSE = 'r_RMSE'
"""Root mean squared error of reflectance of a sample."""

key_sample_result_t_RMSE = 't_RMSE'
"""Root mean squared error of transmittance of a sample."""

key_sample_result_wls = 'wls'
"""List of wavelengths used in a sample."""

key_sample_result_r = 'refls_modeled'
"""List of modeled reflectances."""

key_sample_result_rm = 'refls_measured'
"""List of measured (target) reflectances."""

key_sample_result_re = 'refls_error'
"""List of errors in reflectance. """

key_sample_result_t = 'trans_modeled'
"""List of modeled transmittances."""

key_sample_result_tm = 'trans_measured'
"""List of measured (target) transmittances."""

key_sample_result_te = 'trans_error'
"""List of errors in transmittance. """

key_sample_result_ad = 'absorption_density'
"List of optimized absorption densities."

key_sample_result_sd = 'scattering_density'
"List of optimized scattering densities."

key_sample_result_ai = 'scattering_anisotropy'
"List of optimized scattering anisotropies."

key_sample_result_mf = 'mix_factor'
"List of optimized mix factors."


# Set result dictionary keys

key_set_result_sample_count = 'sample_count'
"""Number of samples in a set."""

key_set_result_total_time_hours = 'total_time_hours'
"""Total wall clock optimization time of all samples in hours."""

key_set_result_time_per_sample_hours = 'time_per_sample_hours'
"""Average wall clock optimization time per sample in hours."""

key_set_result_total_processor_time_hours = 'total_processor_time_hours'
"""Total processor time of all samples in hours."""

key_set_result_processor_time_per_sample_hours = 'processor_time_per_sample_hours'
"""Average processor time per sample in hours."""

key_set_result_re_mean = 'refl_error_mean'
"""Mean of RMSEs of reflectance over all samples. Listed by wavelength."""

key_set_result_te_mean = 'tran_error_mean'
"""Mean of RMSEs of transmittance over all samples. Listed by wavelength."""

key_set_result_re_std = 'refl_error_std'
"""Standard deviation of RMSEs of reflectance over all samples. Listed by wavelength."""

key_set_result_te_std = 'tran_error_std'
"""Standard deviation of RMSEs of transmittance over all samples. Listed by wavelength."""

# starting guess coefficients names
ad_coeffs = 'ad_coeffs'
"""List of coefficients of absorption particle density."""

sd_coeffs = 'sd_coeffs' # scattering density
"""List of coefficients of scattering particle density."""

ai_coeffs = 'ai_coeffs' # scattering anisotropy
"""List of coefficients of scattering anisotropy."""

mf_coeffs = 'mf_coeffs' # mixing factor
"""List of coefficients of mix factor."""

# Default starting guess
starting_guess_set_name = 'linear_starting_guess'
"""Name of the starting guess file."""
