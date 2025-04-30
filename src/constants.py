"""
Constants used throughout the program. Mostly file names and dictionary keys.

Do not change these, it may break reading of existing optimization results and rendering.

"""


imaging_type_refl = 'refl'
"""Imaging type reflectance (for rendering). """

imaging_type_tran = 'tran'
"""Imaging type transmittance (for rendering). """

target_type_slab = 'slab'
"""Rendering target is the slab object."""

target_type_ref = 'reference'
"""Rendering target (white) reference."""

postfix_render_image_format = '.tif'
"""Image format of images rendered with Blender."""

postfix_plot_image_format = '.png'
"""Image format of plots."""

postfix_text_data_format = '.toml'
"""Text file format with witch the results are saved."""

blender_executable_path_win = 'C:\Program Files\Blender Foundation\Blender 3.6/blender.exe'
"""change this to where your Blender is installed."""

blender_foundation_win = 'C:\Program Files\Blender Foundation'
"""By default, all Blender versions on Windows are installed here."""

blender_executable_path_linux = '/snap/bin/blender'
"""Default location of Blender on Ubuntu."""

blender_scene_slab_simulation_template = 'slab_sim_template.blend'
"""Name of the slab simulation template Blender scene."""

blender_script_name = 'bs_render_single.py'
"""Name of the Blender script to be run."""

ramdisk = '/media/ramdisk'
"""Location of the ramdisk if used."""

"""Project root relative to working folder that is assumed to be project_root/src/."""
path_project_root = '../'

# Names of the directories
# NOTE do not change render folder names as they are used by Blender script

dirname_internal = "Internal"

dirname_system_sim = "System simulation"

dirname_light_spectra = "Light spectra"

dirname_reflectance_spectra = "Reflectance spectra"

dirname_source = "src"

dirname_blender_scripts = "blender_scripts"

dirname_definitions = "definitions"

dirname_gsv = "gsv"

dirname_system_sim_rend = "rend"

dirname_system_sim_spectral_rend = "Spectral"

dirname_system_sim_visibility_maps_rend = "Visibility maps"

dirname_system_cube = "Spectral cube"

dirname_slab_sim_rend = 'rend'
"""Folder name for leaf target renders."""

folder_rend_ref_refl = 'rend_refl_ref'
"""Folder name for reflectance white reference renders."""

folder_rend_ref_tran = 'rend_tran_ref'
"""Folder name for transmittance white reference renders."""

dirname_system_simulation = 'System simulation'
"""Top level system simulation directory name."""

dirname_slab_simulation = 'Slab simulation'
"""Top level slab simulation directory name."""

dirname_slab_models = 'Slab models'
"""Top level Slab model directory name."""

dirname_slab_models_default = 'Default slab'
"""Default slab model directory name."""

dirname_opt_target_signal = 'Target signal'
"""Target signal files are stored to this directory."""

dirname_result_signal = 'Result signal'
"""Stores simulated signals matching to target signals. """

signal_directory_prefix = 'Signal'
"""Prefix for signal directories within a single slab simulation such as 'signal_0'. """

dirname_slab_sim_working_temp = 'working_temp'
"""Top level folder where rendered images of slab simulation are saved in their own subdirectories."""

dirname_optimization_results = 'Optimization results'
"""Wavelength-wise results for each signal solved by optimization method are stored here (numerical and png images)."""


# file names

file_opt_target = 'target'
"""Target file name prefix as in 'target_0.toml'."""

file_opt_res = 'final_result'
"""Set result file name."""

filename_slab_sim_result = 'slab_sim_result'
"""File name of the final result toml and plot of a slab simulation.."""

filename_result_signal = 'result_signal'
"""Simulated result signal of a slab simulation. This is postfixed with the signal id by :mod:`file_names`."""

filename_model_parameters = 'surface_model_params'
"""Surface model parameters file."""

file_default_sun = 'default_sun.txt'
"""Default sun spectrum file name that is included in the repository."""

file_default_sky = 'default_sky.txt'
"""Default sky spectrum file name that is included in the repository."""

file_sampling_data = 'sampling'
"""Toml formatted file storing wavelengths for resampling."""

filename_system_sim_control = 'system_sim_control'
"""Toml formatted file for system simulation scene control parameters."""

filename_system_sim_forest_template = 'system_sim_forest_template'

filename_slab_sim_forest_template = 'slab_sim_template'

file_directory_structure = 'directory_structure'
"""Toml formatted file that defines directory structure and core files."""

file_app_info = 'app_info'
"""Toml formatted file that application info such as version number."""

file_dynamic_definitions = 'dynamic'
"""Toml formatted file that dynamically inferred info such as operating system."""

# Resampling keys

key_sampling_wl = 'sampling_wl'
# Dictionary key for resampling data


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

key_wl_result_history_loss_total = 'loss'
"""History of loss listed for each render run."""

key_wl_result_history_loss_over_one = 'over_one'

key_wl_result_history_loss_t = 't_loss'

key_wl_result_history_loss_r = 'r_loss'

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
"""Total mean of RMSEs of reflectance over all samples and wavelengths. """

key_set_result_te_mean = 'tran_error_mean'
"""Total mean of RMSEs of transmittance over all samples and wavelengths. """

key_set_result_r_mean  = 'set_result_r_mean'
"""Total mean of modeled reflectances over all samples and wavelengths. """

key_set_result_t_mean  = 'set_result_t_mean'
"""Total mean of modeled transmittances over all samples and wavelengths. """

key_set_result_rm_mean = 'set_result_rm_mean'
"""Total mean of measured reflectances over all samples and wavelengths. """

key_set_result_tm_mean = 'set_result_tm_mean'
"""Total mean of measured transmittances over all samples and wavelengths. """

key_set_result_r_std  = 'set_result_tm_std'
"""Total standard deviation of modeled reflectances over all samples and wavelengths. """

key_set_result_t_std  = 'set_result_tm_std'
"""Total standard deviation of modeled transmittances over all samples and wavelengths. """

key_set_result_rm_std = 'set_result_tm_std'
"""Total standard deviation of measured reflectances over all samples and wavelengths. """

key_set_result_tm_std = 'set_result_tm_std'
"""Total standard deviation of measured transmittances over all samples and wavelengths. """

key_set_result_re_std = 'refl_error_std'
"""Total standard deviation of RMSEs of reflectance over all samples and wavelengths."""

key_set_result_te_std = 'tran_error_std'
"""Total standard deviation of RMSEs of transmittance over all samples and wavelengths. """

key_set_result_wls = 'wls'
"""List of wavelengths used in the set."""

key_set_result_wl_r_mean = 'set_result_wl_r_mean'
"""Wavelength-wise mean of modeled reflectance. """

key_set_result_wl_t_mean = 'set_result_wl_t_mean'
"""Wavelength-wise mean of modeled transmittance. """

key_set_result_wl_rm_mean = 'set_result_wl_rm_mean'
"""Wavelength-wise mean of measured reflectance. """

key_set_result_wl_tm_mean = 'set_result_wl_tm_mean'
"""Wavelength-wise mean of measured transmittance. """

key_set_result_wl_re_mean = 'set_result_wl_re_mean'
"""Wavelength-wise mean of reflectance error. """

key_set_result_wl_te_mean = 'set_result_wl_te_mean'
"""Wavelength-wise mean of transmittance error. """

key_set_result_wl_ad_mean = 'set_result_wl_ad_mean'
"""Wavelength-wise mean of absorption particle density. """

key_set_result_wl_sd_mean = 'set_result_wl_sd_mean'
"""Wavelength-wise mean of scattering particle density. """

key_set_result_wl_ai_mean = 'set_result_wl_ai_mean'
"""Wavelength-wise mean of scattering anisotropy. """

key_set_result_wl_mf_mean = 'set_result_wl_mf_mean'
"""Wavelength-wise mean of mix factor. """


key_set_result_wl_r_std = 'set_result_wl_r_std'
"""Wavelength-wise standard deviation of modeled reflectance. Set to zero is only one sample was used. """

key_set_result_wl_t_std = 'set_result_wl_t_std'
"""Wavelength-wise standard deviation of modeled transmittance. Set to zero is only one sample was used. """

key_set_result_wl_rm_std = 'set_result_wl_rm_std'
"""Wavelength-wise standard deviation of measured reflectance. Set to zero is only one sample was used. """

key_set_result_wl_tm_std = 'set_result_wl_tm_std'
"""Wavelength-wise standard deviation of measured transmittance. Set to zero is only one sample was used. """

key_set_result_wl_re_std = 'set_result_wl_re_std'
"""Wavelength-wise standard deviation of reflectance error. Set to zero is only one sample was used. """

key_set_result_wl_te_std = 'set_result_wl_te_std'
"""Wavelength-wise standard deviation of transmittance error. Set to zero is only one sample was used. """

key_set_result_wl_ad_std = 'set_result_wl_ad_std'
"""Wavelength-wise standard deviation of absorption particle density. Set to zero is only one sample was used. """

key_set_result_wl_sd_std = 'set_result_wl_sd_std'
"""Wavelength-wise standard deviation of scattering particle density. Set to zero is only one sample was used. """

key_set_result_wl_ai_std = 'set_result_wl_ai_std'
"""Wavelength-wise standard deviation of scattering anisotropy. Set to zero is only one sample was used. """

key_set_result_wl_mf_std = 'set_result_wl_mf_std'
"""Wavelength-wise standard deviation of mix factor. Set to zero is only one sample was used. """


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


# Default RGB wavelengths for false color images
default_R_wl = 630.
"""Default Red wavelength for false color images."""
default_G_wl = 532.
"""Default Green wavelength for false color images."""
default_B_wl = 465.
"""Default Blue wavelength for false color images."""
