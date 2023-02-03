"""
Methods in this module handle writing and reading of various toml formatted txt files.

Hierarchy of the results: final result is mean of sample results,
which is collected from wavelength results.
"""

import os

import numpy as np
import toml

from src.data import file_handling as FH
from src.data import file_names as FN
from src import constants as C, plotter
from src.data import path_handling as P


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

    p = P.join(directory, filename)
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

    p = P.join(directory, filename)

    if not os.path.exists(os.path.abspath(p)):
        raise RuntimeError(f"Cannot read from file '{os.path.abspath(p)}' "
                           f"because it does not exist.")

    with open(p, 'r') as file:
        result = toml.load(file)
    return result


def read_surface_model_parameters():
    """Reads surface model parameters from a file and returns them as a dictionary. """

    p = P.path_file_surface_model_parameters()
    if not os.path.exists(p):
        raise RuntimeError(f'Surface model parameter file "{p}" not found.')
    with open(p, 'r') as file:
        result = toml.load(file)
    return result


def write_surface_model_parameters(parameter_dict):
    """Writes surface model parameters to a file

    :param parameter_dict:
        Parameter dictionary to be saved.
    """

    p = P.path_file_surface_model_parameters()
    with open(p, 'w+') as file:
        toml.dump(parameter_dict, file, encoder=toml.encoder.TomlNumpyEncoder())


def write_set_result(set_name: str):
    """Collect sample results and write final result to a toml file. """

    result_dict = {}
    r = collect_sample_results(set_name)
    sample_count = len(r)
    result_dict[C.key_set_result_sample_count] = sample_count
    result_dict[C.key_set_result_total_time_hours] = np.sum([sr[C.key_sample_result_wall_clock_elapsed_min] for sr in r]) / 60
    result_dict[C.key_set_result_time_per_sample_hours] = np.sum([sr[C.key_sample_result_wall_clock_elapsed_min] for sr in r]) / 60 / sample_count
    result_dict[C.key_set_result_total_processor_time_hours] = np.sum([sr[C.key_sample_result_process_elapsed_min] for sr in r]) / 60
    result_dict[C.key_set_result_processor_time_per_sample_hours] = np.sum([sr[C.key_sample_result_process_elapsed_min] for sr in r]) / 60 / sample_count

    # Total means
    result_dict[C.key_set_result_r_mean]  = np.mean([sr[C.key_sample_result_r] for sr in r])
    result_dict[C.key_set_result_t_mean]  = np.mean([sr[C.key_sample_result_t] for sr in r])
    result_dict[C.key_set_result_rm_mean] = np.mean([sr[C.key_sample_result_rm] for sr in r])
    result_dict[C.key_set_result_tm_mean] = np.mean([sr[C.key_sample_result_tm] for sr in r])
    result_dict[C.key_set_result_re_mean] = np.mean([sr[C.key_sample_result_re] for sr in r])
    result_dict[C.key_set_result_te_mean] = np.mean([sr[C.key_sample_result_te] for sr in r])

    #Total standard deviations
    if sample_count > 1:
        result_dict[C.key_set_result_r_std]  = np.std([sr[C.key_sample_result_r] for sr in r])
        result_dict[C.key_set_result_t_std]  = np.std([sr[C.key_sample_result_t] for sr in r])
        result_dict[C.key_set_result_rm_std] = np.std([sr[C.key_sample_result_rm] for sr in r])
        result_dict[C.key_set_result_tm_std] = np.std([sr[C.key_sample_result_tm] for sr in r])
        result_dict[C.key_set_result_re_std] = np.std([sr[C.key_sample_result_re] for sr in r])
        result_dict[C.key_set_result_te_std] = np.std([sr[C.key_sample_result_te] for sr in r])
    else:
        # Standard deviation not defined for only one sample. Set to zero so that plots can still use it.
        result_dict[C.key_set_result_r_std]  = 0.
        result_dict[C.key_set_result_t_std]  = 0.
        result_dict[C.key_set_result_rm_std] = 0.
        result_dict[C.key_set_result_tm_std] = 0.
        result_dict[C.key_set_result_re_std] = 0.
        result_dict[C.key_set_result_te_std] = 0.

    result_dict[C.key_set_result_wls] = r[0][C.key_sample_result_wls]

    # Wavelength means
    result_dict[C.key_set_result_wl_r_mean]  = np.mean([sr[C.key_sample_result_r] for sr in r], axis=0)
    result_dict[C.key_set_result_wl_t_mean]  = np.mean([sr[C.key_sample_result_t] for sr in r], axis=0)
    result_dict[C.key_set_result_wl_rm_mean]  = np.mean([sr[C.key_sample_result_rm] for sr in r], axis=0)
    result_dict[C.key_set_result_wl_tm_mean]  = np.mean([sr[C.key_sample_result_tm] for sr in r], axis=0)
    result_dict[C.key_set_result_wl_re_mean] = np.mean([sr[C.key_sample_result_re] for sr in r], axis=0)
    result_dict[C.key_set_result_wl_te_mean] = np.mean([sr[C.key_sample_result_te] for sr in r], axis=0)
    result_dict[C.key_set_result_wl_ad_mean] = np.mean([sr[C.key_sample_result_ad] for sr in r], axis=0)
    result_dict[C.key_set_result_wl_sd_mean] = np.mean([sr[C.key_sample_result_sd] for sr in r], axis=0)
    result_dict[C.key_set_result_wl_ai_mean] = np.mean([sr[C.key_sample_result_ai] for sr in r], axis=0)
    result_dict[C.key_set_result_wl_mf_mean] = np.mean([sr[C.key_sample_result_mf] for sr in r], axis=0)

    # Wavelength standard deviations
    if sample_count > 1:
        result_dict[C.key_set_result_wl_r_std]  = np.std([sr[C.key_sample_result_r] for sr in r], axis=0)
        result_dict[C.key_set_result_wl_t_std]  = np.std([sr[C.key_sample_result_t] for sr in r], axis=0)
        result_dict[C.key_set_result_wl_rm_std] = np.std([sr[C.key_sample_result_rm] for sr in r], axis=0)
        result_dict[C.key_set_result_wl_tm_std] = np.std([sr[C.key_sample_result_tm] for sr in r], axis=0)
        result_dict[C.key_set_result_wl_re_std] = np.std([sr[C.key_sample_result_re] for sr in r], axis=0)
        result_dict[C.key_set_result_wl_te_std] = np.std([sr[C.key_sample_result_te] for sr in r], axis=0)
        result_dict[C.key_set_result_wl_ad_std] = np.std([sr[C.key_sample_result_ad] for sr in r], axis=0)
        result_dict[C.key_set_result_wl_sd_std] = np.std([sr[C.key_sample_result_sd] for sr in r], axis=0)
        result_dict[C.key_set_result_wl_ai_std] = np.std([sr[C.key_sample_result_ai] for sr in r], axis=0)
        result_dict[C.key_set_result_wl_mf_std] = np.std([sr[C.key_sample_result_mf] for sr in r], axis=0)
    else:
        # Standard deviation not defined for only one sample. Set to zero so that plots can still use it.
        result_dict[C.key_set_result_wl_r_std] = np.zeros_like(r[0][C.key_sample_result_wls])
        result_dict[C.key_set_result_wl_t_std] = np.zeros_like(r[0][C.key_sample_result_wls])
        result_dict[C.key_set_result_wl_rm_std] = np.zeros_like(r[0][C.key_sample_result_wls])
        result_dict[C.key_set_result_wl_tm_std] = np.zeros_like(r[0][C.key_sample_result_wls])
        result_dict[C.key_set_result_wl_re_std] = np.zeros_like(r[0][C.key_sample_result_wls])
        result_dict[C.key_set_result_wl_te_std] = np.zeros_like(r[0][C.key_sample_result_wls])
        result_dict[C.key_set_result_wl_ad_std] = np.zeros_like(r[0][C.key_sample_result_wls])
        result_dict[C.key_set_result_wl_sd_std] = np.zeros_like(r[0][C.key_sample_result_wls])
        result_dict[C.key_set_result_wl_ai_std] = np.zeros_like(r[0][C.key_sample_result_wls])
        result_dict[C.key_set_result_wl_mf_std] = np.zeros_like(r[0][C.key_sample_result_wls])

    p = P.join(P.path_directory_set_result(set_name), FN.filename_set_result())
    with open(p, 'w+') as file:
        toml.dump(result_dict, file, encoder=toml.encoder.TomlNumpyEncoder())


def read_set_result(set_name: str):
    """Reads the set result file. Created if does not exist."""

    p = P.join(P.path_directory_set_result(set_name), FN.filename_set_result())
    if not os.path.exists(p):
        write_set_result(set_name)
    with open(p, 'r') as file:
        result = toml.load(file)

    return result


def collect_sample_results(set_name: str):
    """Collect results of finished samples in a list of dictionaries.

    :param set_name:
        Set name
    :return:
        List of sample result dictionaries.
    """

    ids = FH.list_finished_sample_ids(set_name)
    collected_results = []
    for _,sample_id in enumerate(ids):
        sample_result_dict = read_sample_result(set_name, sample_id)
        collected_results.append(sample_result_dict)
    return collected_results


def read_sample_result(set_name: str, sample_id: int):
    """Reads sample result file into a dict and returns it.

    :param set_name:
        Set name.
    :param sample_id:
        Sample id.
    :return:
        Result file content as a dict.
    """

    p = P.join(P.path_directory_sample(set_name, sample_id), FN.filename_sample_result(sample_id))
    with open(p, 'r') as file:
        subres_dict = toml.load(file)

    return subres_dict


def write_sample_result(set_name: str, res_dict: dict, sample_id: int) -> None:
    """Writes sample result dictionary into a file.

    :param set_name:
        Set name.
    :param res_dict:
        Sample results as a dictionary to be written to disk.
    :param sample_id:
        Sample id.
    """

    p = P.join(P.path_directory_sample(set_name, sample_id), FN.filename_sample_result(sample_id))
    with open(p, 'w+') as file:
        toml.dump(res_dict, file, encoder=toml.encoder.TomlNumpyEncoder())


def collect_wavelength_result(set_name: str, sample_id: int):
    """Collects wavelength result dictionaries in to a list and returns it.

    :param set_name:
        Set name.
    :param sample_id:
        Sample id.
    :return:
        A list of wavelength result dictionaries.
    """

    p = P.path_directory_subresult(set_name, sample_id)
    subres_list = []
    for filename in os.listdir(p):
        if filename.endswith(C.postfix_text_data_format):
            subres = toml.load(P.join(p, filename))
            subres_list.append(subres)
    return subres_list


def write_wavelength_result(set_name: str, res_dict: dict, sample_id: int) -> None:
    """Writes wavelength result of optimization into a file.

    :param set_name:
        Set name.
    :param res_dict:
        Subresult dictionary.
    :param sample_id:
        Sample id.
    """

    wl = res_dict[C.key_wl_result_wl]
    p = P.path_file_wl_result(set_name, wl, sample_id)
    with open(p, 'w+') as file:
        toml.dump(res_dict, file, encoder=toml.encoder.TomlNumpyEncoder())


def read_wavelength_result(set_name: str, wl: float, sample_id: int):
    """Reads a wavelength result file into a dictionary and returns it.

    :param set_name:
        Set name.
    :param wl:
        Wavelength of the subresult.
    :param sample_id:
        Sample id.
    :return:
        Subresult as a dictionary.
    """

    p = P.path_file_wl_result(set_name, wl, sample_id)
    with open(p, 'r') as file:
        subres_dict = toml.load(file)

    return subres_dict


def write_target(set_name:str, data, sample_id=0) -> None:
    """Writes given list of reflectance and transmittance data to toml formatted file.

    Writes also empty sampling file to the target directory.

    :param set_name:
        Set name.
    :param data:
        List of lists or list of tuples [[wl, r, t], ...]
        Do not use numpy arrays as they may break the toml writer.
    :param sample_id:
        Sample id. Default is 0, which is used for sets with only one sample.
    """

    floated_list = [[float(a), float(b), float(c)] for (a, b, c) in data]
    res = {'wlrt': floated_list}
    p = P.path_file_target(set_name, sample_id)
    if not os.path.exists(p):
        FH.create_first_level_folders(set_name)
        FH.create_opt_folder_structure_for_samples(set_name, sample_id=0)
    with open(p, 'w+') as file:
        toml.dump(res, file)

    write_sampling(set_name)


def read_target(set_name: str, sample_id: int):
    """Read target values for optimization.

    :param set_name:
        Name of the set.
    :param sample_id:
        Sample id.
    :return:
        List of reflectances and transmittances per wavelength [[wl, r, t],...] as numpy array
    """

    with open(P.path_file_target(set_name, sample_id), 'r') as file:
        data = toml.load(file)
        data = data['wlrt']
        data = np.array(data)
        return data


def write_sampling(set_name: str, sampling: list=None):
    """Write resampling data file for given leaf measurement set.

    Preferred workflow is to NOT provide a list of wavelengths here,
    which will result an empty file where you can copy and paste desired
    wavelengths. Providing a wavelengths list here is good for debugging
    or quick experiments though.

    :param set_name:
        Name of the leaf measurement set.
    :param sampling:
        You can give a list of wavelengths here. If none is given, an
        empty wavelength dictionary is written to the file. You can
        later copy paste wavelengths from an ENVI file, for example.
    """

    if sampling is None:
        wls = []
    else:
        # Cast to float in case there were ints and floats mixed in given list.
        # We cannot guard against user-defined files though.
        wls = list(float(a) for a in sampling)

    sampling_dict = {C.key_sampling_wl: wls}

    p = P.path_file_sampling(set_name)

    with open(p, 'w+') as file:
        toml.dump(sampling_dict, file, encoder=toml.encoder.TomlNumpyEncoder())


def read_sampling(set_name: str,):
    """Read resampling wavelengths from a file.

    :param set_name:
        Name of the leaf measurement set.
    :return:
        Return resampling wavelengths as 1D numpy array.
    :raises
        Raises RuntimeError in case some of the entries could not be interpreted as a float.
    """
    p = P.path_file_sampling(set_name)
    with open(p, 'r') as file:
        try:
            data = toml.load(file)
        except toml.decoder.TomlDecodeError as e:
            raise RuntimeError(f"Toml decode error was raised. Check that all entries in the sampling list "
                               f"in file {p} can be interpreted as floats, i.e., '1.0' instead of '1'.") from e
        data = np.array(data[C.key_sampling_wl])
        return data


def write_starting_guess_coeffs(ad_coeffs, sd_coeffs, ai_coeffs, mf_coeffs) -> None:
    """Writes given starting guess coefficients to disk.

    :param ad_coeffs:
        Coefficients for absorption particle density as a list of floats.
    :param sd_coeffs:
        Coefficients for scattering particle density as a list of floats.
    :param ai_coeffs:
        Coefficients for scattering anisotropy as a list of floats.
    :param mf_coeffs:
        Coefficients for mix factor as a list of floats.
    """

    path = P.path_file_default_starting_guess()
    coeff_dict = {C.ad_coeffs:ad_coeffs, C.sd_coeffs:sd_coeffs, C.ai_coeffs:ai_coeffs, C.mf_coeffs:mf_coeffs}
    with open(path, 'w+') as file:
        toml.dump(coeff_dict, file, encoder=toml.encoder.TomlNumpyEncoder())


def read_starting_guess_coeffs():
    """Reads starting guess coefficients from disk and return as dictionary.

    :return:
        Starting guess coefficients in a dictionary.
    """

    path = P.path_file_default_starting_guess()
    with open(path, 'r') as file:
        data = toml.load(file)
        return data


def make_sample_result(set_name:str, sample_id: int, wall_clock_time_min=0.0):
    """Creates the sample result by collecting the data from wavelength results.

    Saves the result as numerical data and plots.

    :param set_name:
        Set name.
    :param sample_id:
        Sample id.
    :param wall_clock_time_min:
        Wall clock time used to optimize this sample.
    """

    # Collect subresults
    wl_res_list = collect_wavelength_result(set_name, sample_id)
    sample_result_dict = {}

    # Set starting value to which earlier result time is added.
    sample_result_dict[C.key_sample_result_wall_clock_elapsed_min] = wall_clock_time_min

    # If we already have existing sample result, with sparser resolution, we'll want to take that
    # into account when saving the new result.
    try:
        previous_result = read_sample_result(set_name, sample_id)  # throws OSError upon failure
        this_result_time = sample_result_dict[C.key_sample_result_wall_clock_elapsed_min]
        previous_result_time = previous_result[C.key_sample_result_wall_clock_elapsed_min]
        sample_result_dict[C.key_sample_result_wall_clock_elapsed_min] = this_result_time + previous_result_time
    except OSError as e:
        pass  # there was no previous result so this is OK

    sample_result_dict[C.key_sample_result_process_elapsed_min] = np.sum(subres[C.key_wl_result_elapsed_time_s] for subres in wl_res_list) / 60.0
    sample_result_dict[C.key_sample_result_r_RMSE] = np.sqrt(np.mean(np.array([subres[C.key_wl_result_refl_error] for subres in wl_res_list]) ** 2))
    sample_result_dict[C.key_sample_result_t_RMSE] = np.sqrt(np.mean(np.array([subres[C.key_wl_result_tran_error] for subres in wl_res_list]) ** 2))
    sample_result_dict[C.key_wl_result_optimizer] = wl_res_list[0][C.key_wl_result_optimizer],
    sample_result_dict[C.key_wl_result_optimizer_ftol] = wl_res_list[0][C.key_wl_result_optimizer_ftol],
    sample_result_dict[C.key_wl_result_optimizer_xtol] = wl_res_list[0][C.key_wl_result_optimizer_xtol],
    sample_result_dict[C.key_wl_result_optimizer_diffstep] = wl_res_list[0][C.key_wl_result_optimizer_diffstep],
    if sample_result_dict[C.key_wl_result_optimizer][0] == 'basin_hopping':
        sample_result_dict['basin_iterations_required'] = sum([(subres[C.key_wl_result_optimizer_result]['nit'] > 1) for subres in wl_res_list])

    # Collect lists from subresults
    wls = np.array([subres[C.key_wl_result_wl] for subres in wl_res_list])
    r   = np.array([subres[C.key_wl_result_refl_modeled] for subres in wl_res_list])
    rm  = np.array([subres[C.key_wl_result_refl_measured] for subres in wl_res_list])
    re  = np.array([subres[C.key_wl_result_refl_error] for subres in wl_res_list])
    t   = np.array([subres[C.key_wl_result_tran_modeled] for subres in wl_res_list])
    tm  = np.array([subres[C.key_wl_result_tran_measured] for subres in wl_res_list])
    te  = np.array([subres[C.key_wl_result_tran_error] for subres in wl_res_list])
    ad  = np.array([subres[C.key_wl_result_history_ad][-1] for subres in wl_res_list])
    sd  = np.array([subres[C.key_wl_result_history_sd][-1] for subres in wl_res_list])
    sa  = np.array([subres[C.key_wl_result_history_ai][-1] for subres in wl_res_list])
    mf  = np.array([subres[C.key_wl_result_history_mf][-1] for subres in wl_res_list])

    # Sort lists by wavelength. This has to be done as the wavelength
    # results are read from files in no particular order.
    sorting_idx = wls.argsort()
    sorting_idx = np.flip(sorting_idx) # flip to get ascending order
    wls = wls[sorting_idx[::-1]]
    r  = r[sorting_idx[::-1]]
    rm = rm[sorting_idx[::-1]]
    re = re[sorting_idx[::-1]]
    t  = t[sorting_idx[::-1]]
    tm = tm[sorting_idx[::-1]]
    te = te[sorting_idx[::-1]]
    ad = ad[sorting_idx[::-1]]
    sd = sd[sorting_idx[::-1]]
    sa = sa[sorting_idx[::-1]]
    mf = mf[sorting_idx[::-1]]

    # Put sorted lists in the dict
    sample_result_dict[C.key_sample_result_wls] = wls
    sample_result_dict[C.key_sample_result_r] = r
    sample_result_dict[C.key_sample_result_rm] = rm
    sample_result_dict[C.key_sample_result_re] = re
    sample_result_dict[C.key_sample_result_t] = t
    sample_result_dict[C.key_sample_result_tm] = tm
    sample_result_dict[C.key_sample_result_te] = te
    sample_result_dict[C.key_sample_result_ad] = ad
    sample_result_dict[C.key_sample_result_sd] = sd
    sample_result_dict[C.key_sample_result_ai] = sa
    sample_result_dict[C.key_sample_result_mf] = mf

    write_sample_result(set_name, sample_result_dict, sample_id)
    plotter.plot_sample_result(set_name, sample_id, dont_show=True, save_thumbnail=True)
