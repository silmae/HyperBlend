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
from src import constants as C


def write_final_result(set_name: str):
    """Collect sample results and write final result to a toml file. """

    result_dict = {}
    r = collect_sample_results(set_name)
    sample_count = len(r)
    result_dict[C.key_set_result_sample_count] = sample_count
    result_dict[C.key_set_result_total_time_hours] = np.sum([sr[C.key_sample_result_wall_clock_elapsed_min] for sr in r]) / 60
    result_dict[C.key_set_result_time_per_sample_hours] = np.sum([sr[C.key_sample_result_wall_clock_elapsed_min] for sr in r]) / 60 / sample_count
    result_dict[C.key_set_result_total_processor_time_hours] = np.sum([sr[C.key_sample_result_process_elapsed_min] for sr in r]) / 60
    result_dict[C.key_set_result_processor_time_per_sample_hours] = np.sum([sr[C.key_sample_result_process_elapsed_min] for sr in r]) / 60 / sample_count
    result_dict[C.key_set_result_refl_error_mean] = np.mean([sr[C.key_sample_result_re] for sr in r])
    result_dict[C.key_set_result_tran_error_mean] = np.mean([sr[C.key_sample_result_te] for sr in r])
    result_dict[C.key_set_result_refl_error_std]= np.std([sr[C.key_sample_result_re] for sr in r])
    result_dict[C.key_set_result_tran_error_std]= np.std([sr[C.key_sample_result_te] for sr in r])
    p = FH.join(FH.path_directory_set_result(set_name), FN.filename_final_result())
    with open(p, 'w+') as file:
        toml.dump(result_dict, file, encoder=toml.encoder.TomlNumpyEncoder())


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

    p = FH.join(FH.path_directory_sample(set_name, sample_id), FN.filename_sample_result(sample_id))
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

    p = FH.join(FH.path_directory_sample(set_name, sample_id), FN.filename_sample_result(sample_id))
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

    p = FH.path_directory_subresult(set_name, sample_id)
    subres_list = []
    for filename in os.listdir(p):
        if filename.endswith(C.postfix_text_data_format):
            subres = toml.load(FH.join(p, filename))
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
    p = FH.path_file_subresult(set_name, wl, sample_id)
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

    p = FH.path_file_subresult(set_name, wl, sample_id)
    with open(p, 'r') as file:
        subres_dict = toml.load(file)

    return subres_dict


def write_target(set_name:str, data, sample_id=0) -> None:
    """Writes given list of reflectance and transmittance data to toml formatted file.

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
    with open(FH.path_file_target(set_name, sample_id), 'w+') as file:
        toml.dump(res, file)


def read_target(set_name: str, sample_id: int):
    """Read target values for optimization.

    :param set_name:
        Name of the set.
    :param sample_id:
        Sample id.
    :return:
        List of reflectances and transmittances per wavelength [[wl, r, t],...] as numpy array
    """

    with open(FH.path_file_target(set_name, sample_id), 'r') as file:
        data = toml.load(file)
        data = data['wlrt']
        data = np.array(data)
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

    path = FH.path_file_default_starting_guess()
    coeff_dict = {C.ad_coeffs:ad_coeffs, C.sd_coeffs:sd_coeffs, C.ai_coeffs:ai_coeffs, C.mf_coeffs:mf_coeffs}
    with open(path, 'w+') as file:
        toml.dump(coeff_dict, file, encoder=toml.encoder.TomlNumpyEncoder())


def read_starting_guess_coeffs():
    """Reads starting guess coefficients from disk and return as dictionary.

    :return:
        Starting guess coefficients in a dictionary.
    """

    path = FH.path_file_default_starting_guess()
    with open(path, 'r') as file:
        data = toml.load(file)
        return data
