"""
Methods in this file handle writing and reading of various toml formatted txt files.
"""

import toml
import os

import numpy as np

from src.data import file_handling as FH
from src.data import file_names as FN
from src import constants as C


def write_final_result(set_name: str):
    """Collect sample results and write final result to a toml file. """

    result_dict = {}
    r = collect_sample_results(set_name)
    sample_count = len(r)
    result_dict['sample_count'] = sample_count
    result_dict['total_time_hours'] = np.sum([sr[C.result_key_wall_clock_elapsed_min] for sr in r]) / 60
    result_dict['time_per_sample_hours'] = np.sum([sr[C.result_key_wall_clock_elapsed_min] for sr in r]) / 60 / sample_count
    result_dict['total_processor_time_hours'] = np.sum([sr[C.result_key_process_elapsed_min] for sr in r]) / 60
    result_dict['processor_time_per_sample_hours'] = np.sum([sr[C.result_key_process_elapsed_min] for sr in r]) / 60 / sample_count
    result_dict['refl_error_mean'] = np.mean([sr[C.result_key_refls_error] for sr in r])
    result_dict['tran_error_mean'] = np.mean([sr[C.result_key_trans_error] for sr in r])
    result_dict['refl_error_std']= np.std([sr[C.result_key_refls_error] for sr in r])
    result_dict['tran_error_std']= np.std([sr[C.result_key_trans_error] for sr in r])
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


def write_sample_result(set_name: str, res_dict: dict, sample_id: int):
    """Writes sample result dictionary into a file."""

    p = FH.join(FH.path_directory_sample(set_name, sample_id), FN.filename_sample_result(sample_id))
    with open(p, 'w+') as file:
        toml.dump(res_dict, file, encoder=toml.encoder.TomlNumpyEncoder())


def collect_subresults(set_name: str, sample_id):
    """Collects subresult dictionaries in to a list and returns it.

    :param sample_id:
    :param set_name:
        Set name.
    :return:
        A list of subresult dictionaries.
    """

    p = FH.path_directory_subresult(set_name, sample_id)
    subres_list = []
    for filename in os.listdir(p):
        if filename.endswith(C.postfix_text_data_format):
            subres = toml.load(FH.join(p, filename))
            subres_list.append(subres)
    return subres_list


def write_subresult(set_name: str, res_dict: dict, sample_id):
    """Writes subresult of optimization of a single wavelength into a file.

    :param sample_id:
    :param set_name:
        Set name.
    :param res_dict:
        Subresult dictionary.
    :return:
        None
    """

    wl = res_dict[C.subres_key_wl]
    p = FH.path_file_subresult(set_name, wl, sample_id)
    with open(p, 'w+') as file:
        toml.dump(res_dict, file, encoder=toml.encoder.TomlNumpyEncoder())


def read_subresult(set_name: str, wl: float, sample_id):
    """Reads a subresult file into a dictionary and returns it.

    :param sample_id:
    :param set_name:
        Set name.
    :param wl:
        Wavelength of the subresult.
    :return:
        Subresult as a dictionary.
    """

    p = FH.path_file_subresult(set_name, wl, sample_id)
    with open(p, 'r') as file:
        subres_dict = toml.load(file)

    return subres_dict


def write_target(set_name:str, data, sample_id=0):
    """Writes given list of reflectance and transmittance data to toml formatted file.

    :param set_name:
        Name of the set.
    :param data:
        List of lists or list of tuples [[wl, r, t], ...]
        Do not use numpy arrays as they may break the toml writer.
    :return:
        None
    """
    floated_list = [[float(a), float(b), float(c)] for (a, b, c) in data]
    res = {'wlrt': floated_list}
    with open(FH.path_file_target(set_name, sample_id), 'w+') as file:
        toml.dump(res, file)


def read_target(set_name: str, sample_id: int):
    """Read target values for optimization.

    :param sample_id:
    :param set_name:
        Name of the set.
    :return:
        List of reflectances and transmittances per wavelength [[wl, r, t],...] as numpy array
    """

    with open(FH.path_file_target(set_name, sample_id), 'r') as file:
        data = toml.load(file)
        data = data['wlrt']
        data = np.array(data)
        return data


def write_starting_guess_coeffs(ad_coeffs, sd_coeffs, ai_coeffs, mf_coeffs):
    """Writes given starting guess coefficients to disk.

    :param ad_coeffs:
    :param sd_coeffs:
    :param ai_coeffs:
    :param mf_coeffs:
    :return:
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
