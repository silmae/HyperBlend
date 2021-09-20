"""
Methods in this file handle writing and reading of various toml formatted txt files.
"""

import toml
import os

import numpy as np

from src import file_handling as FH
from src import constants as C


def read_final_result(set_name: str):
    """Reads final result file into a dict and returns it.

    :param set_name:
        Set name.
    :return:
        Result file content as dict.
    """

    p = os.path.normpath(FH.get_path_opt_result(set_name) + '/' + 'final_result' + C.postfix_text_data_format)
    with open(p, 'r') as file:
        subres_dict = toml.load(file)

    return subres_dict


def write_final_result(set_name: str, res_dict: dict):
    """Writes final result dictionary into a file.

    :param set_name:
        Set name.
    :param res_dict:
        Dictionary to be written.
    :return:
        None
    """
    p = os.path.normpath(FH.get_path_opt_result(set_name) + '/' + 'final_result' + C.postfix_text_data_format)
    with open(p, 'w+') as file:
        toml.dump(res_dict, file, encoder=toml.encoder.TomlNumpyEncoder())


def collect_subresults(set_name: str):
    """Collects subresult dictionaries in to a list and returns it.

    :param set_name:
        Set name.
    :return:
        A list of subresult dictionaries.
    """

    p = FH.get_path_opt_subresult(set_name)
    subres_list = []
    for filename in os.listdir(p):
        if filename.endswith(C.postfix_text_data_format):
            subres = toml.load(os.path.join(p, filename))
            subres_list.append(subres)
            # print(filename)
            # print(subres)
    return subres_list


def write_subresult(set_name: str, res_dict: dict):
    """Writes subresult of optimization of a single wavelength into a file.

    :param set_name:
        Set name.
    :param res_dict:
        Subresult dictionary.
    :return:
        None
    """

    wl = res_dict[C.subres_key_wl]
    p = FH.get_path_opt_subresult(set_name) + '/' + f"subres_wl_{wl:.2f}" + C.postfix_text_data_format
    with open(p, 'w+') as file:
        toml.dump(res_dict, file, encoder=toml.encoder.TomlNumpyEncoder())


def read_subresult(set_name: str, wl: float):
    """Reads a subresult file into a dictionary and returns it.

    :param set_name:
        Set name.
    :param wl:
        Wavelength of the subresult.
    :return:
        Subresult as a dictionary.
    """
    p = FH.get_path_opt_subresult(set_name) + '/' + f"subres_wl_{wl:.2f}" + C.postfix_text_data_format
    with open(p, 'r') as file:
        subres_dict = toml.load(file)

    return subres_dict


def write_target(set_name:str, data):
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
    with open(FH.get_path_opt_target_file(set_name), 'w+') as file:
        toml.dump(res, file)


def read_target(set_name: str):
    """Read target values for optimization.

    :param set_name:
        Name of the set.
    :return:
        List of reflectances and transmittances per wavelength [[wl, r, t],...] as numpy array
    """

    with open(FH.get_path_opt_target_file(set_name), 'r') as file:
        data = toml.load(file)
        data = data['wlrt']
        data = np.array(data)
        return data
