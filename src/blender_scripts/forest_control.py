"""
This script is responsible for writing and reading forest control toml files.

This functionality would be better plced under data.toml_handling, but due to
Blender using different Python environment than the rest of the code, it would
create problems with dependencies.

"""

import os
import toml

from src.data import path_handling as PH
from src import constants as C


def write_forest_control(forest_id: str, control_dict: dict, global_master: bool = False):
    """Writes forest control file.

    :param forest_id:
        Forest id for which the control file is written to.
    :param control_dict:
        Dictionary to be written.
    :param global_master:
        If True, global master file is written to project root. Needs to be done if there are
        changes made to the forest template file. This will be kept safe in the Git repository.
        Default is False.
    """

    if global_master:
        write_dict_as_toml(dictionary=control_dict, directory=PH.path_directory_project_root(), filename=C.file_forest_control)
    else:
        write_dict_as_toml(dictionary=control_dict, directory=PH.path_directory_forest_scene(forest_id=forest_id),
                           filename=C.file_forest_control)


def read_forest_control(forest_id: str) -> dict:
    return read_toml_as_dict(directory=PH.path_directory_forest_scene(forest_id=forest_id), filename=C.file_forest_control)


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

    if not filename.endswith(C.postfix_text_data_format):
        filename = filename + C.postfix_text_data_format

    p = PH.join(directory, filename)
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

    if not filename.endswith(C.postfix_text_data_format):
        filename = filename + C.postfix_text_data_format

    p = PH.join(directory, filename)

    if not os.path.exists(os.path.abspath(p)):
        raise RuntimeError(f"Cannot read from file '{os.path.abspath(p)}' "
                           f"because it does not exist.")

    with open(p, 'r') as file:
        result = toml.load(file)
    return result
