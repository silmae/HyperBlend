"""
Check directory structure and create missing directories.

"""

import os
import logging

from src.data import path_handling as PH
from src.data import toml_handling as TH
from src import constants as C

def initialize():
    logging.info("Initializing HyperBlend")

    check_directory_structure()


def check_directory_structure():
    """Check the directory structure of the project."""

    logging.info("Checking directory structure")

    def_dir = PH.path_directory_definitions()
    try:
        struct_toml = TH.read_toml_as_dict(directory=def_dir, filename=C.file_directory_structure)
    except FileNotFoundError as e:
        logging.error(f"File '{def_dir}/{C.file_directory_structure}' not found. This is an internal "
              f"file that defines the directory structure and some core files that must be "
              f"present for HyperBlend to operate. This file should be included in the installation. "
              f"There is no way to recover from this error. Please re-install the software.")
        exit(1)

    for key, value in struct_toml.items():
        if key == "root":
            _process_dir_struct_sub_entry(value, dir_list=[])


def _process_dir_struct_sub_entry(sub_dict: dict, dir_list):
    """Process a subdictionary of the directory structure definition file.

    This function is called recursively to process all subdirectories.
    It checks if the directory exists and if it contains the expected files.
    If the directory does not exist, it creates it.
    If the expected files are not present, it throws an error.

    :param sub_dict: The subdictionary to process.
    :param dir_list: The list of directories to build the path.
    :return: None
    :raises FileNotFoundError: If one of the expected files is not present. This is unrecoverable error.
    """

    path_builder = dir_list
    entry_name = ""
    entry_type = ""
    expected_file_names = None

    # Loop through all entries in the subdictnionary that was passed
    for key, value in sub_dict.items():
        # print(f"Key in read: {key}, Value: {value}")

        # Store the name of current entry. This is usually the name of the current directory
        if key == "name":
            entry_name = value

        # Store the type of current entry
        if key == "type":
            entry_type = value
        # If this is a directory type there might be some files that are expected to be in the directory
        elif key == "expected_file_names":
            # print(f"Expected file names: {value}")
            expected_file_names = value

        # If the entry is a dictionary (meaning, it is a subdirectory) we need to process it recursively
        if isinstance(value, dict) and value["type"] == "dir":

            # Take the name of the sudictionary and append it to the path builder
            sub_entry_name = value["name"]
            path_builder.append(sub_entry_name)

            # Process the subdirectory recursively
            _process_dir_struct_sub_entry(sub_dict=value, dir_list=path_builder)

            # When we come out, delete the last entry in the path builder to get back to the previous level
            del path_builder[-1]

    # Now all entries are looped through and we can process their contents
    if entry_type == "dir":

        # First, check where we are in the directory structure starting from the project root
        current_path = PH.path_directory_project_root()
        if entry_name != "Root":
            # If we are not already at the root, append all entries in the path builder to the current path
            for dir_name in path_builder:
                current_path = PH.join(current_path, dir_name)

        if not os.path.exists(current_path):
            logging.info(f"Directory '{current_path}' does not exist. Creating directory.")
            os.makedirs(current_path, exist_ok=True)
        else:
            logging.info(f"Directory '{current_path}' exists as it should.")

        if expected_file_names is not None:
            for file_name in expected_file_names:
                file_path = PH.join(current_path, file_name)
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"File '{file_name}' does not exist in '{current_path}'.")
                else:
                    logging.info(f"\tFile '{file_name}' exists as it should.")
