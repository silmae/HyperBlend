import logging
import os

from src import constants as C
from src.data import path_handling as PH, toml_handling as TH
from src.setup.runtime_environment import RuntimeEnvironment

RECOGNIZED_KEYS = ["name", "type", "extensions"]


def check_directory_structure(runtime: RuntimeEnvironment):
    """Check the directory structure of the project."""

    logging.info("Checking directory structure")

    def_dir = PH.path_directory_definitions()
    filename = C.file_directory_structure
    try:
        struct_toml = TH.read_toml_as_dict(directory=def_dir, filename=filename)
    except FileNotFoundError as e:
        logging.error(f"File '{def_dir}/{filename}' not found. This is an internal "
              f"file that defines the directory structure and some core files that must be "
              f"present for HyperBlend to operate. This file should be included in the installation. "
              f"There is no way to recover from this error. Please re-install the software.")
        exit(1)

    for key, value in struct_toml.items():
        if key == "root":
            _process_dir_struct_sub_entry(value, dir_list=[])
        elif key == "version":
            # Check if the version of the directory structure matches the version of HyperBlend
            if value != runtime._HB_VERSION:
                logging.error(f"Version of the directory structure '{value}' does not match "
                      f"the version of HyperBlend '{runtime._HB_VERSION}'. Please re-install "
                      f"HyperBlend.")
                exit(1)


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
    file_extensions = []

    # Loop through all entries in the subdictnionary that was passed
    for key, value in sub_dict.items():

        if key in RECOGNIZED_KEYS:
            # Store the name of current entry. This is usually the name of the current directory
            if key == "name":
                entry_name = value
            # Store the type of current entry
            elif key == "type":
                entry_type = value
            # Store possible filename extensions
            elif key == "extensions":
                file_extensions = value

        # If the entry is a dictionary (meaning, it is a subdirectory) we need to process it recursively
        if isinstance(value, dict):

            # Take the name of the sudictionary and append it to the path builder
            sub_entry_name = value["name"]
            path_builder.append(sub_entry_name)

            # Process the subdirectory recursively
            _process_dir_struct_sub_entry(sub_dict=value, dir_list=path_builder)

            # When we come out, delete the last entry in the path builder to get back to the previous level
            del path_builder[-1]

    # Now all entries are looped through and we can process their contents

    # First, check where we are in the directory structure starting from the project root
    # If we are not already at the root, append all entries in the path builder to the current path
    current_path = PH.path_directory_project_root()
    if entry_name != "Root":
        # Ignore the last part of the path builder if we are dealing with a file
        if entry_type == "file":
            for dir_name in path_builder[:-1]:
                current_path = PH.join(current_path, dir_name)
        else:
            for dir_name in path_builder:
                current_path = PH.join(current_path, dir_name)

    if entry_type == "dir":
        if not os.path.exists(current_path):
            logging.info(f"Directory '{current_path}' does not exist. Creating directory.")
            os.makedirs(current_path, exist_ok=True)
        else:
            logging.info(f"OK - Directory '{current_path}' exists as it should.")

    elif entry_type == "file":

        file_found = False
        with_extension = entry_name

        for extension in file_extensions:
            # Check if the file exists with the given extension
            with_extension = entry_name + '.' + extension
            file_path = PH.join(current_path, with_extension)
            if os.path.exists(file_path):
                file_found = True
                break

        if not file_found:
            raise FileNotFoundError(f"File '{with_extension}' does not exist in '{current_path}'.")
        else:
            logging.info(f"OK - File '{with_extension}' exists in '{current_path}' as it should.")


