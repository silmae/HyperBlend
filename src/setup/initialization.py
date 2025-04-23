"""
Check directory structure and create missing directories.

"""

import logging

from src.setup import system_check as SC
from src.setup.blender_check import check_blender_version
from src.setup.directory_check import check_directory_structure
from src.data import path_handling as PH, toml_handling as TH
from src import constants as C
from src.setup.runtime_environment import RuntimeEnvironment


def initialize():
    logging.info("Initializing HyperBlend")
    runtime = RuntimeEnvironment()
    runtime = _load_app_info(runtime=runtime)
    check_directory_structure(runtime=runtime)
    runtime = SC.gather_system_info(runtime=runtime)
    # TODO find path to Blender executable
    runtime = check_blender_version(runtime=runtime)

    return runtime


def _load_app_info(runtime: RuntimeEnvironment):

    logging.info("Loading app info")

    def_dir = PH.path_directory_definitions()
    filename = C.file_app_info
    try:
        app_info_dict = TH.read_toml_as_dict(directory=def_dir, filename=filename)
    except FileNotFoundError as e:
        logging.error(f"File '{def_dir}/{filename}' not found. This is an internal "
            f"file that defines contains HyperBlend's version data and other information "
            f"needed for checking compatibility with other modules.")
        exit(1)

    for key, value in app_info_dict.items():
        if key == "app_version":
            runtime._HB_VERSION = value
        elif key == "supported_blender_versions":
            runtime._SUPPORTED_BLENDER_VERSIONS = value

    return runtime


