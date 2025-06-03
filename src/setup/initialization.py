"""

This module contains the initialization functions for HyperBlend.

Running the initialization populates :mod:`definitions.runtime_environment`. After this, the values
in there are not supposed to be changed.

"""

import logging
import os
import datetime
import sys
from sys import platform

import numpy as np

from src.setup.directory_check import check_directory_structure
from src.data import path_handling as PH, toml_handling as TH
from src import constants as C
from src.setup.runtime_environment import RuntimeEnvironment


def initialize():
    """Initializes HyperBlend.

    Directory structure is checked and missing directories created as necessary.
    Dynamically checks operating system and found Blender versions.
    """

    runtime = RuntimeEnvironment()
    _init_logging()
    _load_app_info(runtime)
    check_directory_structure(runtime)
    _check_operating_system(runtime)
    _check_blender_version(runtime)
    logging.info("Initialization complete")
    return runtime


def _init_logging():

    path_dir_logs = PH.directory_log()
    if not os.path.exists(path_dir_logs):
        os.makedirs(path_dir_logs)

    log_identifier = str(datetime.datetime.now())
    log_identifier = log_identifier.replace(" ", "_")
    log_identifier = log_identifier.replace(":", "")
    log_identifier = log_identifier.replace(".", "")

    log_file_name = f"{log_identifier}.log"
    log_path = PH.join(path_dir_logs, log_file_name)
    logging.basicConfig(
        level="INFO",
        format="%(asctime)s %(levelname)s: %(message)s",
        force=True,
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logging.info("Logging initialized")


def _load_app_info(runtime: RuntimeEnvironment):
    """Load application information from the app info from :mod:`definitions.app_info`.

    This function reads the application information file, which contains metadata about HyperBlend,
    such as the application version and supported Blender versions. The data is then stored in the
    runtime environment for later use.

    :raises FileNotFoundError: If the app info file is not found. This is an unrecoverable error.
    """

    logging.info("Loading app info")

    def_dir = PH.directory_code_definitions()
    filename = C.file_app_info
    try:
        app_info_dict = TH.read_toml_as_dict(directory=def_dir, filename=filename)
    except FileNotFoundError as e:
        logging.error(
            f"File '{def_dir}/{filename}' not found. This is an internal "
            f"file that defines contains HyperBlend's version data and other information "
            f"needed for checking compatibility with other modules."
        )
        exit(1)

    for key, value in app_info_dict.items():
        if key == "app_version":
            runtime._HB_VERSION = value
        elif key == "supported_blender_versions":
            runtime._SUPPORTED_BLENDER_VERSIONS = value

    logging.info(
        f"App info loaded. Running HyperBlend version: {runtime.hyperblend_version}"
    )


def _check_operating_system(runtime: RuntimeEnvironment):
    """Check the operating system and set the corresponding variable in the runtime environment.

    .. warning:: If the operating system is not recognized (Windows or Linux), the program will exit.

    :raises NotImplementedError: If the operating system is not supported, i.e., Mac OS.
    """

    logging.info("Checking operating system")

    if platform == "linux":
        runtime._OS = platform
    elif platform == "darwin":
        raise NotImplementedError("OS X is not supported.")
    elif platform == "win32":
        runtime._OS = platform
    else:
        logging.error(f"Unknown operating system: {platform}. Cannot continue.")
        exit(1)

    logging.info(f"Operating system: {runtime.operating_system_string}")


def _check_blender_version(runtime: RuntimeEnvironment):
    """Check the installed Blender version and set the corresponding variable in the runtime environment.

    On a Windows machine, the latest supported version is selected. Supported versions are listed in
    :mod:`definitions.app_info`.
    """

    logging.info("Checking Blender version")

    found_versions = []

    operating_system = runtime.operating_system_string
    supported_blender_versions = runtime.supported_blender_versions

    path_foundation = C.blender_foundation_win

    if operating_system == "linux":
        # TODO linux has only one installation at a time? Check it and do something with the information
        raise NotImplementedError("Linux is not supported yet.")
    elif operating_system == "win32":
        if not os.path.exists(path_foundation):
            logging.error(
                f"It seems that there is no Blender installed to the default "
                f"path in {path_foundation}. Install Blender or change the path in "
                f"'constants.py' file."
            )
            exit(1)

        logging.debug(f"Searching for Blender versions from {path_foundation}")
        # check available versions
        for x in os.listdir(path_foundation):
            splitted = x.split(" ")
            version = splitted[1]
            logging.debug(f"Found {x}. Parsed version number: {version}")
            found_versions.append(version)
    elif not operating_system:
        logging.error("Operating system is not recognized.")
        exit(1)

    res = np.array(list(i in supported_blender_versions for i in found_versions))

    if not np.any(res):
        logging.error(
            f"Found Blender versions {found_versions} are not supported. Please install one of the "
            f"supported versions: {supported_blender_versions}"
        )
        exit(1)

    # It's a tuple so take the newest version
    i = np.where(res)[0][-1]
    selected_version = found_versions[i]
    blender_ex_path = f"Blender {selected_version}\\blender.exe"
    full_blender_ex_path = os.path.join(path_foundation, blender_ex_path)
    runtime._BLENDER_EXECUTABLE = full_blender_ex_path

    logging.info(
        f"Autoselected Blender version {selected_version} from installed versions: {found_versions}, "
        f"which is the newest of the supported versions {supported_blender_versions}."
    )
    logging.info(f"Set Blender executable to: {full_blender_ex_path}")
