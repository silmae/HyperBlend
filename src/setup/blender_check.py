import logging
import os

import numpy as np

from src import constants as C
from src.setup.runtime_environment import RuntimeEnvironment


def check_blender_version(runtime: RuntimeEnvironment):

    logging.info("Checking Blender version")

    found_versions = []

    operating_system = runtime._OS
    supported_blender_versions = runtime._SUPPORTED_BLENDER_VERSIONS

    path_foundation = C.blender_foundation_win

    if operating_system == "linux":
        # TODO linux has only one installation at a time? Check it and do something with the information
        raise NotImplementedError("Linux is not supported yet.")
    elif operating_system == "windows":
        # TODO check available versions
        if not os.path.exists(path_foundation):
            logging.error(f"It seems that there is no Blender installed to the default "
                          f"path in {path_foundation}. Install Blender or change the path in "
                          f"'constants.py' file.")
            exit(1)

        logging.info(f"Searching for Blender versions from {path_foundation}")
        for x in os.listdir(path_foundation):
            splitted = x.split(" ")
            version = splitted[1]
            logging.debug(f"Found {x}. Parsed version number: {version}")
            found_versions.append(version)
    elif not operating_system:
        logging.error("Operating system is not recognized.")
        exit(1)

    logging.info(f"Checking found Blender versions against supported versions.")
    logging.info(f"Found versions: {found_versions}")
    logging.info(f"Supported versions: {supported_blender_versions}")

    res = np.array(list(i in supported_blender_versions for i in found_versions))

    if not np.any(res):
        logging.error("Found Blender versions are not supported. Please install one of the "
                      f"supported versions: {supported_blender_versions}")
        exit(1)

    # It's a tuple so take the newest version
    i = np.where(res)[0][-1]
    blender_ex_path = f"Blender {found_versions[i]}\\blender.exe"
    full_blender_ex_path = os.path.join(path_foundation, blender_ex_path)
    runtime._BLENDER_EXECUTABLE = full_blender_ex_path
    logging.info(f"Set Blender executable to: {full_blender_ex_path}")

    return runtime
