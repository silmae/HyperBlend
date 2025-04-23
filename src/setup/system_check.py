"""
This script checks the system requirements for HyperBlend.
"""

from sys import platform
import logging

from src.setup.runtime_environment import RuntimeEnvironment

def gather_system_info(runtime: RuntimeEnvironment):

    os = operating_system()
    if os == "os x":
        raise NotImplementedError("OS X is not supported yet.")
    else:
        runtime._OS = os

    logging.info(f"Operating system: {os}")
    return runtime


def operating_system():
    """
    Check the operating system.
    """

    if platform == "linux":
        return "linux"
    elif platform == "darwin":
        return "os x"
    elif platform == "win32":
        return "windows"


