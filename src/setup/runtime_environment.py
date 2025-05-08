"""

This script contains similar data to the :mod:`src.constants` module except
that rather than being actual hard-coded constants, they are inferred during
initialization of HyperBlend.

"""


class RuntimeEnvironment:
    """Class to hold runtime environment variables for HyperBlend.

    This class is used to store the runtime environment variables for HyperBlend.
    The values are set during initialization and should not be changed afterwards,
    which is why only getters are exposed outwards. The class is supposed to be
    populated by the :mod:`setup.initialization` module.
    """

    def __init__(self):
        self._HB_VERSION = None
        self._SUPPORTED_BLENDER_VERSIONS = []
        self._OS = None
        self._BLENDER_EXECUTABLE = None

    @property
    def hyperblend_version(self):
        return self._HB_VERSION

    @property
    def supported_blender_versions(self):
        return self._SUPPORTED_BLENDER_VERSIONS

    @property
    def operating_system_string(self):
        return self._OS

    @property
    def blender_executable_path(self):
        return self._BLENDER_EXECUTABLE
