"""

This module is an access point to load light spectra to be used in system level simulation.

If you are happy with the default light spectra, you can simply call :func:`load_light()`
and be done with it. If you want to use one of the integrated spectra generation tools,
follow the instructions below.

How to generate sun and sky spectra with SSolar GOA
-------------------------------------------------------

    1. Download Windows version of GOA from
       `goa.uva.es/ssolar_goa-model/ <https://goa.uva.es/ssolar_goa-model/>`_
       or use their GitHub version from
       `github.com/GOA-UVa/SSolar-GOA <https://github.com/GOA-UVa/SSolar-GOA>`_.
    2. (Assuming the Windows version) Click on "Load default settings"
    3. Change wavelength range to 400-2500 nm
    4. Click on "Run"
    5. Move the generated file to where HyperBLend's light files are stored and
       call :func:`load_light()` once. It will the split the sun and sky
       spectra to separate files that can be read by calling the :func:`load_light()` again.

How to generate sun spectra with NASA PSG
---------------------------------------------

    1. Select template "Sun from Earth" and click "Load Spectra".
    2. If you want to select certain location and date on Earth, click
       "Change Object" button. Remember to click save settings!
    3. Click "Change Instrument"

       1. Set "Spectral range" to 400 - 2500 nm. Change units
          from :math:`\mu m` to nm.
       2. Set "Resolution" to 1 nm and units from "Resolving power" to nm.
       3. Set "Spectrum intensity unit" to W/m2/um (spectral irradiance)
    4. Change other settings as you see fit, but know what you are doing!
    5. Click "Generate Spectra"
    6. From the first image, click "Download Spectra" and put the file as in
       GOA instructions above.


How to use fully custom light spectra
---------------------------------------------

If you have a spectra that your grandpa recorded and wrote down on a piece of paper,
you can use it as a light source in HyperBLend. Just write the values to a text file.
The only requirement is that the file
contains wavelength-irradiance pairs in two columns, separated by a space. You can
add comments to the file by starting the line with a hash sign (#). All comments must
occur before the actual data.

"""

import numpy as np
import logging

from src import constants as C
from src.data import light_file_handling as LFH, path_handling as PH
from src.utils import spectra_utils as SU


def load_light(
    file_name: str = None,
    system_sim_name: str = None,
    sampling=None,
    lighting_type="sun",
):
    """Loads a lighting file and returns wavelengths and corresponding irradiances.

    Files formatted so that comment lines are prefixed with '#' and rest of the lines
    contain wavelength-irradiance pairs that can be cast to floats can be read
    directly. For spectra generated with either NASA PSG or SSolar GOA, the files are
    fixed so that they can be read directly later.

    :param file_name: Passed directly to :func:`data.path_handling.find_light_file`.
    :param system_sim_name: Passed directly to :func:`data.path_handling.find_light_file`.
    :param sampling: List of wavelengths as floats. If given, sun data is resampled to wavelengths specified
        in the list. If sampling is None (default), the data is returned as raw.
    :param lighting_type: Lighting type either "sun" or "sky", default is "sun". This is used to get the default
        sun or sky file if ``file_name`` was not given. If both are given, the ``file_name`` has precedence.

    :return: Tuple (np.array[wls], np.array[irradiances]) where wls is a list of wavelengths and irradiances are
        corresponding list of irradiances. The length of the lists vary depending on given sampling.

    :raises: ValueError if ``file_name`` was not provided and ``lighting_type`` is
        not either "sun" or "sky".
    """

    logging.info("Loading light data.")

    if file_name is None:
        if lighting_type == "sun":
            file_name = C.file_default_sun
        elif lighting_type == "sky":
            file_name = C.file_default_sky
        else:
            raise ValueError(
                f"Light file name was not provided. For loading one of the default files, "
                f"expected file type either 'sun' or 'sky', was '{lighting_type}'."
            )

    if not file_name.endswith(".txt"):
        file_name = file_name + ".txt"

    path = PH.find_light_file(file_name, system_sim_name)

    wls, irradiances = LFH.read_light_file(path)

    if sampling is not None:
        new_irradiances = SU.resample(
            original_wl=wls, original_val=irradiances, new_wl=sampling
        )
        wls = sampling
        irradiances = new_irradiances

    logging.info("Light data loaded.")

    return np.array(wls), np.array(irradiances)
