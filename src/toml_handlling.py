import toml

import numpy as np

from src import file_handling as FH
from src import constants as C


def write_target(set_name:str, wls):
    """Writes given list to file as toml.

    Parameters
    ----------
        set_name
            Name of the set.
        wls
            List of lists or list of tuples [[wl, r, t], ...]
            Do not use numpy arrays.
    """
    floated_list = [[float(a), float(b), float(c)] for (a, b, c) in wls]
    res = {'wlrt': floated_list}
    with open(FH.get_path_opt_target_file(set_name), 'w+') as file:
        toml.dump(res, file)


def read_target(set_name: str, select_n=None):
    """Read target values for optimization.

    Parameters
    ----------
        set_name
            Name of the set.
        select_n
            Select n values linearly distributed. Return all values if None.

    Returns
    -------
        list
            list of reflectances and transmittances per wavelength [[wl, r, t],...]
    """

    with open(FH.get_path_opt_target_file(set_name), 'r') as file:
        wls = toml.load(file)
        wls = wls['wlrt']

    if select_n is None:
        return wls
    else:
        max_idx = len(wls)
        selector = np.linspace(0, max_idx, select_n)
        selected = wls[selector]
        return selected
