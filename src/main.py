import logging

from src import constants as C
from src import data_utils as DU
from src import optimization


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    r_m = 0.4
    t_m = 0.4
    optimization.optimize_to_measured(r_m=r_m,t_m=t_m)

    r = DU.get_relative_refl_or_tran(C.imaging_type_refl, 0)
    t = DU.get_relative_refl_or_tran(C.imaging_type_tran, 0)
    print(f"Final reflectance: {r} ({r_m})")
    print(f"Final transmittance: {t} ({t_m})")
