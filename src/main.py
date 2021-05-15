import logging

from src import constants as C
from src import data_utils as DU
from src import optimization
from src import file_handling as FH
from src import toml_handlling as T
from src import plotter


if __name__ == '__main__':

    test_set_name = 'set_ABC'
    plotter.plot_subresult_opt_history(set_name=test_set_name, wl=15.)
    # FH.create_opt_folder_structure(test_set_name)
    # wls1 = [
    #     (10, 0.21, 0.17),
    #     (15, 0.11, 0.161),
    # ]
    # T.write_target(test_set_name, wls1)
    # print(T.read_target(test_set_name, None))

    # logging.basicConfig(level=logging.INFO)
    # r_m = 0.4
    # t_m = 0.4
    # optimization.run_optimization(test_set_name)
    #
    # r = DU.get_relative_refl_or_tran(C.imaging_type_refl, 0)
    # t = DU.get_relative_refl_or_tran(C.imaging_type_tran, 0)
    # print(f"Final reflectance: {r} ({r_m})")
    # print(f"Final transmittance: {t} ({t_m})")
