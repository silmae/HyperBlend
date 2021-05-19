import logging

from src import constants as C
from src import data_utils as DU
from src import optimization
from src import file_handling as FH
from src import toml_handlling as T
from src import plotter


if __name__ == '__main__':
    logging.basicConfig(level='INFO')

    test_set_name = 'set_batched'
    # test_set_name = 'set_threaded'
    # test_set_name = 'set_ABC'
    # -------------
    # Uncomment to re-create plots and save them. The wl must exist in subresults (toml).
    # plotter.plot_subresult_opt_history(set_name=test_set_name, wl=10., save_thumbnail=True)
    # -------------
    FH.create_opt_folder_structure(test_set_name)
    wls1 = [
        (10, 0.10, 0.10),
        (20, 0.20, 0.20),
        (30, 0.30, 0.30),
        (40, 0.40, 0.40),
        (50, 0.30, 0.30),
        (60, 0.20, 0.20),
    ]
    T.write_target(test_set_name, wls1)
    # print(T.read_target(test_set_name, None))

    # logging.basicConfig(level=logging.INFO)
    # r_m = 0.4
    # t_m = 0.4
    optimization.run_optimization_in_batches(test_set_name, batch_n=1)
    #
    # r = DU.get_relative_refl_or_tran(C.imaging_type_refl, 0)
    # t = DU.get_relative_refl_or_tran(C.imaging_type_tran, 0)
    # print(f"Final reflectance: {r} ({r_m})")
    # print(f"Final transmittance: {t} ({t_m})")


    # read = T.read_final_result(test_set_name)
    # for key in read:
    #     print(f"{key}: {read[key]}")
    # plotter.plot_final_result(test_set_name)
    # plotter.plot_final_result(test_set_name, save_thumbnail=True, dont_show=True)
    # print(result_dict)
