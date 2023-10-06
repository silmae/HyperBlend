"""
Shared functionality that is used by the three leaf models.
"""

from multiprocessing import Pool

import numpy as np
import logging

from src.data import path_handling as P, file_handling as FH
from src import constants as C
from src.rendering import blender_control as BC
from src.utils import general_utils as GU, data_utils as DU


# density_scale = 3000
"""Control how much density variables (absorption and scattering density) are scaled 
for rendering. Value of 1000 cannot produce r = 0 or t = 0. Produced values do not 
significantly change when greater than 3000."""


def _convert_raw_params_to_renderable(ad_raw, sd_raw, ai_raw, mf_raw):
    """Convert machine learning parameters [0,1] to rendering parameters (scaling and re-centering).

    :param ad_raw:
        Numpy array absorption particle density [0,1].
    :param sd_raw:
        Numpy array scattering particle density [0,1].
    :param ai_raw:
        Numpy array scattering anisotropy [0,1].
    :param mf_raw:
        Numpy array mix factor [0,1].
    :return:
        Returns corresponding (ad, sd, ai, mf) that can be fed to rendering script.
    """

    ad = ad_raw * C.density_scale
    sd = sd_raw * C.density_scale
    ai = (ai_raw - 0.5) * 2
    mf = mf_raw
    return ad, sd, ai, mf


def _render(args):
    """Internal render function to be called from parallel code.

    Unpacks given arguments. NOTE that they must be given in correct order so this
    is sensitive to refactoring.
    """

    set_name = args[0]
    sample_id = args[1]
    BC.run_render_series(rend_base_path=P.path_directory_working(set_name, sample_id),
                         wl=args[2],
                         ad=args[3],
                         sd=args[4],
                         ai=args[5],
                         mf= args[6],
                         clear_rend_folder=False, clear_references=False,
                         render_references=True, dry_run=False)


def _material_params_to_RT(set_name, sample_id, wls, ad, sd, ai, mf):
    """ Material parameters are converted to reflectance and transmittance by rendering the leaf model.

    :param set_name:
        Set name.
    :param sample_id:
        Sample id.
    :param wls:
        Numpy array wavelengths.
    :param ad:
        Numpy array absorption particle density.
    :param sd:
        Numpy array scattering particle density.
    :param ai:
        Numpy array scattering anisotropy.
    :param mf:
        Numpy array mix factor.
    :return:
        Returns (r,t) lists of reflectances and transmittances, respectively.
    """

    # Render all wavelengths in parallel
    with Pool() as pool:
        n = pool._processes
        logging.info(f"Using {n} threads for rendering.")
        # Divide given parameter arrays into chucks for each worker thread.
        wl_chunks = GU.chunks(wls, n)
        ad_chunks = GU.chunks(ad, n)
        sd_chunks = GU.chunks(sd, n)
        ai_chunks = GU.chunks(ai, n)
        mf_chunks = GU.chunks(mf, n)

        param_list = [(set_name, sample_id, wl, ad, sd, ai, mf) for wl, ad, sd, ai, mf in
                      zip(wl_chunks, ad_chunks, sd_chunks, ai_chunks, mf_chunks)]
        pool.map(_render, param_list)

    # Get reflectance and transmittance values of rendered images
    r = []
    t = []
    for wl in wls:
        r_wl = DU.get_relative_refl_or_tran(C.imaging_type_refl, wl, base_path=P.path_directory_working(set_name, sample_id))
        t_wl = DU.get_relative_refl_or_tran(C.imaging_type_tran, wl, base_path=P.path_directory_working(set_name, sample_id))
        r.append(r_wl)
        t.append(t_wl)

    return r,t


def _build_sample_res_dict(wls, r, r_m, re, t, t_m, te, ad_raw, sd_raw, ai_raw, mf_raw, elapsed_process_min, elapsed_wall_clock_min):
    """Builds result dictionary to be saved on disk.

    :param wls:
        Wavelengths.
    :param r:
        Modeled reflectances.
    :param r_m:
        Measured reflectances.
    :param re:
        Error of modeled reflectances.
    :param t:
        Modeled transmittances.
    :param t_m:
        Measured transmittances.
    :param te:
        Error of modeled transmittances.
    :param ad_raw:
        Numpy array absorption particle density [0,1].
    :param sd_raw:
        Numpy array scattering particle density [0,1].
    :param ai_raw:
        Numpy array scattering anisotropy [0,1].
    :param mf_raw:
        Numpy array mix factor [0,1].
    :param elapsed_process_min:
        Elapsed time of the processes. Applicable only for optimization method.
        For surface and NN method this will be the same than ``elapsed_wall_clock_min``.
    :param elapsed_wall_clock_min:
        Elapsed wall clock time.
    :return:
        Returns built dictionary.
    """

    sample_result_dict = {}
    sample_result_dict[C.key_sample_result_wls] = wls
    sample_result_dict[C.key_sample_result_r] = r
    sample_result_dict[C.key_sample_result_rm] = r_m
    sample_result_dict[C.key_sample_result_re] = re
    sample_result_dict[C.key_sample_result_t] = t
    sample_result_dict[C.key_sample_result_tm] = t_m
    sample_result_dict[C.key_sample_result_te] = te
    sample_result_dict[C.key_sample_result_ad] = ad_raw
    sample_result_dict[C.key_sample_result_sd] = sd_raw
    sample_result_dict[C.key_sample_result_ai] = ai_raw
    sample_result_dict[C.key_sample_result_mf] = mf_raw
    sample_result_dict[C.key_sample_result_process_elapsed_min] = elapsed_process_min
    sample_result_dict[C.key_sample_result_wall_clock_elapsed_min] = elapsed_wall_clock_min
    sample_result_dict[C.key_sample_result_r_RMSE] = np.sqrt(np.mean(re ** 2))
    sample_result_dict[C.key_sample_result_t_RMSE] = np.sqrt(np.mean(te ** 2))
    return sample_result_dict


def initialize_directories(set_name, clear_old_results=False):
    """
    Create necessary directories.

    Optionally, one can wipe out old results of the same set by setting ``clear_old_results=True``.
    """

    FH.create_first_level_folders(set_name)

    ids = FH.list_target_ids(set_name)
    for _, sample_id in enumerate(ids):
        FH.clear_rend_leaf(set_name, sample_id)
        FH.clear_rend_refs(set_name, sample_id)
        if clear_old_results:
            FH.clear_folder(P.path_directory_subresult(set_name, sample_id))
