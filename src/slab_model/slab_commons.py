"""
Shared functionality that is used by all slab models: optimization, surface, and neural network.
"""

import os.path
from multiprocessing import Pool

import numpy as np

from src.data import path_handling as P, file_handling as FH
from src import constants as C
from src.rendering import blender_control as BC
from src.utils import general_utils as GU, data_utils as DU
from src.setup.runtime_environment import RuntimeEnvironment


# TODO find a way to get rid of this hard-coded stuff.
density_scale = 3000
"""Control how much density variables (absorption and scattering density) are scaled 
for rendering. Value of 1000 cannot produce r = 0 or t = 0. Produced values do not 
significantly change when greater than 3000.
"""


def _convert_raw_params_to_renderable(ad_raw, sd_raw, ai_raw, mf_raw):
    """Convert machine learning parameters in range [0,1] to rendering parameters (scaling and re-centering).

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

    ad = ad_raw * density_scale
    sd = sd_raw * density_scale
    ai = (ai_raw - 0.5) * 2
    mf = mf_raw
    return ad, sd, ai, mf


def _render(args):
    """Internal render function to be called from parallel code.

    Unpacks given arguments. NOTE that they must be given in correct order so this
    is sensitive to refactoring.
    """

    runtime = args[0]
    slab_sim_name = args[1]
    signal_id = args[2]
    p = P.directory_slab_optimization_working(slab_sim_name, signal_id)

    if not os.path.exists(p):
        raise FileNotFoundError(
            f"File {p} does not exist. Cannot render the slab model."
        )

    BC.run_render_series(
        runtime=runtime,
        rend_base_path=p,
        wl=args[3],
        ad=args[4],
        sd=args[5],
        ai=args[6],
        mf=args[7],
        clear_rend_folder=False,
        clear_references=False,
        render_references=True,
        dry_run=False,
    )


def _material_params_to_RT(
    runtime: RuntimeEnvironment, slab_sim_name: str, signal_id: int, wls, ad, sd, ai, mf
):
    """Material parameters are converted to reflectance and transmittance by rendering the slab model.

    :param slab_sim_name: Name of the slab simulation.
    :param signal_id:  Signal ID.
    :param wls: Numpy array wavelengths.
    :param ad: Numpy array absorption particle density.
    :param sd: Numpy array scattering particle density.
    :param ai: Numpy array scattering anisotropy.
    :param mf: Numpy array mix factor.
    :return: Returns (r,t) lists of reflectances and transmittances, respectively.
    """

    # Render all wavelengths in parallel
    with Pool() as pool:
        n = pool._processes
        # Divide given parameter arrays into chucks for each worker thread.
        wl_chunks = GU.chunks(wls, n)
        ad_chunks = GU.chunks(ad, n)
        sd_chunks = GU.chunks(sd, n)
        ai_chunks = GU.chunks(ai, n)
        mf_chunks = GU.chunks(mf, n)

        param_list = [
            (runtime, slab_sim_name, signal_id, wl, ad, sd, ai, mf)
            for wl, ad, sd, ai, mf in zip(
                wl_chunks, ad_chunks, sd_chunks, ai_chunks, mf_chunks
            )
        ]
        pool.map(_render, param_list)

    # Get reflectance and transmittance values of rendered images
    r = []
    t = []
    for wl in wls:
        r_wl = DU.get_relative_refl_or_tran(
            C.imaging_type_refl,
            wl,
            base_path=P.directory_slab_optimization_working(slab_sim_name, signal_id),
        )
        t_wl = DU.get_relative_refl_or_tran(
            C.imaging_type_tran,
            wl,
            base_path=P.directory_slab_optimization_working(slab_sim_name, signal_id),
        )
        r.append(r_wl)
        t.append(t_wl)

    return r, t


def _build_sample_res_dict(
    wls,
    r,
    r_m,
    re,
    t,
    t_m,
    te,
    ad_raw,
    sd_raw,
    ai_raw,
    mf_raw,
    elapsed_process_min,
    elapsed_wall_clock_min,
):
    """Builds result dictionary to be saved on disk.

    :param wls: Wavelengths.
    :param r:  Modeled reflectances.
    :param r_m: Measured reflectances.
    :param re: Error of modeled reflectances.
    :param t: Modeled transmittances.
    :param t_m: Measured transmittances.
    :param te: Error of modeled transmittances.
    :param ad_raw: Numpy array absorption particle density [0,1].
    :param sd_raw: Numpy array scattering particle density [0,1].
    :param ai_raw: Numpy array scattering anisotropy [0,1].
    :param mf_raw: Numpy array mix factor [0,1].
    :param elapsed_process_min: Elapsed time of the processes. Applicable only for
        optimization method. For surface and NN method this will be the same
        as ``elapsed_wall_clock_min``.
    :param elapsed_wall_clock_min: Elapsed wall clock time.
    :return: Returns built dictionary.
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
    sample_result_dict[C.key_sample_result_wall_clock_elapsed_min] = (
        elapsed_wall_clock_min
    )
    sample_result_dict[C.key_sample_result_r_RMSE] = np.sqrt(np.mean(re**2))
    sample_result_dict[C.key_sample_result_t_RMSE] = np.sqrt(np.mean(te**2))
    return sample_result_dict


def initialize_directories(slab_sim_name, clear_old_results=False):
    """Create necessary directories.

    Optionally, one can wipe out old results of the same slab simulation by
    setting ``clear_old_results=True``.

    :param slab_sim_name: Slab simulation name.
    :param clear_old_results: If True, old results are deleted.
    """

    FH.create_top_level_slab_sim_directories(slab_sim_name)

    ids = FH.list_target_ids(slab_sim_name)
    for _, signal_id in enumerate(ids):
        FH.clear_rend_slab(slab_sim_name, signal_id)
        FH.clear_rend_refs(slab_sim_name, signal_id)
        if clear_old_results:
            FH.clear_directory(
                P.directory_optimization_result(slab_sim_name, signal_id)
            )
