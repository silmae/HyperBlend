"""
Functionality regarding training data generation for
surface model and neural network training.

This is somewhat specific functionality so it is not included
in the leaf model interface script.
"""

import logging
import math
import os.path

import numpy as np

from src.slab_simulation.training_utils import get_starting_guess_points

from src import plotter, constants as C
from src.data import toml_handling as TH, file_handling as FH, path_handling as PH
from src.slab_simulation.opt import Optimization
from src.slab_simulation.training_utils import prune_training_data
from src.utils import general_utils as GU, data_utils as DU
from src.setup.runtime_environment import RuntimeEnvironment

"""
Additional conditions after analyzing erronous areas. 
Constants for two lines equation k*r + b that cut the 
edges of data set away. 
"""
k1 = 3.8
k2 = 0.5
b1 = 0.02
b2 = -0.035


def visualize_training_data_pruning(
    set_name="training_data", show=False, save=True, solver_name=None
):
    """Visualizes training data. Can be saved to disk or shown directly (or both).

    :param solver_name:
    :param set_name:
        Name of the training data set. Change only if custom name was used in data generation.
    :param show:
        Show interactive plot to user. Default is ```False```.
    :param save:
        Save plot to disk. Default is ```True```.
    """

    # We do not use get_training_data() here because we want the original measured r and t
    # for evenly spaced grid
    result = TH.read_signal_result(set_name, signal_id=0)
    ad = np.array(result[C.key_sample_result_ad])
    sd = np.array(result[C.key_sample_result_sd])
    ai = np.array(result[C.key_sample_result_ai])
    mf = np.array(result[C.key_sample_result_mf])
    r = np.array(result[C.key_sample_result_rm])
    t = np.array(result[C.key_sample_result_tm])
    re = np.array(result[C.key_sample_result_re])
    te = np.array(result[C.key_sample_result_te])
    _, _, _, _, r_bad, t_bad = prune_training_data(
        ad, sd, ai, mf, r, t, re, te, invereted=True
    )
    _, _, _, _, r_good, t_good = prune_training_data(
        ad, sd, ai, mf, r, t, re, te, invereted=False
    )
    plotter.plot_training_data_set(
        r_good=r_good,
        t_good=t_good,
        r_bad=r_bad,
        t_bad=t_bad,
        k1=k1,
        b1=b1,
        k2=k2,
        b2=b2,
        show=show,
        save=save,
        save_name=set_name,
        solver_name=solver_name,
    )


def generate_train_data(
    runtime: RuntimeEnvironment,
    slab_sim_name="training_data",
    dry_run=True,
    train_points_per_dim=10,
    similarity_rt=0.25,
    starting_guess_type="curve",
    data_generation_diff_step=0.01,
    solver_name_to_use: str = None,
    solver_name_to_save: str = None,
):
    """Generate reflectance-transmittance pairs as training data for surface fitting and neural network.

    Generated data will have fake wavelengths attached to them. They run from 1 to the number of
    generated points.

    If ``dry_run=True``, only pretends to generate the points. This is useful for testing how
    different ``train_points_per_dim`` values affect the actual point count.

    Data visualization is saved to disk when the data has been generated.

    :param data_generation_diff_step:
        Used in :py:class:`slab_simulation.opt.Optimization` as a stepsize for finite difference Jacobian
        estimation. Smaller step gives better results, but the variables look cloudy. Big
        step is faster and variables smoother but there will be outliers in the results. Good
        stepsize is between 0.001 and 0.01.
    :param slab_sim_name:
        Optionally change the ``slab_sim_name`` that is used for destination directory. If other
        than default is used, it must be taken into account when training, i.e., pass the same
        name for training method.
    :param dry_run:
        If true, just prints how many points would have been generated. Note that it
        is not the same as ``cuts_per_dim`` ^2 because parts of the space are not
        usable and will be cut out.
    :param train_points_per_dim:
        Into how many parts each dimension (R,T) are cut in interval [0,1].
    :param similarity_rt:
        Controls the symmetry of generated pairs, i.e., how much each R value can differ from
        respective T value. Using greater than 0.25 will cause generating a lot of points
        that will fail to be optimized properly (and will be pruned before training).
    :param starting_guess_type:
        One of 'hard-coded', 'curve', 'surf' in order of increasing complexity.
        Hard-coded 'hard-coded' is only needed if training the other methods from absolute scratch (for
        example if leaf material parameter count or bounds change in future development).
        Curve fitting 'curve' is the method presented in the first HyperBlend paper. It will
        only work in cases where R and T are relatively close to each other (around +- 0.2).
        Surface fitting method 'surf' can be used after the first training iteration has been carried
        out. It can more robustly adapt to situations where R and T are dissimilar.
    :param solver_name_to_use:
        Name of the solver to be used. If None, default solver name is used.
    :param solver_name_to_save:
         Solver used to save the generated training data.

    """

    FH.create_top_level_slab_sim_directories(slab_sim_name)

    data = []
    fake_wl = (
        1  # Set dummy wavelengths so that the rest of the code is ok with the files
    )
    R = np.linspace(0, 1.0, train_points_per_dim, endpoint=True)
    T = np.linspace(0, 1.0, train_points_per_dim, endpoint=True)
    for i, r in enumerate(R):
        for j, t in enumerate(T):
            # Do not allow r+t to exceed 1 as it would break conservation of energy
            if not r + t < 1.0:
                continue
            # ensure some amount of symmetry
            if math.fabs(r - t) > similarity_rt:
                continue

            # TODO removed this for now as new training system should overcome this weakness
            # Cutoff points where R and T are low and dissimilar as they will fail anyway.
            # if t > r * k1 + b1:
            #     continue
            # if t < r * k2 + b2:
            #     continue

            wlrt = [fake_wl, r, t]
            data.append(wlrt)
            fake_wl += 1

    if not dry_run:
        logging.info(
            f"Generated {len(data)} evenly spaced reflectance transmittance targets."
        )

        p_solver_dir = PH.directory_slab_model(slab_model_name=solver_name_to_save)
        if not os.path.exists(p_solver_dir):
            os.makedirs(p_solver_dir)

        TH.write_target(slab_sim_name, data, signal_id=0)
        o = Optimization(
            runtime=runtime,
            set_name=slab_sim_name,
            diffstep=data_generation_diff_step,
            starting_guess_type=starting_guess_type,
            solver_name=solver_name_to_use,
        )
        o.run_optimization(resampled=False)
        visualize_training_data_pruning(
            set_name=slab_sim_name,
            show=False,
            save=True,
            solver_name=solver_name_to_save,
        )
    else:
        logging.info(
            f"Would have generated {len(data)} evenly spaced reflectance transmittance pairs"
            f"but this was just a dry run.."
        )


def generate_starting_guess(
    runtime: RuntimeEnvironment,
    slab_sim_name: str = None,
    solver_name: str = None,
    step=None,
):
    """Generates starting guess to be used later on real data.

    Starting guess is generated by running the optimization procedure on a test target
    that linearly decreases absorption. The result is saved to root folder of the project.

    .. note::
        This does not need to be done unless the starting guess in the code repository
        is corrupt or missing.

    :param slab_sim_name: Name of the slab simulation to be used to store the starting guess data.
    :param solver_name: Name of the solver to be used. If None, default solver name is used.
    :param step: The interval between the wavelengths. Default is 1 nm if None given.
        One can use longer step size for testing.
    """

    if slab_sim_name is None:
        slab_sim_name = C.starting_guess_set_name

    FH.create_top_level_slab_sim_directories(slab_sim_name=slab_sim_name)
    o = Optimization(
        runtime=runtime,
        set_name=slab_sim_name,
        starting_guess_type="hard-coded",
        solver_name=solver_name,
    )
    make_linear_test_target(set_name=slab_sim_name, step=step)
    o.run_optimization(use_threads=True, use_basin_hopping=False, resampled=False)
    fit_starting_guess_coefficients(
        slab_sim_name=slab_sim_name, solver_name=solver_name
    )
    plotter._plot_starting_guess_coeffs_fitting(
        slab_sim_name=slab_sim_name, solver_name=solver_name
    )


def fit_starting_guess_coefficients(
    degree=4, slab_sim_name: str = None, solver_name: str = None
):
    """Fits polynomial coefficients to linear test run that are used as a starting guess for the optimization.

    NOTE: One run of this is already stored in the code repo, so this only needs to be done if they corrupt somehow.

    generate_starting_guess() must be run before this method can be run.

    :param degree: Degree of the polynomial to fit.
    :param slab_sim_name: Custom set name to fetch the data from. If not given, default set name variable
        'starting_guess_set_name' stored in constants.py is used.
    :param solver_name: If not given, default solver name is used.
    :return: None.
    """

    if slab_sim_name is None:
        slab_sim_name = C.starting_guess_set_name
    a_list, ad_list, sd_list, ai_list, mf_list = get_starting_guess_points(
        slab_sim_name=slab_sim_name
    )

    ad_coeffs = GU.fit_poly(a_list, ad_list, degree=degree)
    sd_coeffs = GU.fit_poly(a_list, sd_list, degree=degree)
    ai_coeffs = GU.fit_poly(a_list, ai_list, degree=degree)
    mf_coeffs = GU.fit_poly(a_list, mf_list, degree=degree)
    TH.write_starting_guess_coeffs(
        ad_coeffs, sd_coeffs, ai_coeffs, mf_coeffs, solver_name=solver_name
    )


def make_linear_test_target(set_name: str, step: int = None):
    """Creates a test target where reflectance and transmittance grow linearly from 0 to 0.5.

    Wavelength ranges from 400 to 2500.

    :param set_name:
        Set name to be used (such as 'linearity_test').
    :param step: The interval between the wavelengths. Default is 1 nm. One can use longer step size
        for testing purposes.
    :return:
        None
    """

    if step is None:
        step = 1

    start_wl = 400
    end_wl = 2500
    wls = np.arange(start_wl, end_wl + 1, step=step)
    r_m = np.linspace(0, 0.5, len(wls))
    t_m = np.linspace(0, 0.5, len(wls))
    _make_target(set_name, wls, r_m, t_m)


def _make_target(set_name: str, wls, r_m, t_m, sample_id=None):
    """Write target reflectances and transmittances to disk.

    :param set_name:
        Set name to be used.
    :param wls:
        A list of wavelengths to be saved.
    :param r_m:
        A list of measured reflectances to be saved.
    :param t_m:
        A list of measured transmittances to be saved.
    :return:
        None
    """

    if len(wls) != len(r_m) or len(wls) != len(t_m):
        raise ValueError(
            f"Length of the lists of wavelengths ({len(wls)}), reflectances ({len(r_m)}) or transmittances ({len(t_m)}) did not match."
        )
    if sample_id is None:
        sample_id = 0
    FH.create_signal_optimization_directories(set_name, sample_id)
    target_data = DU.pack_target(wls=wls, refls=r_m, trans=t_m)
    TH.write_target(slab_sim_name=set_name, target=target_data, signal_id=sample_id)
