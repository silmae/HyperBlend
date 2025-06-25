"""

Interface for all leaf material related actions.

"""

import numpy as np
import time
import logging
import os

import src.slab_model.training_data as TD
import src.slab_model.leaf_sampling as sampling
from src.slab_model.opt import Optimization
from src.data import (
    file_handling as FH,
    toml_handling as TH,
    file_names as FN,
    path_handling as PH,
)
from src import plotter
from src.slab_model import nn, surf, slab_commons as LC
from src.prospect import interface
from src.utils import data_utils as DU
from src.setup.runtime_environment import RuntimeEnvironment


def generate_prospect_leaf(
    set_name,
    sample_id=0,
    n=None,
    ab=None,
    ar=None,
    brown=None,
    w=None,
    m=None,
    ant=None,
):
    """Run prospect simulation with given PROSPECT parameters.

    If any of the values are not provided, default values are used (see prospect.p_default_dict).
    You get the default PROSPECT leaf by calling without any arguments.

    Calling this is the same as calling prospect.make_leaf_target().

    :param set_name: Set name where the target is saved.
    :param sample_id: Sample id for this target. Default is 0. Overwrites existing targets if existing id is given.
    :param n: PROSPECT N parameter [unitless]
    :param ab: chlorophyll a + b concentration [ug / cm^2]
    :param ar: cartenoid content [ug / cm^2]
    :param brown: brown pigment [unitless]
    :param w: equivalent water thickness [cm]
    :param m: dry mater content [g / cm^2]
    :param ant: anthocyanin content [ug / cm^2]
    """

    interface.make_leaf_target(set_name, sample_id, n, ab, ar, brown, w, m, ant)


def generate_prospect_leaf_random(slab_sim_name, leaf_count=1):
    """Generate count number of random PROSPECT leaves.

    Calling this is the same as calling prospect.make_random_leaf_targets().

    :param slab_sim_name: Set name to be used.
    :param leaf_count: How many target leaves are generated to the set.
    """

    interface.make_random_leaf_targets(slab_sim_name, leaf_count)


def resample_slab_sim_target(
    slab_sim_name: str,
    wls: list[int] | np.ndarray = None,
    range_start: int = None,
    range_end: int = None,
    resolution: int = None,
):
    """Resamples slab simulation targets.

    After this, you must solve leaf material parameters (for rendering) again.
    Uses sampling information from `sampling.toml` in `targets` directory.

    :param slab_sim_name: Name of the slab simulation.
    :param range_start: Start of the wavelength range to be resampled (inclusive).
    :param range_end: End of the wavelength range to be resampled (inclusive).
    :param resolution: Resolution of the sampling in nm.
    :param wls: List of new wavelengths. If given this overrides ``range_start``,
        ``range_end`` and ``resolution``.

    :raises AttributeError: If neither ``wls`` nor ``range_start``, ``range_end``
        and ``resolution`` are given.
        This error is also raised if ``range_start`` is less than the minimum wavelength
        or ``range_end`` is greater than the maximum wavelength in the target data.
    """

    if wls is not None:

        target = TH.read_target(slab_sim_name=slab_sim_name, signal_id=0)
        target_wls, _, _ = DU.unpack_target(target=target)
        target_wls = np.array(target_wls)

        if np.min(wls) < np.min(target_wls):
            raise AttributeError(
                f"Minimum wavelength {np.min(wls)} in 'wls' is less than minimum wavelength "
                f"{np.min(target_wls)} in target data."
            )
        if np.max(wls) > np.max(target_wls):
            raise AttributeError(
                f"Maximum wavelength {np.max(wls)} in 'wls' is greater than maximum wavelength "
                f"{np.max(target_wls)} in target data."
            )
        new_sampling = np.array(wls)
    elif range_start is not None and range_end is not None and resolution is not None:

        target = TH.read_target(slab_sim_name=slab_sim_name, signal_id=0)
        target_wls, _, _ = DU.unpack_target(target=target)
        target_wls = np.array(target_wls)

        if range_start < np.min(target_wls):
            raise AttributeError(
                f"range_start {range_start} is less than minimum wavelength "
                f"{np.min(target_wls)} in target data."
            )
        if range_end > np.max(target_wls):
            raise AttributeError(
                f"range_end {range_end} is greater than maximum wavelength "
                f"{np.max(target_wls)} in target data."
            )
        new_sampling = np.arange(start=range_start, stop=range_end + 1, step=resolution)
    else:
        raise AttributeError(
            f"Either 'wls' or 'range_start', 'range_end' and 'resolution' must be given. "
        )

    TH.write_sampling(
        slab_sim_name=slab_sim_name, sampling=new_sampling, overwrite=True
    )
    sampling.resample(slab_sim_name=slab_sim_name, plot_resampling=True)


def solve_leaf_material_parameters(
    runtime: RuntimeEnvironment,
    slab_sim_name: str,
    range_start: int = None,
    range_end: int = None,
    resolution=None,
    wls: list[int] | np.ndarray = None,
    solver="nn",
    clear_old_results=False,
    solver_dirname: str = None,
    copyof=None,
):
    """Solves leaf material parameters for rendering.

    The result is saved to disk: this method does not have a return value.

    Note that solvers 'surf' and 'nn' need trained model to work. Pre-trained model are included
    in the Git repository, but you can train your own using ``train_models()`` method. Solver 'opt'
    does not need training.

    :param slab_sim_name: Name of the measurement set.
    :param resolution: If resolution is None (default), spectral sampling defined
        in `sampling.toml` will be used. If resolution is provided and can be interpreted
        as an int, new sampling is written from 400 nm to 2500 nm with given `resolution` nm intervals.
    :param solver: Solving method either 'opt', 'surf' or 'nn'. Opt is slowest and most accurate
        (the original method). Surf is fast but not very accurate. NN is fast and fairly accurate.
        Surf and NN are roughly 200 times faster than opt. Recommended solver is the default 'nn'.
    :param clear_old_results: If True, clear old results of the set. This is handy for redoing the
        same set with different method, for example. Note that existing wavelength results are
        not redone unless first removed.
    :param solver_dirname: Name of the (directory of the) solver to be used. If None, the default
        solver is used.
    :param copyof: Name of the set to copy. Copies target from existing set (walengths, reflectances,
        and transmittances).
    """

    if copyof:
        FH.copy_slab_simulation_target(
            src_slab_sim_name=copyof, dst_slab_sim_name=slab_sim_name
        )
    else:
        LC.initialize_directories(
            slab_sim_name=slab_sim_name, clear_old_results=clear_old_results
        )

    new_sampling_requested = True
    if wls is None and range_start is None and range_end is None and resolution is None:
        new_sampling_requested = False

    if new_sampling_requested:
        resample_slab_sim_target(
            slab_sim_name=slab_sim_name,
            wls=wls,
            range_start=range_start,
            range_end=range_end,
            resolution=resolution,
        )

    # This will result True if new resampling was written or a previous one already exists.
    p = PH.file_slab_target(slab_sim_name=slab_sim_name, signal_id=0, resampled=True)
    use_resampling = os.path.exists(p)

    ids = FH.list_target_ids(slab_sim_name)
    ids.sort()

    if len(ids) < 1:
        raise RuntimeError(
            f"Could not find any target signals for slab simulation '{slab_sim_name}''."
        )

    for _, signal_id in enumerate(ids):
        FH.create_slab_sim_signal_directories(
            slab_sim_name=slab_sim_name, signal_id=signal_id
        )
        logging.info(f"Solving slab parameters of Signal {signal_id}")
        targets = TH.read_target(slab_sim_name, signal_id, resampled=use_resampling)

        if solver == "opt":
            FH.create_signal_optimization_directories(slab_sim_name, signal_id)
            o = Optimization(
                runtime=runtime, set_name=slab_sim_name, solver_name=solver_dirname
            )
            o.run_optimization(resampled=use_resampling)
        elif solver == "surf" or solver == "nn":
            start = time.perf_counter()

            wls = targets[:, 0]
            r_m = targets[:, 1]
            t_m = targets[:, 2]

            if solver == "surf":
                ad_raw, sd_raw, ai_raw, mf_raw = surf.predict(
                    target_refl=r_m, target_tran=t_m, solver_dirname=solver_dirname
                )
            elif solver == "nn":
                ad_raw, sd_raw, ai_raw, mf_raw = nn.predict(
                    target_refl=r_m, target_tran=t_m, solver_dirname=solver_dirname
                )
            else:
                raise AttributeError(f"Unknown solver '{solver}'.")

            ad, sd, ai, mf = LC._convert_raw_params_to_renderable(
                ad_raw, sd_raw, ai_raw, mf_raw
            )

            r, t = LC._material_params_to_RT(
                runtime=runtime,
                slab_sim_name=slab_sim_name,
                signal_id=signal_id,
                wls=wls,
                ad=ad,
                sd=sd,
                ai=ai,
                mf=mf,
            )

            re = np.abs(r - r_m)
            te = np.abs(t - t_m)
            running_time = (time.perf_counter() - start) / 60.0
            time_process_min = running_time
            time_wall_clock_min = running_time
            sample_result_dict = LC._build_sample_res_dict(
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
                time_process_min,
                time_wall_clock_min,
            )

            TH.write_signal_result(slab_sim_name, signal_id, sample_result_dict)

            plotter.plot_signal_result(
                slab_sim_name, signal_id, dont_show=True, save_thumbnail=True
            )
        else:
            raise AttributeError(
                f"Unknown solver '{solver}'. Use one of ['nn','surf','opt']."
            )

    TH.write_slab_sim_result(slab_sim_name)
    plotter.plot_slab_sim_result(slab_sim_name, dont_show=True, save_thumbnail=True)
    plotter.plot_slab_sim_errors(slab_sim_name, dont_show=True, save_thumbnail=True)


def iterative_train(
    runtime: RuntimeEnvironment, iterations=8, training_points=200, dry_run=False
):

    first_run_similarity_requirement = 0.2
    last_run_similarity_requirement = 1.0
    diff_similarity = (
        last_run_similarity_requirement - first_run_similarity_requirement
    ) / iterations
    curr_similarity = first_run_similarity_requirement

    # Diffstep for optimizer's finite difference Jacobian estimation
    first_run_diffstep = 0.01
    diffstep = 0.001

    logging.info(f"Starting training loop")

    for i in range(iterations):

        logging.info(f"Iteration {i}")

        current_iteration_slab_sim_name = f"train_iter_{i+1}"
        previous_iteration_slab_sim_name = f"train_iter_{i}"

        if i == 0:
            # First iteration
            train_models(
                runtime=runtime,
                set_name=current_iteration_slab_sim_name,
                generate_data=True,
                data_generation_diff_step=first_run_diffstep,
                starting_guess_type="curve",
                similarity_rt=first_run_similarity_requirement,
                train_surf=True,
                train_nn=False,
                train_points_per_dim=training_points,
                dry_run=dry_run,
                solver_name_to_save=current_iteration_slab_sim_name,
            )

        elif i == iterations - 1:
            # Last iteration
            train_models(
                runtime=runtime,
                set_name=current_iteration_slab_sim_name,
                generate_data=True,
                data_generation_diff_step=diffstep,
                starting_guess_type="surf",
                similarity_rt=last_run_similarity_requirement,
                train_surf=True,
                train_nn=True,
                learning_rate=0.0005,
                train_points_per_dim=training_points,
                dry_run=dry_run,
                solver_name_to_save=current_iteration_slab_sim_name,
                solver_name_to_use=previous_iteration_slab_sim_name,
            )
        else:
            # Intermediate iterations
            train_models(
                runtime=runtime,
                set_name=current_iteration_slab_sim_name,
                generate_data=True,
                data_generation_diff_step=diffstep,
                starting_guess_type="surf",
                similarity_rt=curr_similarity,
                train_surf=True,
                train_nn=False,
                train_points_per_dim=training_points,
                dry_run=dry_run,
                solver_name_to_save=current_iteration_slab_sim_name,
                solver_name_to_use=previous_iteration_slab_sim_name,
            )

        # At the end of the loop, increase the similarity requirement
        curr_similarity += diff_similarity


def train_models(
    runtime: RuntimeEnvironment,
    set_name="training_data",
    generate_data=False,
    data_generation_diff_step=0.01,
    starting_guess_type="curve",
    similarity_rt=0.25,
    train_surf=True,
    train_nn=True,
    layer_count=5,
    layer_width=1000,
    epochs=300,
    batch_size=32,
    learning_rate=0.01,
    patience=30,
    split=0.1,
    train_points_per_dim=20,
    dry_run=False,
    show_plot=False,
    solver_name_to_save=None,
    solver_name_to_use=None,
):
    """Train surface model and neural network.

    If training data does not yet exist, it must be created by setting ``generate_data=True``. Note that
    this will take a lot of time as the data generation uses the original optimization method. Depending
    on value of ``train_points_per_dim`` the generation time varies from tens of minutes to several days.
    You should generate a few thousand points at least for any accuracy. Models in the repository were
    trained with 40 000 points (4 days generation time). Use ``dry_run=True`` just to print the number of
    points that would have been generated.

    You can select to train surface model (``train_surf``) and neural network (``train_nn``) separately
    or just generate the points.

    TODO: The names are pure chaos now. There is a name for the dataset to use, name for the solver to
        be saved, and a name for the solver that is used as a starting guess. Some sense must be made of this.

    Show plot is safe to be kept at default ``False``. The plots are saved to the disk anyways.

    :param data_generation_diff_step:
    :param starting_guess_type:
            One of 'hard-coded', 'curve', 'surf' in order of increasing complexity.
            Hard-coded 'hard-coded' is only needed if training the other methods from absolute scratch (for
            example if leaf material parameter count or bounds change in future development).
            Curve fitting 'curve' is the method presented in the first HyperBlend paper. It will
            only work in cases where R and T are relatively close to each other (around +- 0.2).
            Surface fitting method 'surf' can be used after the first training iteration has been carried
            out. It can more robustly adapt to situations where R and T are dissimilar.
    :param similarity_rt:
    :param set_name:
        Set name of the training data. New training data is generated with this name if  ``generate_data=True``.
        Otherwise, existing data with this name is used.
    :param show_plot:
        If True, shows interactive plots to user (which halts excecution until window is closed). Regardless
        of this value, the plots are saved to disk. Default is False.
    :param layer_count:
        Number of hidden layers in neural network. Omitted if ``train_nn=False``.
    :param layer_width:
        Width of hidden layers in neural network. Omitted if ``train_nn=False``.
    :param epochs:
        Maximum number of epochs the neural network is trained. Omitted if ``train_nn=False``.
    :param batch_size:
        Batch size when training neural network. Omitted if ``train_nn=False``. Smaller values (e.g. 2) yield better
        accuracy while bigger values (e.g. 32) train faster.
    :param learning_rate:
        Learning rate of the Adam optimizer. Default value of 0.001 is good and this has very little effect on
        training results. Feel free to test different values. Omitted if ``train_nn=False``.
    :param patience:
        Stop NN training if the loss has not improved in this many epochs. Omitted if ``train_nn=False``.
    :param split:
        Percentage [0,1] of data reserved for testing between epochs. Value between 0.1 and 0.2
        is usually sufficient. Omitted if ``train_nn=False``.
    :param generate_data:
        If True, new training data is generated with given ``set_name``. Default is False. The training data
        must exist in order to train the models.
    :param train_points_per_dim:
         Into how many parts each dimension (R,T) are cut in interval [0,1]. Greater value results in more
         training points. Good values from 100 to 500. For testing purposes, low values, e.g., 20 can be used.
         Omitted if ``generate_data=False``.
    :param dry_run:
        Print the number of points that would have been generated, but does not really generate the training points.
        Omitted if ``generate_data=False``.
    :param train_surf:
        If True, train the surface model. Default is True.
    :param train_nn:
        If True, train the neural network. Default is True.
    :param solver_name_to_save: Name of the solver used to get a starting guess if ``starting_guess_type='surf'``.
        For iterative training, this should be the name of the previous iteration's solver.
    """

    if generate_data:
        TD.generate_train_data(
            runtime=runtime,
            set_name=set_name,
            dry_run=dry_run,
            cuts_per_dim=train_points_per_dim,
            similarity_rt=similarity_rt,
            starting_guess_type=starting_guess_type,
            data_generation_diff_step=data_generation_diff_step,
            solver_name_to_use=solver_name_to_use,
            solver_name_to_save=solver_name_to_save,
        )

    if dry_run:
        return

    if train_surf:
        surf.train(training_sim_name=set_name, solver_save_name=solver_name_to_save)
    if train_nn:
        nn.train(
            show_plot=show_plot,
            layer_count=layer_count,
            layer_width=layer_width,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            patience=patience,
            split=split,
            training_sim_name=set_name,
            solver_name=solver_name_to_save,
        )
    #
    # nn_name = FN.get_nn_save_name(layer_count=layer_count, layer_width=layer_width, batch_size=batch_size,
    #                               lr=learning_rate, split=split, training_set=set_name)

    visualize_leaf_models(
        training_set_name=set_name,
        show_plot=False,
        plot_surf=train_surf,
        plot_nn=train_nn,
        solver_name=solver_name_to_save,
    )


def visualize_leaf_models(
    training_set_name: str,
    show_plot=False,
    plot_surf=True,
    plot_nn=True,
    plot_points=True,
    solver_name=None,
):
    """Visualize trained surface and neural network model against training data.

    The plot is always saved to disk regardless of ``show_plot`` flag.

    :param solver_name:
    :param show_plot:
        If True, show interactive plot. Default is false.
    """

    plotter.plot_trained_leaf_models(
        save_thumbnail=True,
        show_plot=show_plot,
        plot_surf=plot_surf,
        plot_nn=plot_nn,
        plot_points=plot_points,
        set_name=training_set_name,
        solver_name=solver_name,
    )
