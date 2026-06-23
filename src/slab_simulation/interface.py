"""
Interface for the slab model. Most common needs can be called from here.
You should avoid calling slab simulation related code directly unless it
cannot be avoided. It just keeps the hierarchy more clear.
"""

import numpy as np
import time
import logging
import os

import src.slab_simulation.training_data as TD
import src.slab_simulation.leaf_sampling as sampling
from src.slab_simulation.opt import Optimization
from src.data import (
    file_handling as FH,
    toml_handling as TH,
    file_names as FN,
    path_handling as PH,
)
from src import plotter
from src.slab_simulation import nn, surf, slab_commons as LC
from src.prospect import interface as PI
from src.utils import data_utils as DU
from src.setup.runtime_environment import RuntimeEnvironment


def generate_prospect_leaf(
    slab_sim_name,
    signal_id=0,
    n=None,
    ab=None,
    ar=None,
    brown=None,
    w=None,
    m=None,
    ant=None,
):
    """Run prospect simulation with given arguments.

    Calling this is the same as calling :py:func:`prospect.interface.make_leaf_target`.
    See explanation of the arguments there.
    If any of the values are not provided, default values are used.
    You get the default PROSPECT leaf by calling without any of the optional arguments.

    :param slab_sim_name: See :term:`slab_sim_name`.
    :param signal_id: Signal id for this target. Default is 0. Overwrites existing targets if existing id is given.
    :param n: PROSPECT N parameter [unitless]
    :param ab: chlorophyll a + b concentration [ug / cm^2]
    :param ar: carotenoid content [ug / cm^2]
    :param brown: brown pigment [unitless]
    :param w: equivalent water thickness [cm]
    :param m: dry mater content [g / cm^2]
    :param ant: anthocyanin content [ug / cm^2]
    """

    PI.make_leaf_target(slab_sim_name, signal_id, n, ab, ar, brown, w, m, ant)


def generate_prospect_leaf_random(slab_sim_name, leaf_count=1):
    """Generate one or more random PROSPECT leaves.

    Calling this is the same as calling :py:func:`prospect.interface.make_random_leaf_targets`.

    :param slab_sim_name: See :term:`slab_sim_name`.
    :param leaf_count: How many target leaves will be generated.
    """

    PI.make_random_leaf_targets(slab_sim_name, leaf_count)


def resample_slab_sim_target(
    slab_sim_name: str,
    wls: list[int] | np.ndarray = None,
    range_start: int = None,
    range_end: int = None,
    resolution: int = None,
):
    """Runs a spectral resampling of all target signals in this slab simulation.

    After this, you must solve leaf material parameters (for rendering) again
    by calling :py:func:`solve_leaf_material_parameters`.
    Reads the old sampling information from `sampling.toml` in `targets` directory
    and rewrites it before calling the actual resampling from :py:func:`leaf_sampling.resample`.

    :param slab_sim_name: See :term:`slab_sim_name`.
    :param wls: List of new wavelengths. If given, this overrides ``range_start``,
        ``range_end`` and ``resolution``.
    :param range_start: Start of the wavelength range to be resampled (inclusive).
    :param range_end: End of the wavelength range to be resampled (inclusive).
    :param resolution: Resolution of the sampling in nm.

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


def solve_slab_material_parameters(
    runtime: RuntimeEnvironment,
    slab_sim_name: str,
    range_start: int = None,
    range_end: int = None,
    resolution=None,
    wls: list[int] | np.ndarray = None,
    solver="nn",
    clear_old_results=False,
    solver_dirname: str = None,
    copyof: str | None = None,
):
    """Solves leaf material parameters that are needed for rendering.

    The result is saved to disk (the actual data and plots).

    If any of ``wls``, ``range_start``, ``range_end`` and ``resolution`` are given, spectral resampling is
    called before solving the slab parameters. See documentation of these arguments from
    :py:func:`slab_simulation.interface.resample_slab_sim_target`.

    .. note::
        Solvers 'surf' and 'nn' need a trained model to work. Pre-trained model are included
        in the Git repository, but you can train your own using
        :py:func:`slab_simulation.interface.train_models` method. Solver 'opt' does not need prior training,
        but it is slow. In case you have several trained models (whether surf or nn), you can also
        provide the `solver_dirname` to specify which solver to use.

    :param runtime: See :term:`runtime`.
    :param slab_sim_name: See :term:`slab_sim_name`.
    :param solver: Solving method either 'opt', 'surf' or 'nn'. Opt is slowest and most accurate
        (the original method). Surf is fast but not very accurate. NN is fast and fairly accurate.
        Surf and NN are roughly 200 times faster than opt. Recommended solver is the default 'nn'.
    :param clear_old_results: If True, clear old results of the slab simulation. This is handy for redoing the
        same slab simulation with different method, for example. When False, the old results are
        not overwritten and solver just skips the signals that are already solved.
    :param solver_dirname: Name of the (directory of the) solver to be used. If None, the default
        solver is used.
    :param copyof: Name of the slab simulation to copy. Copies target from existing slab simulation
        (walengths, reflectances, and transmittances).

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


def train_models(
    runtime: RuntimeEnvironment,
    slab_sim_name_for_training="training_data",
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
    this will take a lot of time as the data generation uses the original optimization method explained in

    :cite:`riihiaho22`.
    Depending on value of ``train_points_per_dim`` the generation time varies from tens of minutes to several days.
    You should generate a few thousand points (which equals to ``train_points_per_dim``:math:`^{2}`)
    at least for any accuracy. Models in the repository were
    trained with 40 000 points (4 days generation time). Use ``dry_run=True`` just to print the number of
    points that would have been generated.

    You can select to train surface model (``train_surf``) and neural network (``train_nn``) separately
    or just generate the points by setting both to False.

    The plots are saved to the disk even if ``show_plot`` is set to False.

    :param runtime: See :term:`runtime`.
    :param slab_sim_name_for_training:
        The name of the slab simulation that contains or will contain the training data.
        New training data is generated with this name if  ``generate_data=True``.
        Otherwise, existing data with this name is used.
    :param generate_data:
        If True, new training data is generated with given ``slab_sim_name_for_training``.
        Default is False. The training data must exist in order to train the models.
    :param data_generation_diff_step:
        Used in :py:class:`slab_simulation.opt.Optimization` as a stepsize for finite difference Jacobian
        estimation. Smaller step gives better results, but the variables look cloudy. Big
        step is faster and variables smoother but there will be outliers in the results. Good
        stepsize is between 0.001 and 0.01.
    :param starting_guess_type:
        String, one of 'hard-coded', 'curve', 'surf' in order of increasing complexity.
        Hard-coded is only needed if training the other methods from absolute scratch (for
        example if leaf material parameter count or bounds change in future development).
        Curve fitting 'curve' is the method presented in the first HyperBlend paper
        :cite:`riihiaho22`. It will
        only work in cases where R and T are relatively close to each other (around +- 0.2).
        Surface fitting method 'surf' can be used after the first training iteration has been carried
        out. It can more robustly adapt to situations where R and T are dissimilar.
    :param similarity_rt:
        Controls the symmetry of generated pairs, i.e., how much each R value can differ from
        respective T value. Using greater than 0.25 will cause generating a lot of points
        that will fail to be optimized properly (and will be pruned before training). This
        wastes computational resources. Good results were obtained in :cite:`riihiaho25`
        by training multiple times and gradually loosening the similarity requirement.
    :param train_surf:
        If True, train the surface model. Default is True.
    :param train_nn:
        If True, train the neural network. Default is True.
    :param layer_count:
        Number of hidden layers in neural network. Omitted if ``train_nn=False``.
    :param layer_width:
        Width (in number of nodes) of hidden layers in neural network.
        Omitted if ``train_nn=False``.
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
    :param train_points_per_dim:
         Into how many parts each dimension (R,T) are cut in interval [0,1]. Greater value results in more
         training points. Good values from 100 to 500. For testing purposes, low values, e.g., 20 can be used.
         Omitted if ``generate_data=False``.
    :param dry_run:
        Print the number of points that would have been generated, but does not really generate the training points.
        Omitted if ``generate_data=False``.
    :param show_plot:
        If True, shows interactive plots to user (which halts execution until window is closed). Regardless
        of this value, the plots are saved to disk. Default is False.
    :param solver_name_to_save:
        Solver used to save the generated training data and used to train the models if any.
    :param solver_name_to_use:
        Name of the solver to be used. If None, default solver name is used.
        For iterative training, this should be the name of the previous iteration's solver.
    """

    if generate_data:
        TD.generate_train_data(
            runtime=runtime,
            slab_sim_name=slab_sim_name_for_training,
            dry_run=dry_run,
            train_points_per_dim=train_points_per_dim,
            similarity_rt=similarity_rt,
            starting_guess_type=starting_guess_type,
            data_generation_diff_step=data_generation_diff_step,
            solver_name_to_use=solver_name_to_use,
            solver_name_to_save=solver_name_to_save,
        )

    # Do not try to train if it was only a dry run
    if dry_run:
        return

    if train_surf:
        surf.train(
            training_sim_name=slab_sim_name_for_training,
            solver_save_name=solver_name_to_save,
        )
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
            training_sim_name=slab_sim_name_for_training,
            solver_name=solver_name_to_save,
        )

    visualize_slab_model_training(
        training_slab_sim_name=slab_sim_name_for_training,
        show_plot=False,
        plot_surf=train_surf,
        plot_nn=train_nn,
        solver_name=solver_name_to_save,
    )


def iterative_train(
    runtime: RuntimeEnvironment, iterations=8, train_points_per_dim=200, dry_run=False
):
    """Iteratively train the slab models several times.

    .. note::
        This method has many hard-coded values that are passed to
        :py:func:`slab_simulation.interface.train_models`. You may want to modify
        them to your needs.

    :param runtime: See :term:`runtime`.
    :param iterations: The number of iterations to run.
    :param train_points_per_dim: See :py:func:`slab_simulation.interface.train_models`.
    :param dry_run: See :py:func:`slab_simulation.interface.train_models`.
    """

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
                slab_sim_name_for_training=current_iteration_slab_sim_name,
                generate_data=True,
                data_generation_diff_step=first_run_diffstep,
                starting_guess_type="curve",
                similarity_rt=first_run_similarity_requirement,
                train_surf=True,
                train_nn=False,
                train_points_per_dim=train_points_per_dim,
                dry_run=dry_run,
                solver_name_to_save=current_iteration_slab_sim_name,
            )

        elif i == iterations - 1:
            # Last iteration
            train_models(
                runtime=runtime,
                slab_sim_name_for_training=current_iteration_slab_sim_name,
                generate_data=True,
                data_generation_diff_step=diffstep,
                starting_guess_type="surf",
                similarity_rt=last_run_similarity_requirement,
                train_surf=True,
                train_nn=True,
                learning_rate=0.0005,
                train_points_per_dim=train_points_per_dim,
                dry_run=dry_run,
                solver_name_to_save=current_iteration_slab_sim_name,
                solver_name_to_use=previous_iteration_slab_sim_name,
            )
        else:
            # Intermediate iterations
            train_models(
                runtime=runtime,
                slab_sim_name_for_training=current_iteration_slab_sim_name,
                generate_data=True,
                data_generation_diff_step=diffstep,
                starting_guess_type="surf",
                similarity_rt=curr_similarity,
                train_surf=True,
                train_nn=False,
                train_points_per_dim=train_points_per_dim,
                dry_run=dry_run,
                solver_name_to_save=current_iteration_slab_sim_name,
                solver_name_to_use=previous_iteration_slab_sim_name,
            )

        # At the end of the loop, increase the similarity requirement
        curr_similarity += diff_similarity


def visualize_slab_model_training(
    training_slab_sim_name: str,
    show_plot=False,
    plot_surf=True,
    plot_nn=True,
    plot_points=True,
    solver_name=None,
):
    """Visualize trained surface and neural network model against training data.

    The plot is always saved to disk regardless of ``show_plot`` attribute.

    :param training_slab_sim_name:
        Name of the slab simulation that was used as training data.
    :param show_plot:
        If True, show interactive plot. Default is false.
    :param plot_surf:
        If True, plot surface model against training data. Default is False.
    :param plot_nn:
        If True, plot neural network model against training data. Default is True.
    :param plot_points:
        If True, plot training data points. Default is True.
    :param solver_name:
        Name of the solver to be plotted. Default is None and plots the default solver.
    """

    plotter.plot_trained_leaf_models(
        save_thumbnail=True,
        show_plot=show_plot,
        plot_surf=plot_surf,
        plot_nn=plot_nn,
        plot_points=plot_points,
        slab_sim_name=training_slab_sim_name,
        solver_name=solver_name,
    )
