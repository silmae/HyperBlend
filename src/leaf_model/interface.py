"""

Interface for all leaf material related actions.

"""

import numpy as np

import time
import logging

import src.leaf_model.training_data
import src.leaf_model.leaf_sampling as sampling
from src.leaf_model.opt import Optimization
from src.data import file_handling as FH, toml_handling as TH, file_names as FN
from src import plotter
from src.leaf_model import nn, surf, leaf_commons as LC
from src.prospect import prospect
from src.utils import data_utils as DU


def generate_prospect_leaf(set_name, sample_id=0, n=None, ab=None, ar=None, brown=None, w=None, m=None, ant=None):
    """ Run prospect simulation with given PROSPECT parameters.

    If any of the values are not provided, default values are used (see prospect.p_default_dict).
    You get the default PROSPECT leaf by calling without any arguments.

    Calling this is the same as calling prospect.make_leaf_target().

    :param set_name:
        Set name where the target is saved.
    :param sample_id:
        Sample id for this target. Default is 0. Overwrites existing targets if existing id is given.
    :param n:
        PROSPECT N parameter [unitless]
    :param ab:
        chlorophyll a + b concentration [ug / cm^2]
    :param ar:
        cartenoid content [ug / cm^2]
    :param brown:
        brown pigment [unitless]
    :param w:
        equivalent water thickness [cm]
    :param m:
        dry mater content [g / cm^2]
    :param ant:
        anthocyanin content [ug / cm^2]
    """

    prospect.make_leaf_target(set_name, sample_id, n, ab, ar, brown, w, m, ant)


def generate_prospect_leaf_random(set_name, leaf_count=1):
    """ Generate count number of random PROSPECT leaves.

    Calling this is the same as calling prospect.make_random_leaf_targets().

    :param set_name:
        Set name to be used.
    :param leaf_count:
        How many target leaves are generated to the set.
    """

    prospect.make_random_leaf_targets(set_name, leaf_count)


def resample_leaf_targets(set_name: str, new_sampling=None):
    """Resamples leaf targets.

    After this, you must solve leaf material parameters (for rendering) again.
    Uses sampling information from `sampling.toml` in `targets` directory.

    :param set_name:
        Set to be resampled.
    :param new_sampling:
        List of new wavelenghts. Optional. If not given, an empty sampling file
        is written that can be modified manually.
    """

    TH.write_sampling(set_name=set_name, sampling=new_sampling, overwrite=True)
    sampling.resample(set_name=set_name)


def solve_leaf_material_parameters(set_name: str, resolution=None, use_dumb_sampling=False, solver='nn', clear_old_results=False, solver_model_name=None,
                                   copyof=None, plot_resampling=True, surf_model_name=None):
    """Solves leaf material parameters for rendering.
    
    The result is saved to disk: this method does not have a return value.
    
    Note that solvers 'surf' and 'nn' need trained model to work. Pre-trained model are included  
    in the Git repository, but you can train your own using ``train_models()`` method. Solver 'opt' 
    does not need training.

    :param set_name:
        Name of the measurement set.
    :param resolution: 
        If resolution is None (default), spectral sampling defined in `sampling.toml` will be used.
        If resolution is provided and can be interpreted as an int, new sampling is written from 400 nm
        to 2500 nm with given `resolution` nm intervals.
    :param use_dumb_sampling:

    :param solver:
        Solving method either 'opt', 'surf' or 'nn'. Opt is slowest and most accurate (the original method). Surf is 
        fast but not very accurate. NN is fast and fairly accurate. Surf and NN are roughly 200 times faster 
        than opt. Recommended solver is the default 'nn'.  
    :param clear_old_results: 
        If True, clear old results of the set. This is handy for redoing the same set with different method, for 
        example. Note that existing wavelength results are not redone unless first removed.
    :param solver_model_name:
        Name of the neural network or surface model to use. The default is 'nn_default' but if you have trained your own custom
        NN, use that name (or rename your NN to 'nn_default.pt'. TODO fix docs
    :param copyof: 
        Name of the set to copy. Copies target from existing set (walengths, reflectances, and transmittances).
    """

    if copyof:
        FH.copy_target(from_set=copyof, to_set=set_name)
    else:
        LC.initialize_directories(set_name=set_name, clear_old_results=clear_old_results)

    if resolution is not None:
        step = int(resolution) # let it fail if cannot be cast to int
        target = TH.read_target(set_name=set_name, sample_id=0) # raises error if target not found
        wls,_,_ = DU.unpack_target(target=target)
        wls = np.array(wls)
        sampling_start = max(np.min(wls), 400)
        sampling_end = min(np.max(wls) + 1, 2501)
        sampling_even = np.arange(sampling_start, sampling_end, step=step)
        TH.write_sampling(set_name=set_name, sampling=sampling_even, overwrite=True)
    else:
        # If given resolution is None, i.e., we expect proper sampling to exist but it does not
        if sampling.sampling_empty(set_name=set_name) and not use_dumb_sampling:
            raise RuntimeError(f"Sampling has not been defined for set '{set_name}'. "
                               f"Cannot solve leaf material parameters.")

    if not use_dumb_sampling:
        sampling.resample(set_name=set_name, plot_resampling=plot_resampling)

    ids = FH.list_target_ids(set_name)
    ids.sort()

    if len(ids) < 1:
        raise RuntimeError(f'Could not find any targets for set "{set_name}".')

    for _, sample_id in enumerate(ids):
        FH.create_opt_folder_structure_for_samples(set_name, sample_id)
        logging.info(f'Starting optimization of sample {sample_id}')
        targets = TH.read_target(set_name, sample_id, resampled=not use_dumb_sampling)

        # TODO the sampling is now a problem as the new sampling cannot properly handle the
        #    5 nm resolution used in the published tests.
        # Spectral resolution
        if resolution != 1 and use_dumb_sampling:
            targets = targets[::resolution]

        if solver == 'opt':
            # use_resampling = not disable_sampling and resolution is not None
            o = Optimization(set_name=set_name, surf_model_name=surf_model_name)
            o.run_optimization(resampled=not use_dumb_sampling)
        elif solver == 'surf' or solver == "nn":
            start = time.perf_counter()

            wls = targets[:, 0]
            r_m = targets[:, 1]
            t_m = targets[:, 2]

            if solver == 'surf' and solver_model_name is not None:
                ad_raw, sd_raw, ai_raw, mf_raw = surf.predict(r_m=r_m, t_m=t_m, surface_model_name=solver_model_name)
            elif solver == "nn":
                if solver_model_name: # when using custom NN
                    ad_raw, sd_raw, ai_raw, mf_raw = nn.predict(r_m=r_m, t_m=t_m, nn_name=solver_model_name)
                else: # when using default NN
                    ad_raw, sd_raw, ai_raw, mf_raw = nn.predict(r_m=r_m, t_m=t_m)

            ad, sd, ai, mf = LC._convert_raw_params_to_renderable(ad_raw, sd_raw, ai_raw, mf_raw)
            r, t = LC._material_params_to_RT(set_name, sample_id, wls, ad, sd, ai, mf)

            re = np.abs(r - r_m)
            te = np.abs(t - t_m)
            running_time = (time.perf_counter() - start) / 60.0
            time_process_min = running_time
            time_wall_clock_min = running_time
            sample_result_dict = LC._build_sample_res_dict(wls, r, r_m, re, t, t_m, te, ad_raw, sd_raw, ai_raw, mf_raw, time_process_min, time_wall_clock_min)

            TH.write_sample_result(set_name, sample_result_dict, sample_id)

            plotter.plot_sample_result(set_name, sample_id, dont_show=True, save_thumbnail=True)
        else:
            raise AttributeError(f"Unknown solver '{solver}'. Use one of ['nn','surf','opt'].")

    TH.write_set_result(set_name)
    plotter.plot_set_result(set_name, dont_show=True, save_thumbnail=True)
    plotter.plot_set_errors(set_name, dont_show=True, save_thumbnail=True)


def train_models(set_name='training_data', generate_data=False, data_generation_diff_step=0.01,
                 starting_guess_type='curve', surface_model_name=None, similarity_rt=0.25, train_surf=True,
                 train_nn=True, layer_count=5, layer_width=1000, epochs=300, batch_size=32, learning_rate=0.01,
                 patience=30, split=0.1, train_points_per_dim=20, dry_run=False, show_plot=False):
    """Train surface model and neural network.
    
    If training data does not yet exist, it must be created by setting ``generate_data=True``. Note that 
    this will take a lot of time as the data generation uses the original optimization method. Depending 
    on value of ``train_points_per_dim`` the generation time varies from tens of minutes to several days. 
    You should generate a few thousand points at least for any accuracy. Models in the repository were 
    trained with 40 000 points (4 days generation time). Use ``dry_run=True`` just to print the number of 
    points that would have been generated.
    
    You can select to train surface model (``train_surf``) and neural network (``train_nn``) separately 
    or just generate the points.
    
    Show plot is safe to be kept at default ``False``. The plots are saved to the disk anyways. 
    
    :param data_generation_diff_step:
    :param surface_model_name:
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
    """

    if generate_data:
        src.leaf_model.training_data.generate_train_data(set_name=set_name, dry_run=dry_run,
                                                         cuts_per_dim=train_points_per_dim, similarity_rt=similarity_rt,
                                                         starting_guess_type=starting_guess_type,
                                                         surf_model_name=surface_model_name, data_generation_diff_step=data_generation_diff_step)

    if dry_run:
        return

    if train_surf:
        surf.train(set_name=set_name)
    if train_nn:
        nn.train(show_plot=show_plot, layer_count=layer_count, layer_width=layer_width, epochs=epochs,
                 batch_size=batch_size, learning_rate=learning_rate, patience=patience, split=split, set_name=set_name)

    nn_name = FN.get_nn_save_name(layer_count=layer_count, layer_width=layer_width, batch_size=batch_size,
                                  lr=learning_rate, split=split, training_set=set_name)

    visualize_leaf_models(show_plot=False,training_set_name=set_name, nn_name=nn_name, plot_nn=train_nn, plot_surf=train_surf)


def visualize_leaf_models(training_set_name:str, show_plot=False, nn_name='nn_default', plot_surf=True,
                          plot_nn=True, plot_points=True):
    """Visualize trained surface and neural network model against training data.

    The plot is always saved to disk regardless of ``show_plot`` flag.

    :param show_plot:
        If True, show interactive plot. Default is false.
    """

    plotter.plot_trained_leaf_models(save_thumbnail=True, show_plot=show_plot, plot_surf=plot_surf, plot_nn=plot_nn,
                                     plot_points=plot_points, nn_name=nn_name, set_name=training_set_name)
