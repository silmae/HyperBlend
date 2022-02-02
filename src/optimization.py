"""
Optimization class and related methods.
"""

import math
import time
import logging
from multiprocessing import Pool

import scipy.optimize as optimize
import numpy as np


from src import constants as C
from src.data.toml_handling import make_sample_result
from src.rendering import blender_control as B
from src.utils import data_utils as DU
from src.utils import general_utils as GU
from src.data import file_handling as FH, toml_handling as TH, path_handling as P
from src import plotter
from src.surface_model import fitting_function as FF

hard_coded_starting_guess = [0.28, 0.43, 0.77, 0.28]
"""This should be used only if the starting guess based on polynomial fitting is not available. 
 Will produce worse results and is slower. """

LOWER_BOUND = [0.000001, 0.000001, 0.0, 0.0]
"""Lower constraints of the minimization problem. Absorption and scattering particle density cannot be exactly 
zero as it may cause problems in rendering. """

UPPER_BOUND = [1.0, 1.0, 1.0, 1.0]
"""Upper limit of the minimization problem."""


class Optimization:
    """
        Optimization class runs a least squares optimization of the HyperBlend leaf spectral model
        against a set of measured leaf spectra.
    """

    def __init__(self, set_name: str, ftol=1e-2, ftol_abs=1.0, xtol=1e-5, diffstep=5e-3,
                 clear_wl_results=False, use_hard_coded_starting_guess=False):
        """Initialize new optimization object.

        Creates necessary folder structure if needed.

        :param set_name:
            Set name. This is used to identify the measurement set.
        :param ftol:
            Function value (difference between measured and modeled) change between iterations considered
            as 'still converging'.
            This is a stop criterion for the optimizer. Smaller value leads to more accurate result, but increases
            the optimization time. Works in tandem with xtol, so whichever value is reached first will stop
            the optimization for that wavelength.
        :param ftol_abs:
            Absolute termination condition for basin hopping. There will be no more basin hopping iterations
             if reached function value is smaller than this value. Only used if run with basin hopping algorithm,
             which can help if optimization gets caught in local minima. Basin hopping can be turned on when
             Optimization.run() is called.
        :param xtol:
            Controls how much the leaf material parameters need to change between iterations to be considered
            'progressing'. Greater value stops the optimization earlier.
        :param diffstep:
            Stepsize for finite difference Jacobian estimation. Smaller step gives
            better results, but the variables look cloudy. Big step is faster and variables
            smoother but there will be outliers in the results. Good stepsize is between 0.001 and 0.01.
        :param clear_wl_results:
            If True, clears the all wavelength results. Default is False. This is useful when running multiple
            tests with the same set name and you want to overwrite the last result. Old render are always cleared.
        :param use_hard_coded_starting_guess:
            Use hard-coded starting guess instead of the one based on polynomial fitting (the default).
            Use this only if the default starting guess is unavailable for some reason.
        """

        self.bounds = (LOWER_BOUND, UPPER_BOUND)
        # Control how much density variables (absorption and scattering density) are scaled for rendering. Value of 100 cannot
        # produce r = 0 or t = 0. Produced values do not significantly change when greater than 300.
        self.density_scale = 300

        # Verbosity levels 0, 1 or 2
        self.optimizer_verbosity = 2

        self.set_name = set_name
        self.ftol = ftol
        self.ftol_abs = ftol_abs
        self.xtol = xtol
        self.diffstep = diffstep
        self.use_hard_coded_starting_guess = use_hard_coded_starting_guess

        FH.create_first_level_folders(self.set_name)

        ids = FH.list_target_ids(self.set_name)
        for _, sample_id in enumerate(ids):
            FH.clear_rend_leaf(self.set_name, sample_id)
            FH.clear_rend_refs(self.set_name, sample_id)
            if clear_wl_results:
                FH.clear_folder(P.path_directory_subresult(self.set_name, sample_id))

    def run_optimization(self, use_threads=True, use_basin_hopping=False, resolution=1, prediction_method='surface'):
        """Runs the optimization for each sample in the set.

        It is safe to interrupt this method at any point as intermediate results are
        saved to disk and existing results are not optimized again.

        Loops through target toml files in set's target folder.

        :param use_threads:
            If True, use parallel computation on CPU.
        :param use_basin_hopping:
            If True, use basin hopping algorithm on top of the default least squares method.
            It helps in not getting stuck to local optima, but is significantly slower.
        :param resolution:
            Spectral resolution. Default value 1 will optimize all wavelengths. Value 10
            would optimize every 10th spectral band.
        :param prediction_method:
            Either 'optimization' for original method or 'surface' for new faster method.
        """

        ids = FH.list_target_ids(self.set_name)

        if len(ids) < 1:
            raise RuntimeError(f'Could not find any targets for set "{self.set_name}".')

        ids.sort()

        for _,sample_id in enumerate(ids):
            FH.create_opt_folder_structure_for_samples(self.set_name, sample_id)
            logging.info(f'Starting optimization of sample {sample_id}')
            total_time_start = time.perf_counter()
            targets = TH.read_target(self.set_name, sample_id)

            # Spectral resolution
            if resolution != 1:
                targets = targets[::resolution]

            if prediction_method == 'surface':

                start = time.perf_counter()
                wls = targets[:, 0]
                r_m = targets[:,1]
                t_m = targets[:,2]
                param_dict = TH.read_surface_model_parameters()
                ad_p = param_dict['ad']
                sd_p = param_dict['sd']
                ai_p = param_dict['ai']
                mf_p = param_dict['mf']
                ad_raw = np.clip(FF.function_exp(np.array([r_m, t_m]), *ad_p), 0.0, 1.0)
                sd_raw = np.clip(FF.function_log(np.array([r_m, t_m]), *sd_p), 0.0, 1.0)
                ai_raw = np.clip(FF.function2(np.array([r_m, t_m]), *ai_p), 0.0, 1.0)
                mf_raw = np.clip(FF.function_free(np.array([r_m, t_m]), *mf_p), 0.0, 1.0)
                ad = ad_raw * self.density_scale
                sd = sd_raw * self.density_scale
                ai = np.clip(ai_raw - 0.5, 0., 1.)
                mf = mf_raw

                with Pool() as pool:
                    n = pool._processes
                    print(f" thread count = {n}")
                    wl_chunks = GU.chunks(wls, n)
                    ad_chunks = GU.chunks(ad, n)
                    sd_chunks = GU.chunks(sd, n)
                    ai_chunks = GU.chunks(ai, n)
                    mf_chunks = GU.chunks(mf, n)

                    param_list = [(self.set_name, sample_id, wl,ad,sd,ai,mf) for wl,ad,sd,ai,mf in zip(wl_chunks, ad_chunks, sd_chunks, ai_chunks, mf_chunks)]
                    pool.map(optimize_series, param_list)

                # B.run_render_series(rend_base_path=P.path_directory_working(self.set_name, sample_id),wl=wls,ad=ad,
                #                     sd=sd,ai=ai,mf=mf,clear_rend_folder=False,clear_references=False,
                #                     render_references=True,dry_run=False)

                r = []
                t = []
                for wl in wls:
                    r_wl = DU.get_relative_refl_or_tran(C.imaging_type_refl, wl,base_path=P.path_directory_working(self.set_name, sample_id))
                    t_wl = DU.get_relative_refl_or_tran(C.imaging_type_tran, wl,base_path=P.path_directory_working(self.set_name, sample_id))
                    r.append(r_wl)
                    t.append(t_wl)

                re = np.abs(r - r_m)
                te = np.abs(t - t_m)

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
                sample_result_dict[C.key_sample_result_process_elapsed_min] = (time.perf_counter() - start) / 60.0
                sample_result_dict[C.key_sample_result_wall_clock_elapsed_min] = (time.perf_counter() - start) / 60.0
                sample_result_dict[C.key_sample_result_r_RMSE] = np.sqrt(np.mean(re ** 2))
                sample_result_dict[C.key_sample_result_t_RMSE] = np.sqrt(np.mean(te ** 2))

                TH.write_sample_result(self.set_name, sample_result_dict, sample_id)
                plotter.plot_sample_result(self.set_name, sample_id, dont_show=True, save_thumbnail=True)

                # if use_threads:
                #     param_list = [(a[0], a[1], a[2], self.set_name, self.density_scale, sample_id, ad_p, sd_p, ai_p, mf_p) for a in targets]
                #     with Pool() as pool:
                #         pool.map(run_surface_model_threaded, param_list)
                # else:
                #     for target in targets:
                #         wl = target[0]
                #         r_m = target[1]
                #         t_m = target[2]
                #         run_surface_model_wl(wl, r_m, t_m, self.set_name, self.density_scale, sample_id, ad_p, sd_p, ai_p, mf_p)

            elif prediction_method == 'optimization':

                if use_threads:
                    param_list = [(a[0], a[1], a[2], self.set_name, self.diffstep,
                               self.ftol, self.xtol, self.bounds, self.density_scale, self.optimizer_verbosity,
                               use_basin_hopping, sample_id, self.ftol_abs, self.use_hard_coded_starting_guess) for a in targets]
                    with Pool() as pool:
                        pool.map(optimize_single_wl_threaded, param_list)
                else:
                    for target in targets:
                        wl = target[0]
                        r_m = target[1]
                        t_m = target[2]
                        optimize_single_wl(wl, r_m, t_m, self.set_name, self.diffstep,
                               self.ftol, self.xtol, self.bounds, self.density_scale, self.optimizer_verbosity,
                               use_basin_hopping, sample_id, self.ftol_abs, self.use_hard_coded_starting_guess)

            else:
                raise RuntimeError(f'Prediction method "{prediction_method}" not recognized.')

            if prediction_method == 'optimization':
                logging.info(f"Finished optimizing of all wavelengths of sample {sample_id}. Saving sample result")
                elapsed_min = (time.perf_counter() - total_time_start) / 60.
                TH.make_sample_result(self.set_name, sample_id, wall_clock_time_min=elapsed_min)

        # Plot averages if there was more than one sample
        # if len(FH.list_finished_sample_ids(self.set_name)) > 1:
        plotter.plot_set_result(self.set_name)
        plotter.plot_set_errors(self.set_name)
        TH.write_set_result(self.set_name)


def optimize_series(args):
    set_name = args[0]
    sample_id = args[1]
    wls = args[2]
    ad = args[3]
    sd = args[4]
    ai = args[5]
    mf = args[6]
    # print(f"#### {wls} ####")
    B.run_render_series(rend_base_path=P.path_directory_working(set_name, sample_id),wl=wls,ad=ad,
                                    sd=sd,ai=ai,mf=mf,clear_rend_folder=False,clear_references=False,
                                    render_references=True,dry_run=False)

def optimize_single_wl_threaded(args):
    """Unpacks arguments from pool.map call."""

    optimize_single_wl(*args)


def run_surface_model_threaded(args):
    """Unpacks arguments from pool.map call."""

    run_surface_model_wl(*args)


def run_surface_model_wl(wl: float, r_m: float, t_m: float, set_name: str, density_scale, sample_id: int,
                         ad_p, sd_p, ai_p, mf_p):
    """

    :param wl: Wavelength.
    :param r_m: Measured reflectance.
    :param t_m: Measured transmittance.
    :param set_name: Set name.
    :param density_scale: Density scaling factor for Blender.
    :param sample_id: Sample id.
    :param ad_p: Absorption density parameters. Loaded from a file.
    :param sd_p: Scattering density parameters. Loaded from a file.
    :param ai_p: Scattering anisotropy parameters. Loaded from a file.
    :param mf_p: Mixing factor parameters. Loaded from a file.
    :return:
    """

    print(f'Optimizing wavelength {wl} nm started.', flush=True)
    if FH.subresult_exists(set_name, wl, sample_id):
        print(f"Subresult for sample {sample_id} wl {wl:.2f} already exists. Skipping optimization.", flush=True)
        return

    start = time.perf_counter()

    ad = np.clip(FF.function_exp(np.array([r_m, t_m]), *ad_p), 0.0, 1.0)
    sd = np.clip(FF.function_log(np.array([r_m, t_m]), *sd_p), 0.0, 1.0)
    ai = np.clip(FF.function2(np.array([r_m, t_m]), *ai_p) - 0.5, 0.0, 1.0)
    mf = np.clip(FF.function_free(np.array([r_m, t_m]), *mf_p), 0.0, 1.0)

    B.run_render_single(rend_base_path=P.path_directory_working(set_name, sample_id),
                        wl=wl,
                        ad=ad * density_scale,
                        sd=sd * density_scale,
                        ai=ai,
                        mf=mf,
                        clear_rend_folder=False,
                        clear_references=False,
                        render_references=True,
                        dry_run=False)

    r = DU.get_relative_refl_or_tran(C.imaging_type_refl, wl, base_path=P.path_directory_working(set_name, sample_id))
    t = DU.get_relative_refl_or_tran(C.imaging_type_tran, wl, base_path=P.path_directory_working(set_name, sample_id))

    elapsed = time.perf_counter() - start

    res_dict = {
        C.key_wl_result_wl: wl,
        C.key_wl_result_x0: 0,
        C.key_wl_result_x_best: [ad, sd, ai, mf],
        C.key_wl_result_refl_measured: r_m,
        C.key_wl_result_tran_measured: t_m,
        C.key_wl_result_refl_modeled: r,
        C.key_wl_result_tran_modeled: t,
        C.key_wl_result_refl_error: math.fabs(r - r_m),
        C.key_wl_result_tran_error: math.fabs(t - t_m),
        C.key_wl_result_render_calls: 1,
        C.key_wl_result_optimizer: 'surface_model',
        C.key_wl_result_optimizer_ftol: 0,
        C.key_wl_result_optimizer_xtol: 0,
        C.key_wl_result_optimizer_diffstep: 0,
        C.key_wl_result_optimizer_result: '',
        C.key_wl_result_elapsed_time_s: elapsed,
        C.key_wl_result_history_r: [r],
        C.key_wl_result_history_t: [t],
        C.key_wl_result_history_ad: [ad],
        C.key_wl_result_history_sd: [sd],
        C.key_wl_result_history_ai: [ai],
        C.key_wl_result_history_mf: [mf],
    }
    # print(res_dict)
    logging.info(f'Optimizing wavelength {wl} nm finished. Writing wavelength result and plot to disk.')

    TH.write_wavelength_result(set_name, res_dict, sample_id)

def optimize_single_wl(wl: float, r_m: float, t_m: float, set_name: str, diffstep,
                       ftol, xtol, bounds, density_scale, optimizer_verbosity,
                       use_basin_hopping: bool, sample_id: int, ftol_abs, use_hard_coded_starting_guess):
    """Optimize single wavelength to given reflectance and transmittance.

    Result is saved in a .toml file and plotted as an image.

    :param wl:
        Wavelength to be optimized.
    :param r_m:
        Measured reflectance.
    :param t_m:
        Measured transmittance.
    :param set_name:
        Set name (name of the working folder).
    :param diffstep:
        Stepsize for finite difference Jacobian estimation. Smaller step gives
        better results, but the variables look cloudy. Big step is faster and variables
        smoother but there will be outliers in the results. Good stepsize is between 0.001 and 0.01.
   :param ftol:
            Function value (difference between measured and modeled) change between iterations considered
            as 'still converging'.
            This is a stop criterion for the optimizer. Smaller value leads to more accurate result, but increases
            the optimization time. Works in tandem with xtol, so whichever value is reached first will stop
            the optimization for that wavelength.
    :param xtol:
        Controls how much the leaf material parameters need to change between iterations to be considered
        'progressing'. Greater value stops the optimization earlier.
    :param bounds:
        Bounds of the optimization problem. A tuple ([l1,l2,..], [h1,h2,...]).
    :param density_scale:
        Scaling parameter for absorption and scattering density. The values for rendering are much
        higher than the ones used in optimization.
    :param optimizer_verbosity:
        Optimizer verbosity: 0 least verbose, 2 very verbose.
    :param use_basin_hopping:
        If True, use basin hopping algorithm to escape lacal minima (reduce outliers).
        Using this considerably slows down the optimization (nearly two-fold).
    :param sample_id:
        Sample id.
    :param ftol_abs:
        Absolute termination condition for basin hopping. There will be no more basin hopping iterations
         if reached function value is smaller than this value. Only used if run with basin hopping algorithm,
         which can help if optimization gets caught in local minima. Basin hopping can be turned on when
         Optimization.run() is called.
    :param use_hard_coded_starting_guess:
            Use hard-coded starting guess instead of the one based on polynomial fitting (the default).
            Use this only if the default starting guess is unavailable for some reason.
    """

    print(f'Optimizing wavelength {wl} nm started.', flush=True)
    if FH.subresult_exists(set_name, wl, sample_id):
        print(f"Subresult for sample {sample_id} wl {wl:.2f} already exists. Skipping optimization.", flush=True)
        return

    start = time.perf_counter()
    history = []

    def distance(r, t):
        """Distance function as squared sum of errors."""

        r_diff = (r - r_m)
        t_diff = (t - t_m)
        dist = math.sqrt(r_diff * r_diff + t_diff * t_diff)
        return dist

    def f(x):
        """Function to be minimized F = sum(d_i²)."""

        B.run_render_single(rend_base_path=P.path_directory_working(set_name, sample_id),
                            wl=wl,
                            ad=x[0] * density_scale,
                            sd=x[1] * density_scale,
                            ai=x[2] - 0.5,  # for optimization from for 0 to 1, but in Blender it goes [-0.5,0.5]
                            mf=x[3],
                            clear_rend_folder=False,
                            clear_references=False,
                            render_references=False,
                            dry_run=False)

        r = DU.get_relative_refl_or_tran(C.imaging_type_refl, wl, base_path=P.path_directory_working(set_name, sample_id))
        t = DU.get_relative_refl_or_tran(C.imaging_type_tran, wl, base_path=P.path_directory_working(set_name, sample_id))
        # Debug print
        # print(f"rendering with x = {printable_variable_list(x)} resulting r = {r:.3f}, t = {t:.3f}")
        # Scale distance with the desnity scale.
        dist = distance(r, t) * density_scale
        history.append([*x, r, t])

        # Give big penalty if r+t > 1 as it is non-physical behavior.
        penalty = 0
        some_big_number = 1e6
        if r + t > 1:
            penalty = some_big_number
        return dist + penalty

    # Render references here as it only needs to be done once per wavelength
    B.run_render_single(rend_base_path=P.path_directory_working(set_name, sample_id), wl=wl, ad=0, sd=0, ai=0,
                        mf=0, clear_rend_folder=False, clear_references=False, render_references=True, dry_run=False)

    if use_hard_coded_starting_guess:
        x_0 = hard_coded_starting_guess
    else:
        x_0 = get_starting_guess(1 - (r_m + t_m))

    print(f"wl ({wl:.2f})x_0: {x_0}", flush=True)

    opt_method = 'least_squares'
    if not use_basin_hopping:
        res = optimize.least_squares(f, x_0, bounds=bounds, method='dogbox', verbose=optimizer_verbosity,
                                     gtol=None, diff_step=diffstep, ftol=ftol, xtol=xtol)
    else:
        opt_method = 'basin_hopping'

        class Stepper(object):
            """Custom stepper for basin hopping."""

            def __init__(self, stepsize=0.1):
                """Stepsize as persentage [0,1] of the bounded interval."""

                self.stepsize = stepsize

            def __call__(self, x):
                """Called by basin hopping algorithm to calculate the step length.

                Respects the bounds.
                """

                for i in range(len(x)):
                    bound_length = math.fabs(bounds[1][i] - bounds[0][i])
                    s = bound_length * self.stepsize  # max stepsizeas percentage
                    x[i] += np.random.uniform(-s, s)
                    if x[i] > bounds[1][i]:
                        x[i] = bounds[1][i]
                    if x[i] < bounds[0][i]:
                        x[i] = bounds[0][i]

                return x

        def callback(x, f_val, accepted):
            """Callback to terminate at current iteration if the function value is low enough.

            NOTE: f is the value of function f so do not call f(x) in here!
            """
            print(f'Callback value: {f_val[0]:.6f}')
            if f_val <= ftol_abs:
                print(f'Callback value: {f_val[0]:.6f} is smaller than treshold {ftol_abs:.6f}')
                return True

        def custom_local_minimizer(fun, x0, *args, **kwargs):
            """Run the default least_squares optimizer as a local minimizer for basin hopping."""

            res_lsq = optimize.least_squares(fun, x0, bounds=bounds, method='dogbox', verbose=optimizer_verbosity,
                                             gtol=None, diff_step=diffstep, ftol=ftol, xtol=xtol)
            return res_lsq

        custom_step = Stepper()
        minimizer_kwargs = {'bounds': bounds, 'options': None, 'method': custom_local_minimizer}
        res = optimize.basinhopping(f, x0=x_0, stepsize=0.1, niter=2, T=0, interval=1,
                                    take_step=custom_step, callback=callback, minimizer_kwargs=minimizer_kwargs)

    elapsed = time.perf_counter() - start

    # Render one more time with best values (in case it was not the last run)
    B.run_render_single(rend_base_path=P.path_directory_working(set_name, sample_id),
                        wl=wl,
                        ad=res.x[0] * density_scale,
                        sd=res.x[1] * density_scale,
                        ai=res.x[2] - 0.5,  # for optimization from for 0 to 1, but in Blender it goes [-0.5,0.5]
                        mf=res.x[3],
                        clear_rend_folder=False,
                        clear_references=False,
                        render_references=False,
                        dry_run=False)
    r_best = DU.get_relative_refl_or_tran(C.imaging_type_refl, wl, base_path=P.path_directory_working(set_name, sample_id))
    t_best = DU.get_relative_refl_or_tran(C.imaging_type_tran, wl, base_path=P.path_directory_working(set_name, sample_id))

    # Create wavelength result dictionary to be saved on disk.
    res_dict = {
        C.key_wl_result_wl: wl,
        C.key_wl_result_x0: x_0,
        C.key_wl_result_x_best: res.x,
        C.key_wl_result_refl_measured: r_m,
        C.key_wl_result_tran_measured: t_m,
        C.key_wl_result_refl_modeled: r_best,
        C.key_wl_result_tran_modeled: t_best,
        C.key_wl_result_refl_error: math.fabs(r_best - r_m),
        C.key_wl_result_tran_error: math.fabs(t_best - t_m),
        C.key_wl_result_render_calls: len(history),
        C.key_wl_result_optimizer: opt_method,
        C.key_wl_result_optimizer_ftol: ftol,
        C.key_wl_result_optimizer_xtol: xtol,
        C.key_wl_result_optimizer_diffstep: diffstep,
        C.key_wl_result_optimizer_result: res,
        C.key_wl_result_elapsed_time_s: elapsed,
        C.key_wl_result_history_r: [float(h[4]) for h in history],
        C.key_wl_result_history_t: [float(h[5]) for h in history],
        C.key_wl_result_history_ad: [float(h[0]) for h in history],
        C.key_wl_result_history_sd: [float(h[1]) for h in history],
        C.key_wl_result_history_ai: [float(h[2]) for h in history],
        C.key_wl_result_history_mf: [float(h[3]) for h in history],
    }
    # print(res_dict)
    logging.info(f'Optimizing wavelength {wl} nm finished. Writing wavelength result and plot to disk.')

    TH.write_wavelength_result(set_name, res_dict, sample_id)
    # Save the plot of optimization history
    # Plotter can re-create the plots from saved toml data, so there's no need to
    # run the whole optimization just to change the images.
    plotter.plot_wl_optimization_history(set_name, wl, sample_id, dont_show=True, save_thumbnail=True)


def get_starting_guess(absorption: float):
    """
    Gives starting guess for given absorption.
    """

    def f(coeffs, lb, ub):
        n = len(coeffs)
        res = 0
        for i in range(n):
            a = (n-i-1)
            res += coeffs[a] * absorption**a
        if res < lb:
            res = lb
        if res > ub:
            res = ub
        return res

    coeff_dict = TH.read_starting_guess_coeffs()
    absorption_density      = f(coeff_dict[C.ad_coeffs], LOWER_BOUND[0], UPPER_BOUND[0])
    scattering_density      = f(coeff_dict[C.sd_coeffs], LOWER_BOUND[1], UPPER_BOUND[1])
    scattering_anisotropy   = f(coeff_dict[C.ai_coeffs], LOWER_BOUND[2], UPPER_BOUND[2])
    mix_factor              = f(coeff_dict[C.mf_coeffs], LOWER_BOUND[3], UPPER_BOUND[3])
    return absorption_density, scattering_density, scattering_anisotropy, mix_factor
