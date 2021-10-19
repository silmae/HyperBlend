"""
Desired folder structure for optimization process:

- project_root
    - optimization
        - set_name
            - target (refl tran)
                - target.toml
            - working_temp
                - rend_leaf
                - rend_refl_ref
                - rend_tran_ref
            - result
                - final_result.toml
                - plot
                    - result_plot.jpg
                    - parameter_a.png
                    - parameter_b.png
                    - ...
                - sub_result
                    - wl_1XXX.toml
                    - wl_2XXX.toml
                    - ...

For parallel processing

1. fetch wavelengths, and target reflectance and transmittance from optimization/target/target.toml before threading
2. deside a starting guess (constant?)
3. rend references to working_temp/rend_refl_ref and rend_tran_ref
4. make a thread pool
5. run wavelength-wise optimization
    5.1 rend leaf to rend_leaf folder
    5.2 retrieve r and t
    5.3 compare to target
    5.4 finish when good enough
    5.5 save result (a,b,c,d,r,t, and metadata) to sub_reesult/wl_XX.toml
6. collect subresults to single file (add RMSE and such)
7. plot resulting (a,b,c,d,r,t) 

"""

import math
import time
import logging
import multiprocessing
from multiprocessing import Pool

import scipy.optimize as optimize
from scipy.optimize import OptimizeResult
import numpy as np

from src import constants as C, utils
from src import blender_control as B
from src import data_utils as DU
from src import file_handling as FH
from src import toml_handlling as T
from src import plotter

hard_coded_starting_guess = [0.28, 0.43, 0.27, 0.28]

class Optimization:

    def __init__(self, set_name: str, ftol=1e-2, ftol_abs=1.0, xtol=1e-5, diffstep=5e-3,
                 clear_subresults=False, use_hard_coded_starting_guess=False):
        """Initialize new optimization object.

        Creates necessary folder structure if needed.

        :param set_name:
            Set name.
        :param ftol:
            Function value change tolerance for lsq minimization
        :param ftol_abs:
            Absolute termination condition for basin hopping. Only used if run with basin hopping algorithm.
        :param xtol:
            Variable value change tolerance for lsq minimization
        :param diffstep:
            Stepsize for finite difference jacobian estimation. Smaller step gives
            better results, but the variables look cloudy. Big step is faster and variables
            smoother but there will be outliers in the results. Good stepsize is between 0.001 and 0.01.
        :param clear_subresults:
            If True, clear the subresults folder. Default is False. This is useful when running multiple
            tests with the same set name and you want to overwrite the last result.
        """

        # Bounds
        # Do not let densities (x1,x2) drop to 0 as it will result in nonphysical behavior.
        self.lb = [0.01, 0.01, 0.0, 0.0]
        self.ub = [1.0, 1.0, 1.0, 1.0]
        self.bounds = (self.lb, self.ub)
        # Control how much density variables (x1,x2) are scaled for rendering. Value of 100 cannot
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
        # TODO check these later
        ids = FH.list_target_ids(self.set_name)
        for _, sample_id in enumerate(ids):
            FH.clear_rend_leaf(self.set_name, sample_id)
            FH.clear_rend_refs(self.set_name, sample_id)
            if clear_subresults:
                FH.clear_folder(FH.get_path_opt_subresult(self.set_name, sample_id))

    def run_optimization(self, use_threads=True, use_basin_hopping=False, resolution=1):
        """Run optimization.

        Reads target reflectance and transmittance from a file.

        :param use_basin_hopping:
        :param use_threads:
            If True use parallel computation.
        :param opt_method:
            Optimization method to be used. Check implementation for available options.
        :param resolution:
            Spectral resolution. Default value 1 will optimize all wavelengths. Value 10
            would optimize every 10th spectral band.
        """


        ids = FH.list_target_ids(self.set_name)
        ids.sort()

        for _,sample_id in enumerate(ids):
            FH.create_opt_folder_structure_for_samples(self.set_name, sample_id)
            logging.info(f'Starting optimization of sample {sample_id}')
            total_time_start = time.perf_counter()
            targets = T.read_target(self.set_name, sample_id)

            # Spectral resolution
            if resolution != 1:
                targets = targets[0:-1:resolution]

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

            logging.info(f"Finished optimizing of all wavelengths of sample {sample_id}. Saving sample result")
            elapsed_min = (time.perf_counter() - total_time_start) / 60.
            self.make_sample_result(sample_id, wall_clock_time_min=elapsed_min)


    def make_sample_result(self, sample_id: int, wall_clock_time_min=0.0):
        """Collects the final result from existing subresults.

        Saves final result as toml and plotted image.

        :param wall_clock_time_min:
            Wall clock time may differ from summed subresult time if computed in parallel.
        """

        # Collect subresults
        subreslist = T.collect_subresults(self.set_name, sample_id)
        result_dict = {}

        # Set starting value to which earlier result time is added.
        result_dict[C.result_key_wall_clock_elapsed_min] = wall_clock_time_min

        try:
            previous_result = T.read_sample_result(self.set_name, sample_id)  # throws OSError upon failure
            this_result_time = result_dict[C.result_key_wall_clock_elapsed_min]
            previous_result_time = previous_result[C.result_key_wall_clock_elapsed_min]
            result_dict[C.result_key_wall_clock_elapsed_min] = this_result_time + previous_result_time
        except OSError as e:
            pass  # this is ok for the first round

        result_dict[C.result_key_process_elapsed_min] = np.sum(
            subres[C.subres_key_elapsed_time_s] for subres in subreslist) / 60.0
        result_dict[C.result_key_r_RMSE] = np.sqrt(
            np.mean(np.array([subres[C.subres_key_reflectance_error] for subres in subreslist]) ** 2))
        result_dict[C.result_key_t_RMSE] = np.sqrt(
            np.mean(np.array([subres[C.subres_key_transmittance_error] for subres in subreslist]) ** 2))
        result_dict[C.subres_key_optimizer] = subreslist[0][C.subres_key_optimizer],
        result_dict[C.subres_key_optimizer_ftol] = self.ftol,
        result_dict[C.subres_key_optimizer_xtol] = self.xtol,
        result_dict[C.subres_key_optimizer_diffstep] = self.diffstep,
        if result_dict[C.subres_key_optimizer][0] == 'basin_hopping':
            result_dict['basin_iterations_required'] = sum([(subres[C.subres_key_optimizer_result]['nit'] > 1) for subres in subreslist])
        result_dict[C.result_key_wls] = [subres[C.subres_key_wl] for subres in subreslist]
        result_dict[C.result_key_refls_modeled] = [subres[C.subres_key_reflectance_modeled] for subres in subreslist]
        result_dict[C.result_key_refls_measured] = [subres[C.subres_key_reflectance_measured] for subres in subreslist]
        result_dict[C.result_key_refls_error] = [subres[C.subres_key_reflectance_error] for subres in subreslist]
        result_dict[C.result_key_trans_modeled] = [subres[C.subres_key_transmittance_modeled] for subres in subreslist]
        result_dict[C.result_key_trans_measured] = [subres[C.subres_key_transmittance_measured] for subres in subreslist]
        result_dict[C.result_key_trans_error] = [subres[C.subres_key_transmittance_error] for subres in subreslist]

        result_dict[C.result_key_absorption_density] = [subres[C.subres_key_history_absorption_density][-1] for subres in subreslist]
        result_dict[C.result_key_scattering_density] = [subres[C.subres_key_history_scattering_density][-1] for subres in subreslist]
        result_dict[C.result_key_scattering_anisotropy] = [subres[C.subres_key_history_scattering_anisotropy][-1] for subres in subreslist]
        result_dict[C.result_key_mix_factor] = [subres[C.subres_key_history_mix_factor][-1] for subres in subreslist]

        T.write_sample_result(self.set_name, result_dict, sample_id)
        plotter.plot_sample_result(self.set_name, sample_id, dont_show=True, save_thumbnail=True)

def optimize_single_wl_threaded(args):
    """Unpacks arguments from pool.map call."""

    optimize_single_wl(*args)

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
    :param use_basin_hopping:
        If True, use basin hopping algorithm to escape lacal minima (reduce outliers).
        Using this considerably slows down the optimization (nearly two-fold).
    """

    print(f'Optimizing wavelength {wl} nm started.', flush=True)
    exists = FH.subresult_exists(set_name, wl, sample_id)
    if exists:
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

        B.run_render_single(rend_base_path=FH.get_path_opt_working(set_name, sample_id),
                            wl=wl,
                            abs_dens=x[0] * density_scale,
                            scat_dens=x[1] * density_scale,
                            scat_ai=x[2],
                            mix_fac=x[3],
                            clear_rend_folder=False,
                            clear_references=False,
                            render_references=False,
                            dry_run=False)

        r = DU.get_relative_refl_or_tran(C.imaging_type_refl, wl, base_path=FH.get_path_opt_working(set_name, sample_id))
        t = DU.get_relative_refl_or_tran(C.imaging_type_tran, wl, base_path=FH.get_path_opt_working(set_name, sample_id))
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

    B.run_render_single(rend_base_path=FH.get_path_opt_working(set_name, sample_id), wl=wl, abs_dens=0, scat_dens=0, scat_ai=0,
                        mix_fac=0, clear_rend_folder=False, clear_references=False, render_references=True,
                        dry_run=False)
    if use_hard_coded_starting_guess:
        x_0 = hard_coded_starting_guess
    else:
        x_0 = get_starting_guess(1 - (r_m + t_m))
    print(f"wl ({wl:.2f})x_0: {x_0}", flush=True)

    # Save the starting guess into history. This will not be included in any plots.
    history.append([*x_0, 0.0, 0.0])
    opt_method = 'least_squares'
    if not use_basin_hopping:
        res = optimize.least_squares(f, x_0, bounds=bounds, method='dogbox', verbose=optimizer_verbosity,
                                     gtol=None,
                                     diff_step=diffstep, ftol=ftol, xtol=xtol)

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

            res_lsq = optimize.least_squares(fun, x0, bounds=bounds, method='dogbox',
                                             verbose=optimizer_verbosity,
                                             gtol=None, diff_step=diffstep, ftol=ftol, xtol=xtol)
            return res_lsq

        custom_step = Stepper()
        minimizer_kwargs = {'bounds': bounds, 'options': None, 'method': custom_local_minimizer}
        res = optimize.basinhopping(f, x0=x_0, stepsize=0.1, niter=2, T=0, interval=1,
                                    take_step=custom_step, callback=callback, minimizer_kwargs=minimizer_kwargs)

    elapsed = time.perf_counter() - start

    # Create subresult dictionary to be saved in file.
    res_dict = {
        C.subres_key_wl: wl,
        C.subres_key_reflectance_measured: r_m,
        C.subres_key_transmittance_measured: t_m,
        C.subres_key_reflectance_modeled: history[-1][4],
        C.subres_key_transmittance_modeled: history[-1][5],
        C.subres_key_reflectance_error: math.fabs(history[-1][4] - r_m),
        C.subres_key_transmittance_error: math.fabs(history[-1][5] - t_m),
        C.subres_key_iterations: len(history) - 1,
        C.subres_key_optimizer: opt_method,
        C.subres_key_optimizer_ftol: ftol,
        C.subres_key_optimizer_xtol: xtol,
        C.subres_key_optimizer_diffstep: diffstep,
        C.subres_key_optimizer_result: res,
        C.subres_key_elapsed_time_s: elapsed,
        C.subres_key_history_reflectance: [float(h[4]) for h in history],
        C.subres_key_history_transmittance: [float(h[5]) for h in history],
        C.subres_key_history_absorption_density: [float(h[0]) for h in history],
        C.subres_key_history_scattering_density: [float(h[1]) for h in history],
        C.subres_key_history_scattering_anisotropy: [float(h[2]) for h in history],
        C.subres_key_history_mix_factor: [float(h[3]) for h in history],
    }
    # print(res_dict)
    logging.info(f'Optimizing wavelength {wl} nm finished. Writing subesult and plot to disk.')

    T.write_subresult(set_name, res_dict, sample_id)
    # Save the plot of optimization history
    # Plotter can re-create the plots from saved toml data, so there's no need to
    # run the whole optimization just to change the images.
    plotter.plot_subresult_opt_history(set_name, wl, sample_id, dont_show=True, save_thumbnail=True)

def printable_variable_list(as_array):
    l = [f'{variable:.3f}' for variable in as_array]
    return l


def get_starting_guess(absorption: float):
    """
    Gives starting guess for given absorption.
    """

    def f(coeffs):
        return coeffs[2] * absorption * absorption + coeffs[1] * absorption + coeffs[0]

    coeff_dict = T.read_starting_guess_coeffs()
    absorption_density = coeff_dict[C.ad_coeffs]
    scattering_density = coeff_dict[C.sd_coeffs]
    scattering_anisotropy = coeff_dict[C.ai_coeffs]
    mix_factor = coeff_dict[C.mf_coeffs]
    return [f(absorption_density), f(scattering_density), f(scattering_anisotropy), f(mix_factor)]
