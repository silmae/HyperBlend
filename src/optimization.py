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
    5.5 save result (a,b,c,d,r,t, and metadata) to sub_result/wl_XX.toml
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
from src.render_parameters import RenderParametersForSingle
from src.render_parameters import RenderParametersForSeries
from src import blender_control as B
from src import data_utils as DU
from src import file_handling as FH
from src import toml_handlling as T
from src import plotter


# Bounds
lb = [0.01, 0.01, -0.5, 0]
ub = [1, 1, 0.5, 1]
bounds = (lb, ub)
density_scale = 100

# Scale x
# x_scale = [0.01, 0.01, 1, 1]


def init(set_name: str, clear_subresults: bool):
    """Create empty folders etc."""

    FH.create_opt_folder_structure(set_name)
    FH.clear_rend_leaf(set_name)
    FH.clear_rend_refs(set_name)
    if clear_subresults:
        FH.clear_folder(FH.get_path_opt_subresult(set_name))


def run_optimization_in_batches(set_name: str, batch_n=1, opt_method='basin_hopping'):
    """
    Maybe better not to use this as it may cause some discontinuity in variable space.
    """

    wl_n = len(T.read_target(set_name))
    batch_size = int(wl_n / batch_n)
    step_size = int(wl_n/batch_size)
    for i in range(batch_n):
        selector = []
        for j in range(batch_size):
            selector.append(i+j*step_size)
        wls = T.read_target(set_name)[selector]
        print(f"Batch {i}: \n{wls}")
        run_optimization(set_name, wls, opt_method=opt_method)
    # do the last item for odd list
    if wl_n % batch_n != 0:
        selector = [wl_n-1]
        wls = T.read_target(set_name)[selector]
        print(f"Batch {i+1}: \n{wls}")
        run_optimization(set_name, wls, opt_method=opt_method)


def run_optimization(set_name: str, targets=None, use_threads=True, opt_method='basin_hopping', resolution=1,
                     bandwise=False):
    """Run optimization batch.

    Give targets as a batch. If none given, all target wls are run, except those excluded by resolution.

    :param set_name:
        Set name.
    :param targets:
        List of target wavelengths. If none given, the whole target list
        on disk is used. This is for running wavelengths in batches.
        TODO May behave incorrectly if resolution (other than 1) is given.
    :param use_threads:
        If True use parallel computation.
    :param opt_method:
        Optimization method to be used. Check implementation for available options.
    :param resolution:
        Spectral resolution. Default value 1 will optimize all wavelengths. Value 10
        would optimize every 10th spectral band.
    """

    total_time_start = time.perf_counter()

    if targets is None:
        targets = T.read_target(set_name)

    # Spectral resolution
    if resolution is not 1:
        targets = targets[0:-1:resolution]

    if not bandwise:
        optimize_spectrawise(targets=targets, set_name=set_name, opt_method=opt_method)
    else:
        if use_threads:
            cpu_count = multiprocessing.cpu_count()
            consecutive_targets_parallel = utils.chunks(targets, cpu_count)
            for target in consecutive_targets_parallel:
                param_list = [(a[0], a[1], a[2], set_name, opt_method, resolution*cpu_count) for a in target]
                with Pool() as pool:
                    pool.map(optimize_single_wl_threaded, param_list)
        else:
            for target in targets:
                wl = target[0]
                r_m = target[1]
                t_m = target[2]
                optimize_single_wl(wl, r_m, t_m, set_name, opt_method, resolution)

    logging.info("Finished optimizing of all wavelengths. Saving final result")
    elapsed_min = (time.perf_counter() - total_time_start) / 60.
    make_final_result(set_name, wall_clock_time_min=elapsed_min)


def optimize_spectrawise(targets, set_name: str, opt_method: str):
    print(f"Fake spectrawise optimization")
    print(f"target shape {targets.shape}")
    wl_list = targets[:, 0]
    rm_list = targets[:, 1]
    tm_list = targets[:, 2]
    print(f"wl_list shape {wl_list.shape}")
    print(f"rm_list shape {rm_list.shape}")
    print(f"tm_list shape {tm_list.shape}")

    n = len(wl_list)

    def RMSE(a,b):
        diff = a-b
        return np.sqrt(np.mean(diff*diff))

    def SAM(a,b):
        return np.arccos(a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def distance_spectral(r_list, t_list):
        rmse_r = RMSE(r_list, rm_list)
        rmse_t = RMSE(t_list, tm_list)
        rmse = rmse_r + rmse_t
        sam_r = SAM(r_list, rm_list)
        sam_t = SAM(t_list, tm_list)
        sam = sam_r + sam_t
        return rmse + sam

    def render_references():
        rpfs = RenderParametersForSeries()
        rpfs.clear_on_start = True
        rpfs.clear_references = True
        rpfs.render_references = True
        rpfs.dry_run = False
        rpfs.wl_list = wl_list
        rpfs.abs_dens_list = np.ones_like(wl_list)
        rpfs.scat_dens_list = np.ones_like(wl_list)
        rpfs.scat_ai_list = np.zeros_like(wl_list)
        rpfs.mix_fac_list = np.ones_like(wl_list)
        B.run_render_series(rpfs, rend_base=FH.get_path_opt_working(set_name))

    def f(x):
        rpfs = RenderParametersForSeries()
        rpfs.clear_on_start = True
        rpfs.clear_references = False
        rpfs.render_references = False
        rpfs.dry_run = False
        rpfs.wl_list = wl_list
        rpfs.abs_dens_list = x[0:n] * density_scale
        rpfs.scat_dens_list = x[n:2*n] * density_scale
        rpfs.scat_ai_list = x[2*n:3*n]
        rpfs.mix_fac_list = x[3*n:4*n]
        B.run_render_series(rpfs, rend_base=FH.get_path_opt_working(set_name))
        r_list = DU.get_relative_refl_or_tran_series(C.imaging_type_refl, rpfs.wl_list, base_path=FH.get_path_opt_working(set_name))
        t_list = DU.get_relative_refl_or_tran_series(C.imaging_type_tran, rpfs.wl_list, base_path=FH.get_path_opt_working(set_name))
        return distance_spectral(r_list, t_list)

    render_references()
    x1 = np.ones_like(wl_list) * 0.5
    x2 = np.ones_like(wl_list) * 0.5
    x3 = np.ones_like(wl_list) * 0.2
    x4 = np.ones_like(wl_list) * 0.5
    X_0 = np.array([item for sublist in [x1,x2,x3,x4] for item in sublist])
    # dist = f(X)
    # print(dist)

    res = optimize.least_squares(f, X_0, method='trf', verbose=2, gtol=None, diff_step=0.01)
    print(res)




def optimize_single_wl_threaded(args):
    optimize_single_wl(args[0], args[1], args[2], args[3], args[4], args[5])


def optimize_single_wl(wl: float, r_m: float, t_m: float, set_name: str, opt_method: str, resolution:int):
    """Optimize stuff"""

    print(f'Optimizing wavelength {wl} nm started.', flush=True)

    if FH.subresult_exists(set_name, wl):
        print(f"Subresult for wl {wl:.2f} already exists. Skipping optimization.", flush=True)
        return

    start = time.perf_counter()
    history = []

    def distance(r, t):
        r_diff = (r - r_m)
        t_diff = (t - t_m)
        dist = math.sqrt(r_diff * r_diff + t_diff * t_diff)
        return dist

    def f(x):
        """Function to be minimized F = sum(d_i²)."""

        rps = RenderParametersForSingle()
        rps.clear_rend_folder = False
        rps.clear_references = False
        rps.render_references = False
        rps.dry_run = False
        rps.wl = wl
        rps.abs_dens = x[0] * density_scale
        rps.scat_dens = x[1] * density_scale
        rps.scat_ai = x[2]
        rps.mix_fac = x[3]
        B.run_render_single(rps, rend_base=FH.get_path_opt_working(set_name))

        r = DU.get_relative_refl_or_tran(C.imaging_type_refl, rps.wl, base_path=FH.get_path_opt_working(set_name))
        t = DU.get_relative_refl_or_tran(C.imaging_type_tran, rps.wl, base_path=FH.get_path_opt_working(set_name))
        # Debug print
        # print(f"rendering with x = {printable_variable_list(x)} resulting r = {r:.3f}, t = {t:.3f}")
        dist = distance(r, t) * density_scale
        history.append([*x, r, t])

        penalty = 0
        some_big_number = 1e6
        if r+t > 1:
            penalty = some_big_number
        return dist + penalty


    # Do this once to render references
    rps = RenderParametersForSingle()
    rps.render_references = True
    rps.clear_rend_folder = False
    rps.clear_references = False
    rps.dry_run = False
    rps.wl = wl
    rps.abs_dens = 0
    rps.scat_dens = 0
    rps.scat_ai = 0
    rps.mix_fac = 0
    B.run_render_single(rps, rend_base=FH.get_path_opt_working(set_name))

    previous_wl = wl-resolution
    if FH.subresult_exists(set_name, previous_wl):
        adjacent_result = T.read_subresult(set_name, previous_wl)
        a = adjacent_result[C.subres_key_history_absorption_density][-1]
        b = adjacent_result[C.subres_key_history_scattering_density][-1]
        c = adjacent_result[C.subres_key_history_scattering_anisotropy][-1]
        d = adjacent_result[C.subres_key_history_mix_factor][-1]
        print(f"Using result of previous wl ({previous_wl}) as a starting guess.", flush=True)
    else:
        a = 0.5
        b = 0.5
        c = 0.2
        d = 0.5


    x_0 = [a,b,c,d]
    # x_0 =  [0.21553118, 2.28501613, 0.45281115, 0.50871691]
    print(f"wl ({wl:.2f})x_0: {x_0}", flush=True)

    history.append([*x_0, 0.0, 0.0])
    seed = 123

    print(f'optimizing with {opt_method}', flush=True)
    if opt_method == 'least_squares':
        res = optimize.least_squares(f, x_0,  bounds=bounds, method='trf', verbose=2, gtol=None, diff_step=0.01)
    elif opt_method == 'shgo':
        shgo_bounds = [(b[0], b[1]) for b in zip(lb, ub)]
        res = optimize.shgo(f, shgo_bounds, iters=10, n=2, sampling_method='sobol')
        print(f'result: \n{res}', flush=True)
    elif opt_method == 'anneal':
        anneal_bounds = list(zip(lb, ub))
        res = optimize.dual_annealing(f, anneal_bounds, seed=seed, maxiter=500, maxfun=1000, initial_temp=5000,
                                      x0=x_0, restart_temp_ratio=0.9999, visit=2.1, accept=-9000)
        print(f'result: \n{res}', flush=True)
    elif opt_method == 'basin_hopping':
        class Stepper(object):

            def __init__(self, stepsize=0.1):
                self.stepsize = stepsize

            def __call__(self, x):

                for i in range(len(x)):
                    bound_length = ub[i] - lb[i]
                    s = bound_length * self.stepsize # max stepsize as percentage
                    x[i] += np.random.uniform(-s, s)
                    if x[i] > ub[i]:
                        x[i] = ub[i]
                    if x[i] < lb[i]:
                        x[i] = lb[i]

                return x
        f_tol = 0.001
        def callback(x, f, accepted):
            print("####Callback message########")
            print(x)
            print(f)
            print(accepted)
            print("############################")
            if f <= f_tol:
                return True

        def custom_local_minimizer(fun, x0, args=(), maxfev=None, stepsize=0.1, maxiter=100, callback=None, **options):
            res_lsq = optimize.least_squares(fun, x0, bounds=bounds, method='trf', verbose=2,
                                         gtol=None, diff_step=0.01, max_nfev=150)
            return res_lsq

        custom_step = Stepper()
        minimizer_options = None
        minimizer_kwargs = {'bounds': bounds, 'options': minimizer_options, 'method': custom_local_minimizer}
        res = optimize.basinhopping(f, x0=x_0, stepsize=1., niter=20, T=0.1, interval=5, niter_success=10, seed=seed,
                                    take_step=custom_step, callback=callback, minimizer_kwargs=minimizer_kwargs)
        print(f'basing hopping result: \n{res}', flush=True)
    else:
        raise Exception(f"Optimization method '{opt_method}' not recognized.")
    elapsed = time.perf_counter() - start

    # TODO temporary solution: add one extra element to the end of the history
    # to place the best value at the end even if used optimizer does not converge
    f(res.x)

    res_dict = {
        C.subres_key_wl: wl,
        C.subres_key_reflectance_measured: r_m,
        C.subres_key_transmittance_measured: t_m,
        C.subres_key_reflectance_modeled: history[-1][4],
        C.subres_key_transmittance_modeled: history[-1][5],
        C.subres_key_reflectance_error: math.fabs(history[-1][4] - r_m),
        C.subres_key_transmittance_error: math.fabs(history[-1][5] - t_m),
        C.subres_key_iterations: len(history) - 1,
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

    T.write_subresult(set_name, res_dict)
    # Save the plot of optimization history
    # Plotter can re-create the plots from saved toml data, so there's no need to
    # run the whole optimization just to change the images.
    plotter.plot_subresult_opt_history(set_name, wl, save_thumbnail=True, dont_show=True)


def make_final_result(set_name:str, wall_clock_time_min=0.0):
    """
    :param set_name:
        Set name.
    :param wall_clock_time_min:
        Wall clock time may differ from summed subresult time if computed in parallel.
    """

    # Collect subresults
    subreslist = T.collect_subresults(set_name)
    result_dict = {}

    # Set starting value to which earlier result time is added.
    result_dict[C.result_key_wall_clock_elapsed_min] = wall_clock_time_min

    try:
        previous_result = T.read_final_result(set_name)  # throws OSError upon failure
        this_result_time = result_dict[C.result_key_wall_clock_elapsed_min]
        previous_result_time = previous_result[C.result_key_wall_clock_elapsed_min]
        result_dict[C.result_key_wall_clock_elapsed_min] = this_result_time + previous_result_time
    except OSError as e:
        pass  # this is ok for the first round

    result_dict[C.result_key_process_elapsed_min] = np.sum(subres[C.subres_key_elapsed_time_s] for subres in subreslist) / 60.0
    result_dict[C.result_key_r_RMSE] = np.sqrt(np.mean(np.array([subres[C.subres_key_reflectance_error] for subres in subreslist])**2))
    result_dict[C.result_key_t_RMSE] = np.sqrt(np.mean(np.array([subres[C.subres_key_transmittance_error] for subres in subreslist])**2))
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

    T.write_final_result(set_name, result_dict)
    plotter.plot_final_result(set_name, save_thumbnail=True, dont_show=True)


def printable_variable_list(as_array):
    l = [f'{variable:.3f}' for variable in as_array]
    return l
