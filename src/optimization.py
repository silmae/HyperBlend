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

import scipy.optimize as optimize


from src import constants as C
from src.render_parameters import RenderParametersForSingle
from src import blender_control as B
from src import data_utils as DU
from src import file_handling as FH
from src import toml_handlling as T

# Bounds
lb = [0, 0, -0.5, 0]
ub = [10, 10, 0.5, 1]
bounds = (lb, ub)

# Scale x
x_scale = [0.01, 0.01, 1, 1]


def init(set_name: str):
    """Create empty folders etc."""

    FH.create_opt_folder_structure(set_name)
    FH.clear_rend_leaf(set_name)
    FH.clear_rend_refs(set_name)


def run_optimization(set_name: str):
    """Run the whole thing."""

    targets = T.read_target(set_name, None)

    for target in targets:
        wl = target[0]
        r_m = target[1]
        t_m = target[2]
        history, elapsed = optimize_single_wl(wl, r_m, t_m, set_name)
        res_dict = {
            'wl': wl,
            'reflectance_measured': r_m,
            'transmittance_measured': t_m,
            'reflectance_modeled': history[-1][4],
            'transmittance_modeled': history[-1][5],
            'reflectance_error': math.fabs(history[-1][4] - r_m),
            'transmittance_error': math.fabs(history[-1][5] - t_m),
            'iterations': len(history)-1,
            'elapsed_time_s': elapsed,
            'history_reflectance': [float(h[4]) for h in history],
            'history_transmittance': [float(h[5]) for h in history],
            'history_absorption_density': [float(h[0]) for h in history],
            'history_scattering_density': [float(h[1]) for h in history],
            'history_scattering_anisotropy': [float(h[2]) for h in history],
            'history_mix_factor': [float(h[3]) for h in history],
            'history': history,
        }
        print(res_dict)

        T.write_subresult(set_name, res_dict)


def printable_variable_list(as_array):
    l = [f'{variable:.3f}' for variable in as_array]
    return l


def optimize_single_wl(wl: float, r_m: float, t_m: float, set_name: str):
    """Optimize stuff"""

    start = time.perf_counter()
    history = []

    def distance(r, t):
        dist = math.sqrt((r - r_m) * (r - r_m) + (t - t_m) * (t - t_m))
        return dist

    def f(x):
        """Function to be minimized F = sum(d_i²)."""

        rps = RenderParametersForSingle()
        rps.clear_rend_folder = True
        rps.clear_references = False
        rps.render_references = False
        rps.dry_run = False
        rps.wl = wl
        rps.abs_dens = x[0] * 100
        rps.scat_dens = x[1] * 100
        rps.scat_ai = x[2]
        rps.mix_fac = x[3]
        B.run_render_single(rps, rend_base=FH.get_path_opt_working(set_name))

        r = DU.get_relative_refl_or_tran(C.imaging_type_refl, rps.wl, base_path=FH.get_path_opt_working(set_name))
        t = DU.get_relative_refl_or_tran(C.imaging_type_tran, rps.wl, base_path=FH.get_path_opt_working(set_name))
        print(f"rendering with x = {printable_variable_list(x)} resulting r = {r:.3f}, t = {t:.3f}")
        dist = distance(r, t)
        history.append([*x, r, t])

        penalty = 0
        some_big_number = 1e6
        if r+t > 1:
            penalty = some_big_number
        return dist + penalty


    # Do this once to set render references and clear all old stuff
    rps = RenderParametersForSingle()
    rps.render_references = True
    rps.dry_run = False
    rps.wl = wl
    rps.abs_dens = 0
    rps.scat_dens = 0
    rps.scat_ai = 0
    rps.mix_fac = 0
    B.run_render_single(rps, rend_base=FH.get_path_opt_working(set_name))

    # initial guess
    a = 1
    b = 0.88
    c = 0.2
    d = 0.5
    x_0 = [a,b,c,d]
    history.append([*x_0, 0.0, 0.0])

    res = optimize.least_squares(f, x_0,  bounds=bounds, method='trf', ftol=0.01, verbose=2, gtol=None, diff_step=0.01)
    elapsed = time.perf_counter() - start
    print(res)
    return history, elapsed
