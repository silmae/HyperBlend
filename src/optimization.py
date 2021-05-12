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
import scipy.optimize as optimize

from src import constants as C
from src.render_parameters import RenderParametersForSingle
from src import blender_control as B
from src import data_utils as DU
from src import file_handling as FH


def optimize_to_measured(set_name: str, r_m, t_m):
    """Optimize stuff"""

    def printable_variable_list(as_array):
        l = [f'{variable:.3f}' for variable in as_array]
        return l

    wl = 1

    def f(x):
        """Function to be minimized F = sum(d_i²)."""

        rps = RenderParametersForSingle()
        rps.clear_rend_folder = True
        rps.clear_references = False
        rps.render_references = False
        rps.dry_run = False
        rps.wl = wl
        rps.abs_dens = x[0]
        rps.scat_dens = x[1]
        rps.scat_ai = x[2]
        rps.mix_fac = x[3]
        B.run_render_single(rps, rend_base=FH.get_path_opt_working(set_name))

        r = DU.get_relative_refl_or_tran(C.imaging_type_refl, rps.wl)
        t = DU.get_relative_refl_or_tran(C.imaging_type_tran, rps.wl)
        print(f"rendering with x = {printable_variable_list(x)} resulting r = {r:.3f}, t = {t:.3f}")
        dist = math.sqrt((r - r_m)*(r - r_m) + (t-t_m) * (t-t_m))

        penalty = 0
        some_big_number = 1e6
        if r+t > 1:
            penalty = some_big_number
        return dist + penalty

    # Do this once to set render references and clear all old stuff
    rps = RenderParametersForSingle()
    rps.clear_rend_folder = True
    rps.clear_references = True
    rps.render_references = True
    rps.dry_run = False
    rps.wl = wl
    rps.abs_dens = 0
    rps.scat_dens = 0
    rps.scat_ai = 0
    rps.mix_fac = 0
    B.run_render_single(rps, rend_base=FH.get_path_opt_working(set_name))

    # initial guess
    a = 100
    b = 88
    c = 0.2
    d = 0.5
    x_0 = [a,b,c,d]

    # Bounds
    lb = [0,0,-0.5,0]
    ub = [1000,1000,0.5,1]
    bounds = (lb,ub)

    # Scale x
    x_scale = [0.01,0.01,1,1]

    res = optimize.least_squares(f, x_0, bounds=bounds, method='trf', ftol=0.01, x_scale=x_scale, verbose=2, gtol=None, diff_step=0.01)
    return res
