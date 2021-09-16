"""
Some quick test.

TODO Out of date!!
"""

import os
import time
import numpy as np
from src import blender_control as B
# from src.plotter import Plotter
from src import constants as C



clear_on_start = True
clear_references = True
render_references = True
dry_run = False
n_time_test = 50
wl_list = np.linspace(400, 1500, num=n_time_test)
abs_dens_list = np.linspace(1, 100, num=n_time_test)
scat_dens_list = np.linspace(10, 88.9, num=n_time_test)
scat_ai_list = np.linspace(0.25, 0.25, num=n_time_test)
mix_fac_list = np.linspace(0.5, 0.5, num=n_time_test)
path_base_time_test = C.path_project_root
time_test_base_path = os.path.abspath(path_base_time_test + '/' + 'blender_test_run')

def preset_test_time_bandwise():
    print(f"Running bandwise test for n={n_time_test}...")
    start = time.perf_counter()
    for i,wl in enumerate(wl_list):
        if i == 0:
            clear_rend_folder = True
            clear_references = True
        else:
            clear_rend_folder = False
            clear_references = False
        B.run_render_single(rend_base_path=time_test_base_path,
                            wl=wl,
                            abs_dens=abs_dens_list[i],
                            scat_dens=scat_dens_list[i],
                            scat_ai=scat_ai_list[i],
                            mix_fac=mix_fac_list[i],
                            clear_rend_folder=clear_rend_folder,
                            clear_references=clear_references,
                            render_references=render_references,
                            dry_run=dry_run)
    seconds = time.perf_counter() - start
    print(f"Test loop time  {seconds:.1f} seconds")


# def preset_make_steady_scat():
#     rpfs = RenderParametersForSeries()
#     rpfs.clear_on_start = True
#     rpfs.clear_references = True
#     rpfs.render_references = True
#     rpfs.dry_run = False
#     n = 10
#     rpfs.wl_list = np.linspace(400, 1500, num=n)
#     rpfs.abs_dens_list = np.linspace(1, 1, num=n)
#     rpfs.scat_dens_list = np.linspace(80, 90, num=n)
#     rpfs.scat_ai_list = np.linspace(0.25, 0.25, num=n)
#     rpfs.mix_fac_list = np.linspace(1.0, 1.0, num=n)
#
#     B.run_render_series(rpfs)
#     plotter = Plotter(rpfs, x_label='Scattering density', x_values=rpfs.scat_dens_list)
#     dump = plotter.dump()
#     plotter.plot_wl_series(silent=False)
#
#
# def preset_make_varying_abs():
#     rpfs = RenderParametersForSeries()
#     rpfs.clear_on_start = True
#     rpfs.clear_references = True
#     rpfs.render_references = True
#     rpfs.dry_run = False
#     n = 10
#     rpfs.wl_list = np.linspace(400, 1500, num=n)
#     rpfs.abs_dens_list = np.linspace(1, 100, num=n)
#     rpfs.scat_dens_list = np.linspace(88.9, 88.9, num=n)
#     rpfs.scat_ai_list = np.linspace(0.25, 0.25, num=n)
#     rpfs.mix_fac_list = np.linspace(0.5, 0.5, num=n)
#
#     B.run_render_series(rpfs)
#     plotter = Plotter(rpfs, x_label='Absorption density', x_values=rpfs.abs_dens_list)
#     dump = plotter.dump()
#     plotter.plot_wl_series(silent=False)
#
# def preset_make_varying_mix():
#     rpfs = RenderParametersForSeries()
#     rpfs.clear_on_start = True
#     rpfs.clear_references = True
#     rpfs.render_references = True
#     rpfs.dry_run = False
#     n = 10
#     rpfs.wl_list = np.linspace(400, 1500, num=n)
#     rpfs.abs_dens_list = np.linspace(50, 50, num=n)
#     rpfs.scat_dens_list = np.linspace(88.9, 88.9, num=n)
#     rpfs.scat_ai_list = np.linspace(0.25, 0.25, num=n)
#     rpfs.mix_fac_list = np.linspace(0.0, 1.0, num=n)
#
#     B.run_render_series(rpfs)
#     plotter = Plotter(rpfs, x_label='Mix factor', x_values=rpfs.mix_fac_list)
#     dump = plotter.dump()
#     plotter.plot_wl_series(silent=False)
#
# def preset_make_varying_ai():
#     rpfs = RenderParametersForSeries()
#     rpfs.clear_on_start = True
#     rpfs.clear_references = True
#     rpfs.render_references = True
#     rpfs.dry_run = False
#     n = 10
#     rpfs.wl_list = np.linspace(400, 1500, num=n)
#     rpfs.abs_dens_list = np.linspace(1, 1, num=n)
#     rpfs.scat_dens_list = np.linspace(88.9, 88.9, num=n)
#     rpfs.scat_ai_list = np.linspace(-0.35, 0.35, num=n)
#     rpfs.mix_fac_list = np.linspace(1.0, 1.0, num=n)
#
#     B.run_render_series(rpfs)
#     plotter = Plotter(rpfs, x_label='Scattering anisotropy', x_values=rpfs.scat_ai_list)
#     dump = plotter.dump()
#     plotter.plot_wl_series(silent=False)

