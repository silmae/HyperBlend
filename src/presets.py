
import os
import time
import numpy as np
from src.render_parameters import RenderParametersForSeries
from src.render_parameters import RenderParametersForSingle
from src import blender_control as B
from src.plotter import Plotter
from src import constants as C


rpfs_time_test = RenderParametersForSeries()
rpfs_time_test.clear_on_start = True
rpfs_time_test.clear_references = True
rpfs_time_test.render_references = True
rpfs_time_test.dry_run = False
n_time_test = 50
rpfs_time_test.wl_list = np.linspace(400, 1500, num=n_time_test)
rpfs_time_test.abs_dens_list = np.linspace(1, 100, num=n_time_test)
rpfs_time_test.scat_dens_list = np.linspace(10, 88.9, num=n_time_test)
rpfs_time_test.scat_ai_list = np.linspace(0.25, 0.25, num=n_time_test)
rpfs_time_test.mix_fac_list = np.linspace(0.5, 0.5, num=n_time_test)
path_base_time_test = C.ramdisk
time_test_base_path = os.path.abspath(path_base_time_test + '/' + 'blender_test_run')

def preset_test_time_bandwise():
    print(f"Running bandwise test for n={n_time_test}...")
    start = time.perf_counter()
    for i,wl in enumerate(rpfs_time_test.wl_list):
        rps = RenderParametersForSingle()
        if i == 0:
            rps.clear_rend_folder = True
            rps.clear_references = True
        else:
            rps.clear_rend_folder = False
            rps.clear_references = False
        rps.render_references = rpfs_time_test.render_references
        rps.dry_run = rpfs_time_test.dry_run
        rps.wl = wl
        rps.abs_dens = rpfs_time_test.abs_dens_list[i]
        rps.scat_dens = rpfs_time_test.scat_dens_list[i]
        rps.scat_ai = rpfs_time_test.scat_ai_list[i]
        rps.mix_fac = rpfs_time_test.mix_fac_list[i]
        B.run_render_single(rps, rend_base=time_test_base_path)
    seconds = time.perf_counter() - start
    print(f"Test loop time  {seconds:.1f} seconds")

def preset_test_time_spectrawise():
    print(f"Running spectrawise test for n={n_time_test}...")
    start = time.perf_counter()
    B.run_render_series(rpfs_time_test, rend_base=time_test_base_path)
    seconds = time.perf_counter() - start
    print(f"Test loop time  {seconds:.1f} seconds")

def preset_make_steady_scat():
    rpfs = RenderParametersForSeries()
    rpfs.clear_on_start = True
    rpfs.clear_references = True
    rpfs.render_references = True
    rpfs.dry_run = False
    n = 10
    rpfs.wl_list = np.linspace(400, 1500, num=n)
    rpfs.abs_dens_list = np.linspace(1, 1, num=n)
    rpfs.scat_dens_list = np.linspace(80, 90, num=n)
    rpfs.scat_ai_list = np.linspace(0.25, 0.25, num=n)
    rpfs.mix_fac_list = np.linspace(1.0, 1.0, num=n)

    B.run_render_series(rpfs)
    plotter = Plotter(rpfs, x_label='Scattering density', x_values=rpfs.scat_dens_list)
    dump = plotter.dump()
    plotter.plot_wl_series(silent=False)


def preset_make_varying_abs():
    rpfs = RenderParametersForSeries()
    rpfs.clear_on_start = True
    rpfs.clear_references = True
    rpfs.render_references = True
    rpfs.dry_run = False
    n = 10
    rpfs.wl_list = np.linspace(400, 1500, num=n)
    rpfs.abs_dens_list = np.linspace(1, 100, num=n)
    rpfs.scat_dens_list = np.linspace(88.9, 88.9, num=n)
    rpfs.scat_ai_list = np.linspace(0.25, 0.25, num=n)
    rpfs.mix_fac_list = np.linspace(0.5, 0.5, num=n)

    B.run_render_series(rpfs)
    plotter = Plotter(rpfs, x_label='Absorption density', x_values=rpfs.abs_dens_list)
    dump = plotter.dump()
    plotter.plot_wl_series(silent=False)

def preset_make_varying_mix():
    rpfs = RenderParametersForSeries()
    rpfs.clear_on_start = True
    rpfs.clear_references = True
    rpfs.render_references = True
    rpfs.dry_run = False
    n = 10
    rpfs.wl_list = np.linspace(400, 1500, num=n)
    rpfs.abs_dens_list = np.linspace(50, 50, num=n)
    rpfs.scat_dens_list = np.linspace(88.9, 88.9, num=n)
    rpfs.scat_ai_list = np.linspace(0.25, 0.25, num=n)
    rpfs.mix_fac_list = np.linspace(0.0, 1.0, num=n)

    B.run_render_series(rpfs)
    plotter = Plotter(rpfs, x_label='Mix factor', x_values=rpfs.mix_fac_list)
    dump = plotter.dump()
    plotter.plot_wl_series(silent=False)

def preset_make_varying_ai():
    rpfs = RenderParametersForSeries()
    rpfs.clear_on_start = True
    rpfs.clear_references = True
    rpfs.render_references = True
    rpfs.dry_run = False
    n = 10
    rpfs.wl_list = np.linspace(400, 1500, num=n)
    rpfs.abs_dens_list = np.linspace(1, 1, num=n)
    rpfs.scat_dens_list = np.linspace(88.9, 88.9, num=n)
    rpfs.scat_ai_list = np.linspace(-0.35, 0.35, num=n)
    rpfs.mix_fac_list = np.linspace(1.0, 1.0, num=n)

    B.run_render_series(rpfs)
    plotter = Plotter(rpfs, x_label='Scattering anisotropy', x_values=rpfs.scat_ai_list)
    dump = plotter.dump()
    plotter.plot_wl_series(silent=False)

