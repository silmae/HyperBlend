
import os
import numpy as np
from src.render_parameters import RenderParametersForSeries
from src import blender_control as B
from src.plotter import Plotter
from src import constants as C

def preset_test_series():
    test_base_path = os.path.abspath(C.ramdisk + '/' + 'blender_test_run')
    rpfs = RenderParametersForSeries()
    rpfs.clear_on_start = True
    rpfs.clear_references = True
    rpfs.render_references = True
    rpfs.dry_run = False
    n = 2000
    rpfs.wl_list = np.linspace(400, 1500, num=n)
    rpfs.abs_dens_list = np.linspace(1, 100, num=n)
    rpfs.scat_dens_list = np.linspace(10, 88.9, num=n)
    rpfs.scat_ai_list = np.linspace(0.25, 0.25, num=n)
    rpfs.mix_fac_list = np.linspace(0.5, 0.5, num=n)

    B.run_render_series(rpfs, rend_base=test_base_path)

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

