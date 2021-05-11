import math
import matplotlib.pyplot as plt
import os
import sys       # to get command line args
import logging
import numpy as np
import argparse  # to parse options for us and print a nice help message

import subprocess
import time
import toml
import scipy as sc
import scipy.optimize as optimize

from src import file_handling as FH
from src import constants as C



def get_relative_refl_or_tran(imaging_type: str, wl: float):
    """Returns leaf reflectance (transmittance) divided by reference reflectance (transmittance). """

    leaf_mean = get_rend_as_mean(FH.search_by_wl(C.target_type_leaf, imaging_type, wl))
    reference_mean = get_rend_as_mean(FH.search_by_wl(C.target_type_ref, imaging_type, wl))
    relative = leaf_mean / reference_mean
    return relative


def get_rend_as_mean(image_file_path: os.path) -> float:
    array = get_rend_as_ndarray_wl(image_file_path)
    return np.mean(array)


def get_rend_as_ndarray_wl(image_file_path: os.path):
    if os.path.exists(image_file_path):
        array = plt.imread(image_file_path)
        return array
    else:
        raise Exception(f"Image {image_file_path} does not exist.")


# TODO Revive this method later for a series of renders
# def collect_rends(rend_path):
#     refl_image_count = 0
#     tran_image_count = 0
#     refl_wls = []
#     tran_wls = []
#     for filename in os.listdir(os.path.normpath(rend_path)):
#         if filename.endswith(image_type):
#             found_image_path = os.path.normpath(rend_path + '/' + filename)
#             if filename.startswith(reflectance_prefix):
#                 refl_image_count += 1
#                 tail = filename.split("_wl", 1)[1]
#                 wl_s = tail.split(".", 1)[0]
#                 wl_int = int(wl_s)
#                 refl_wls.append(wl_int)
#                 logging.info(f"Reflectance image '{found_image_path}' found")
#                 continue
#             elif filename.startswith(transmittance_prefix):
#                 tran_image_count += 1
#                 tail = filename.split("_wl", 1)[1]
#                 wl_s = tail.split(".", 1)[0]
#                 wl_int = int(wl_s)
#                 tran_wls.append(wl_int)
#                 logging.info(f"Transmittance image '{found_image_path}' found")
#                 continue
#             else:
#                 logging.warning(f"File '{os.path.join(filename)}' not recognised as reflectance or transmittance image.")
#                 continue
#         else:
#             logging.info(f"Non-image file '{os.path.normpath(rend_path + '/' + filename)}' found")
#             continue
#
#     if refl_image_count is not tran_image_count:
#         raise Exception(f"Reflectance and transmittance image count mismatch ({refl_image_count} / {tran_image_count}).")
#
#     if refl_wls != tran_wls:
#         raise Exception(f"Wavelength lists of reflectance and transmittance are not equal."
#                         f"Check file naming and parser code.")
#     else:
#         wls = refl_wls
#         del refl_wls
#         del tran_wls
#
#     # Everything should now be in order so we can extract the actual image data.
#
#     print(f"Found reflectance and transmittance data with {len(wls)} wavelengths. Proceeding to load images...", end='')
#
#     refl_images = []
#     tran_images = []
#     for filename in os.listdir(os.path.normpath(rend_path)):
#         if filename.endswith(image_type):
#             found_image_path = os.path.normpath(rend_path + '/' + filename)
#             if filename.startswith(reflectance_prefix):
#                 refl_images.append(plt.imread(found_image_path))
#                 continue
#             elif filename.startswith(transmittance_prefix):
#                 tran_images.append(plt.imread(found_image_path))
#                 continue
#             else:
#                 continue
#         else:
#             continue
#
#     refl_images = np.array(refl_images)
#     tran_images = np.array(tran_images)
#
#     ## Sorting, because the files are not necessarily accessed in wavelegth order.
#     wls = np.array(wls)
#     refl_images = np.array(refl_images)
#     tran_images = np.array(tran_images)
#
#     inds = wls.argsort()
#     wls = wls[inds]
#     refl_images = refl_images[inds]
#     tran_images = tran_images[inds]
#
#     print("done.")
#     return refl_images, tran_images, wls

def plot_mean_refl_tran(relf_list, tran_list, wls):
    # mean_refl = refl_images / refl_ref_images
    # mean_tran = tran_images / tran_ref_images
    # mean_refl = np.mean(mean_refl, axis=(1,2))
    # mean_tran = np.mean(mean_tran, axis=(1,2))

    refl_plot_handle = plt.plot(wls, relf_list, label="Reflectance")
    tran_plot_handle = plt.plot(wls, 1 - tran_list, label="Transmittance")
    plt.legend()
    plt.ylim([0,1])
    plt.show()

class RenderParametersForSeries:

    clear_on_start = True
    clear_references = True
    render_references = True
    dry_run = False
    wl_list = None
    abs_dens_list = None
    scat_dens_list = None
    scat_ai_list = None
    mix_fac_list = None


    def get_as_dict(self):
        d = {
            'clear_on_start': self.clear_on_start,
            'clear_references': self.clear_references,
            'render_references': self.render_references,
            'dry_run': self.dry_run,
            'wl_list': self.wl_list,
            'abs_dens_list': self.abs_dens_list,
            'scat_dens_list': self.scat_dens_list,
            'scat_ai_list': self.scat_ai_list,
            'mix_fac_list': self.mix_fac_list,
        }
        return d

    def get_single(self, i):
        if i >= len(self.wl_list):
            raise IndexError(f"Index {i} out of bounds for list of {len(self.wl_list)}.")
        else:
            rps = RenderParametersForSingle()
            # rps.clear_rend_folder = self.clear_on_start
            rps.clear_references = self.clear_references
            rps.render_references = self.render_references
            rps.dry_run = self.dry_run
            rps.wl = self.wl_list[i]
            rps.abs_dens = self.abs_dens_list[i]
            rps.scat_dens = self.scat_dens_list[i]
            rps.scat_ai = self.scat_ai_list[i]
            rps.mix_fac = self.mix_fac_list[i]
            return rps



class RenderParametersForSingle:

    clear_rend_folder = True
    clear_references = True
    render_references = True
    dry_run = False

    wl = None
    abs_dens = None
    scat_dens = None
    scat_ai = None
    mix_fac = None


def run_render_series(rp: RenderParametersForSeries):
    """This is mainly an utility function to plot a full wavelength series once the parameters are found."""

    start = time.perf_counter()
    for i,wl in enumerate(rp.wl_list):

        rps = rp.get_single(i)

        if i == 0 and rp.clear_on_start:
            # scirpt_args += ['-c']  # c for clearing main rend folder
            rps.clear_rend_folder = True
        if i == 0 and rp.clear_references:
            # scirpt_args += ['-cr']  # cr for clearing reference folders
            rps.clear_references = True

        run_render_single(rps)

    seconds = time.perf_counter() - start
    print(f"Render loop run for {seconds:.1f} seconds")


def run_render_single(rps: RenderParametersForSingle):

    # Basic arguments that will always be passed on:
    blender_args = [
        C.blender_executable_path,
        "--background",  # Run Blender in the background.
        os.path.normpath(C.project_root_path + "leafShader.blend"),  # Blender file to be run.
        "--python",  # Execute a python script with the Blender file.
        os.path.normpath(C.project_root_path + "testScript.py"),  # Python script file to be run.
        "--log-level", "0",

    ]

    scirpt_args = ['--']
    if rps.clear_rend_folder:
        scirpt_args += ['-c']  # clear rend
    if rps.clear_references:
        scirpt_args += ['-cr']  # clear refs
    if rps.render_references:
        scirpt_args += ['-r']  # render refs


    scirpt_args += ['-wl', f'{rps.wl}']  # wavelength to be used
    scirpt_args += ['-da', f'{rps.abs_dens}']  # absorption density
    scirpt_args += ['-ds', f'{rps.scat_dens}']  # scattering density
    scirpt_args += ['-ai', f'{rps.scat_ai}']  # scattering anisotropy
    scirpt_args += ['-mf', f'{rps.mix_fac}']  # mixing factor

    with open(os.devnull, 'wb') as stream:
        subprocess.run(blender_args + scirpt_args, stdout=stream)

class Plotter:

    def __init__(self, rp: RenderParametersForSeries, x_label: str, x_values):
        self.rp = rp
        self.x_label = x_label
        self.x_values = x_values
        self.plot_folder = os.path.normpath(C.project_root_path + 'plot')
        x_ndvalues = np.array(x_values)
        self.filename = f"{x_label}_{x_ndvalues.min():.1f}-{x_ndvalues.max():.1f}"
        self.r = None
        self.t = None

        refl_list = []
        tran_list = []
        # Get data according to wavelengths
        for i, wl in enumerate(self.rp.wl_list):
            refl_list.append(get_relative_refl_or_tran(C.imaging_type_refl, wl))
            tran_list.append(get_relative_refl_or_tran(C.imaging_type_tran, wl))

        self.r = np.array(refl_list)
        self.t = np.array(tran_list)

    def dump(self):
        if not os.path.exists(self.plot_folder):
            os.makedirs(self.plot_folder)
        d = self.rp.get_as_dict()
        d['x_label'] = self.x_label
        d['x_values'] = self.x_values
        d['reflectance'] = self.r
        d['transmittance'] = self.t
        path = os.path.normpath(f"{self.plot_folder}/{self.filename}.toml")
        with open(path, 'w') as f:
            toml.dump(d, f, encoder=toml.TomlNumpyEncoder())
        return d

    def plot_wl_series(self, invert_tran_y=True, save_thumbnail=True, silent=False):
        refl_color = 'blue'
        tran_color = 'orange'

        fig, axr = plt.subplots()

        axr.set_xlabel(self.x_label)
        axr.set_ylabel('Reflectance', color=refl_color)
        axr.tick_params(axis='y', labelcolor=refl_color)
        # But use given x_values for plotting
        axr.plot(self.x_values, self.r, label="Reflectance", color=refl_color)

        axt = axr.twinx()
        axt.set_ylabel('Transmittance', color=tran_color)
        axt.tick_params(axis='y', labelcolor=tran_color)
        axt.plot(self.x_values, self.t, label="Transmittance", color=tran_color)

        axr.set_ylim([0, 1])
        if invert_tran_y:
            axt.set_ylim([1, 0])
        else:
            axt.set_ylim([0, 1])
        # plt.legend()

        if save_thumbnail:
            if not os.path.exists(self.plot_folder):
                os.makedirs(self.plot_folder)
            path = os.path.normpath(f"{self.plot_folder}/{self.filename}.png")
            plt.savefig(path)

        if not silent:
            plt.show()

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

    run_render_series(rpfs)
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

    run_render_series(rpfs)
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

    run_render_series(rpfs)
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

    run_render_series(rpfs)
    plotter = Plotter(rpfs, x_label='Scattering anisotropy', x_values=rpfs.scat_ai_list)
    dump = plotter.dump()
    plotter.plot_wl_series(silent=False)


def optimize_to_measured(r_m, t_m):
    """Optimize stuff"""

    def printable_variable_list(as_array):
        l = [f'{variable:.3f}' for variable in as_array]
        return l

    wl = 0

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
        run_render_single(rps)

        r = get_relative_refl_or_tran(C.imaging_type_refl, rps.wl)
        t = get_relative_refl_or_tran(C.imaging_type_tran, rps.wl)
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
    run_render_single(rps)

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

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    # rpfs = RenderParametersForSeries()
    # rpfs.clear_on_start = True
    # rpfs.clear_references = True
    # rpfs.render_references = True
    # rpfs.dry_run = False
    # n = 10
    # rpfs.wl_list = np.linspace(400, 1500, num=n)
    # rpfs.abs_dens_list = np.linspace(100, 100, num=n)
    # rpfs.scat_dens_list = np.linspace(76, 78, num=n)
    # rpfs.scat_ai_list = np.linspace(0.09, 0.09, num=n)
    # rpfs.mix_fac_list = np.linspace(0.5, 0.5, num=n)
    #
    # # run_render_series(rpfs)
    # plotter = Plotter(rpfs, x_label='mix factor', x_values=rpfs.mix_fac_list)
    # dd = plotter.dump()
    # plotter.plot_wl_series(silent=True)
    # preset_make_varying_mix()

    r_m = 0.4
    t_m = 0.4
    optimize_to_measured(r_m=r_m,t_m=t_m)

    r = get_relative_refl_or_tran(C.imaging_type_refl, 0)
    t = get_relative_refl_or_tran(C.imaging_type_tran, 0)
    print(f"Final reflectance: {r} ({r_m})")
    print(f"Final transmittance: {t} ({t_m})")


    # print(f"reflectance = {get_relative_refl_or_tran(imaging_type_refl, wl)}")
    # print(f"transmittance = {get_relative_refl_or_tran(imaging_type_tran, wl)}")

    # refl_images, tran_images, wls = collect_rends(rend_path_leaf)
    # refl_ref_images, _, _ = collect_rends(rend_path_refl_ref)
    # _, tran_ref_images, _ = collect_rends(rend_path_tran_ref)
    # plot_mean_refl_tran(refl_images, tran_images, refl_ref_images, tran_ref_images, wls)
