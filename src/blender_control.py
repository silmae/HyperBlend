
import os

import subprocess
import time
from sys import platform
import logging

from src import constants as C
from src.render_parameters import RenderParametersForSeries
from src.render_parameters import RenderParametersForSingle

def run_render_series(rp: RenderParametersForSeries, rend_base: str):
    """This is mainly an utility function to plot a full wavelength series once the parameters are found."""

    bpath = C.blender_executable_path_win
    if not platform.startswith('win'):
        bpath = C.blender_executable_path_linux
        # logging.info("Running on linux machine.")
    else:
        pass
        # logging.info("Running on windows machine.")

    # Basic arguments that will always be passed on:
    blender_args = [
        bpath,
        "--background",  # Run Blender in the background.
        os.path.normpath(C.path_project_root + "leafShader.blend"),  # Blender file to be run.
        "--python",  # Execute a python script with the Blender file.
        os.path.normpath(C.path_project_root + "bs_render_series.py"),  # Python script file to be run.
        # "--log-level", "0",

    ]

    scirpt_args = ['--']
    p = os.path.abspath(rend_base)
    scirpt_args += ['-p', f'{p}']
    if rp.clear_on_start:
        scirpt_args += ['-c']  # c for clearing main rend folder
    if rp.clear_references:
        scirpt_args += ['-cr']  # cr for clearing reference folders
    if rp.render_references:
        scirpt_args += ['-r']  # render refs
    iam = rp.wl_list
    scirpt_args += ['-wl', f'{list(rp.wl_list)}']  # wavelength to be used
    scirpt_args += ['-da', f'{list(rp.abs_dens_list)}']  # absorption density
    scirpt_args += ['-ds', f'{list(rp.scat_dens_list)}']  # scattering density
    scirpt_args += ['-ai', f'{list(rp.scat_ai_list)}']  # scattering anisotropy
    scirpt_args += ['-mf', f'{list(rp.mix_fac_list)}']  # mixing factor
    # logging.info(f"running Blender with '{blender_args + scirpt_args}'")

    start = time.perf_counter()
    with open(os.devnull, 'wb') as stream:
        subprocess.run(blender_args + scirpt_args, stdout=stream)

    seconds = time.perf_counter() - start
    print(f"Render loop run for {seconds:.1f} seconds")


def run_render_single(rps: RenderParametersForSingle, rend_base: str):

    bpath = C.blender_executable_path_win
    if not platform.startswith('win'):
        bpath = C.blender_executable_path_linux
        # logging.info("Running on linux machine.")
    else:
        pass
        # logging.info("Running on windows machine.")

    # Basic arguments that will always be passed on:
    blender_args = [
        bpath,
        "--background",  # Run Blender in the background.
        os.path.normpath(C.path_project_root + "leafShader.blend"),  # Blender file to be run.
        "--python",  # Execute a python script with the Blender file.
        os.path.normpath(C.path_project_root + "bs_render_single.py"),  # Python script file to be run.
        # "--log-level", "0",

    ]

    scirpt_args = ['--']
    p = os.path.abspath(rend_base)
    scirpt_args += ['-p', f'{p}']
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
    # logging.info(f"running Blender with '{blender_args + scirpt_args}'")

    with open(os.devnull, 'wb') as stream:
        subprocess.run(blender_args + scirpt_args, stdout=stream)

