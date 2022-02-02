"""
This file passes the rendering parameters to the Blender rendering script.
"""

import os

import subprocess
import time
from sys import platform
import logging

from src import constants as C


def run_render_series(rend_base_path: str, wl, ad, sd, ai, mf,
                      clear_rend_folder=True, clear_references=True, render_references=True, dry_run=False):
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
        os.path.normpath(C.path_project_root + C.blender_scene_name),  # Blender file to be run.
        "--python",  # Execute a python script with the Blender file.
        os.path.normpath(C.path_project_root + 'bs_render_series.py'),  # Python script file to be run.
        # "--log-level", "0",

    ]

    scirpt_args = ['--']
    p = os.path.abspath(rend_base_path)
    scirpt_args += ['-p', f'{p}']
    if clear_rend_folder:
        scirpt_args += ['-c']  # c for clearing main rend folder
    if clear_references:
        scirpt_args += ['-cr']  # cr for clearing reference folders
    if render_references:
        scirpt_args += ['-r']  # render refs
    if dry_run:
        scirpt_args += ['-y']  # no render

    scirpt_args += ['-wl', f'{list(wl)}']  # wavelength to be used
    scirpt_args += ['-da', f'{list(ad)}']  # absorption density
    scirpt_args += ['-ds', f'{list(sd)}']  # scattering density
    scirpt_args += ['-ai', f'{list(ai)}']  # scattering anisotropy
    scirpt_args += ['-mf', f'{list(mf)}']  # mixing factor
    logging.info(f"running Blender with '{blender_args + scirpt_args}'")

    start = time.perf_counter()
    # Uncomment for debugging
    # logging.info(f"running Blender with '{blender_args + scirpt_args}'")

    # Direct Blender logging info to null stream to avoid cluttering of console.
    with open(os.devnull, 'wb') as stream:
        try:
            subprocess.run(blender_args + scirpt_args, stdout=stream)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Blender script argument string is too long for Windows to handle. Use less "
                                    f"wavelengths to reduce the amount of passed information. You can also try "
                                    f"running in separate batches.") from e

    seconds = time.perf_counter() - start
    print(f"Render loop run for {seconds:.1f} seconds")


def run_render_single(rend_base_path: str, wl:float, ad:float, sd:float, ai:float, mf:float,
                      clear_rend_folder=True, clear_references=True, render_references=True, dry_run=False):
    """Render an image with given values. The image is saved on disk with given wavelength in it's name.


    :param rend_base_path:
        Base path for Blender renders (set_name/working_temp/).
    :param wl:
        Wavelegth (for image name generation).
    :param ad:
        Absorption particle density.
    :param sd:
        Scattering particle density.
    :param ai:
        Scattering anisotropy. Values > 0 means forward scattering and < 0 backward scattering.
    :param mf:
        Mixing factor for absorbing and scattering shader. Value 0 means full absorption and 1 full scattering.
    :param clear_rend_folder:
        Clear main rend folder (called rend).
    :param clear_references:
        Clear reference folders (rend_refl_ref and rend_tran_ref).
    :param render_references:
        If True, render reference images. These need to be rendered only at the beginning
        of each wavelength optimization.
    :param dry_run:
        If True, Blender will not render anything but only print out some debugging stuff.
    :return:
        None
    """

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
        os.path.normpath(C.path_project_root + C.blender_scene_name),  # Blender file to be run.
        "--python",  # Execute a python script with the Blender file.
        os.path.normpath(C.path_project_root + C.blender_script_name),  # Python script file to be run.
        # "--log-level", "0",

    ]

    scirpt_args = ['--']
    p = os.path.abspath(rend_base_path)
    scirpt_args += ['-p', f'{p}']
    if clear_rend_folder:
        scirpt_args += ['-c']  # clear rend
    if clear_references:
        scirpt_args += ['-cr']  # clear refs
    if render_references:
        scirpt_args += ['-r']  # render refs
    if dry_run:
        scirpt_args += ['-y']  # no render

    scirpt_args += ['-wl', f'{wl}']  # wavelength to be used
    scirpt_args += ['-da', f'{ad}']  # absorption density
    scirpt_args += ['-ds', f'{sd}']  # scattering density
    scirpt_args += ['-ai', f'{ai}']  # scattering anisotropy
    scirpt_args += ['-mf', f'{mf}']  # mixing factor

    # Uncomment for debugging
    # logging.info(f"running Blender with '{blender_args + scirpt_args}'")

    # Direct Blender logging info to null stream to avoid cluttering of console.
    with open(os.devnull, 'wb') as stream:
        subprocess.run(blender_args + scirpt_args, stdout=stream)
