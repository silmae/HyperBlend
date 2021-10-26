import os

import subprocess
import time
from sys import platform
import logging

from src import constants as C


def run_render_single(rend_base_path: str, wl:float, abs_dens:float, scat_dens:float, scat_ai:float, mix_fac:float,
                      clear_rend_folder=True, clear_references=True, render_references=True, dry_run=False):
    """Render an image with given values. The image is saved on disk with given wavelength in it's name.


    :param rend_base_path:
        Base path for Blender renders (set_name/working_temp/).
    :param wl:
        Wavelegth (for image name generation).
    :param abs_dens:
        Absorption particle density.
    :param scat_dens:
        Scattering particle density.
    :param scat_ai:
        Scattering anisotropy. Values > 0 means forward scattering and < 0 backward scattering.
    :param mix_fac:
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
    scirpt_args += ['-da', f'{abs_dens}']  # absorption density
    scirpt_args += ['-ds', f'{scat_dens}']  # scattering density
    scirpt_args += ['-ai', f'{scat_ai}']  # scattering anisotropy
    scirpt_args += ['-mf', f'{mix_fac}']  # mixing factor
    # Uncomment for debugging
    # logging.info(f"running Blender with '{blender_args + scirpt_args}'")

    with open(os.devnull, 'wb') as stream:
        subprocess.run(blender_args + scirpt_args, stdout=stream)
