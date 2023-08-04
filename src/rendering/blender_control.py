"""
This file passes the rendering parameters to the Blender rendering script.
"""

import os

import subprocess
import time
from sys import platform
import logging

from src import constants as C
from src.data import path_handling as PH


def _get_blender_executable_path():
    """Returns path to Blender executable file.

    Checks whether running on Windows. If not, returns the default location on Linux.

    :returns
        Path to Blender executable.
    :raises
        FileNotFoundError if cannot find the path on Windows. This will happen if Blender
        is not installed, or the version does not match the path given in `constants.py` file.
        The version should not cause problems on Linux machine.
    """

    bpath = C.blender_executable_path_win
    if not platform.startswith('win'):
        bpath = C.blender_executable_path_linux

    if not os.path.exists(bpath):
        raise FileNotFoundError(f"Could not find Blender executable from '{os.path.abspath(bpath)}'. "
                                f"Check Blender installation and set correct path to 'constants.py'. ")
    return bpath


def _get_base_blender_args(script_name: str, scene_path: str):
    """ Return basic arguments passed to Blender.

    :param script_name:
        Name of the Blender script to be run. These are found under `src/blender_scripts`.
    :param scene_path:
        Blend file to be run. For leaf simulations this is found from the main project folder.
        For forest scenes it is either the template found in  main project folder or
        a modified copy of it in scenes folder.
    :return:
        List of basic arguments for Blender scripts. Add additional arguments after '--' that are passed
        to the script itself.
    :raises
        RuntimeError if either script or scene cannot be found.
    """

    if not script_name.endswith('.py'):
        script_name = script_name + '.py'

    script_path = PH.join(PH.path_directory_blender_scripts(), script_name)
    if not os.path.exists(script_path):
        raise RuntimeError(f"Cannot find script '{script_path}'.")

    if not os.path.exists(scene_path):
        raise RuntimeError(f"Cannot find scene '{scene_path}'.")

    blender_args = [
        _get_blender_executable_path(),
        "--background",  # Run Blender in the background.
        "--python-exit-code", # Tell Blender to set exit code
        "1",                  # to 1 if the script does not execute properly.
        scene_path,  # Blender file to be run.
        "--python",  # Execute a python script with the Blender file.
        script_path,  # Python script file to be run.
        # "--log-level", "0",
    ]
    return blender_args


def run_render_series(rend_base_path: str, wl, ad, sd, ai, mf,
                      clear_rend_folder=True, clear_references=True, render_references=True, dry_run=False):
    """This is mainly an utility function to plot a full wavelength series once the parameters are found."""

    blender_args = _get_base_blender_args(script_name='bs_render_series.py', scene_path=os.path.normpath(C.path_project_root + C.blender_scene_name))

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
    logging.info(f"Render loop run for {seconds:.1f} seconds")


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

    blender_args = _get_base_blender_args(script_name=C.blender_script_name, scene_path=os.path.normpath(C.path_project_root + C.blender_scene_name))

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


def run_reflectance_lab(rend_base_path: str, dry_run=False, sun_power=None):

    blender_args = _get_base_blender_args(script_name='bs_reflectance_lab.py', scene_path=os.path.normpath(C.path_project_root + C.blender_scene_name))

    scirpt_args = ['--']
    p = os.path.abspath(rend_base_path)
    scirpt_args += ['-p', f'{p}']
    if dry_run:
        scirpt_args += ['-y']  # no render
    if sun_power is not None:
        scirpt_args += ['-s', f'{sun_power}']  # no render

    # Direct Blender logging info to null stream to avoid cluttering of console.
    with open(os.devnull, 'wb') as stream:
        subprocess.run(blender_args + scirpt_args)#, stdout=stream)


def setup_forest(scene_id, leaf_id_list=None, base_sun_power: float = None):
    """ Setup the forest for rendering.

    Currently does not do much.

    TODO setup ground
    TODO setup trees
    TODO setup materials for spectral and rgb rendering
    TODO setup sun
    TODO setup sky

    :param scene_id:
    :param leaf_id_list:
    :return:
    """

    logging.info(f"Calling forest scene setup")

    blender_args = _get_base_blender_args(script_name='bs_setup_forest.py',
                                          scene_path=PH.path_file_forest_scene(scene_id))

    scirpt_args = ['--']
    scirpt_args += ['-id', f'{scene_id}']

    if leaf_id_list is not None and len(leaf_id_list) > 0:
        scirpt_args += ['-l_ids', f'{list(leaf_id_list)}']  # available leaf indexes

    if base_sun_power is not None:
        scirpt_args += ['-sp', str(base_sun_power)]

    with open(os.devnull, 'wb') as stream:
        status = subprocess.run(blender_args + scirpt_args)#, stdout=stream)
        if status.returncode != 0:
            logging.fatal(f"Failed to setup forest scene file.")
            exit(1)


def render_forest(scene_id: str, render_mode: str):
    """Render forest preview images.

    TODO consider folder structure.. maybe rend/spectral/ for HSI and previews directly to rend/
    TODO set materials to RGB mode
    TODO render map
    TODO render Drone RGB
    TODO render Walker RGB
    TODO render Sleeper RGB

    TODO consider just setting parameters for animation rendering and render from the blend file.
    TODO set materials to spectral
    TODO set camera to Drone HSI and render

    :param scene_id:
    :param render_mode:
    :return:
    """

    logging.info(f"render_forest() called, I can possibly do something.")

    scene_path = PH.path_file_forest_scene(scene_id)
    blender_args = _get_base_blender_args(script_name='bs_render_forest', scene_path=scene_path)

    scirpt_args = ['--']
    scirpt_args += ['-id', f'{scene_id}']
    scirpt_args += ['-rm', render_mode]

    with open(os.devnull, 'wb') as stream:
        subprocess.run(blender_args + scirpt_args)#, stdout=stream)

if __name__ == '__main__':
    """
    This is testing ground for Blender commands.
    """
    logging.basicConfig(level='INFO')
    from src.data import file_handling as FH
    # scene_id = FH.duplicate_forest_scene_from_template()
    scene_id = '0102231033'
    print("Hello, you have reached HyperBlend's Blender control unit!")
    # test_scene_id = "0123456789"
    # setup_forest(scene_id)
    render_forest(scene_id, render_mode='abundances')
