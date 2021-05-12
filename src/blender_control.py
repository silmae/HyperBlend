
import os

import subprocess
import time

from src import constants as C
from src.render_parameters import RenderParametersForSeries
from src.render_parameters import RenderParametersForSingle

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


def run_render_single(rps: RenderParametersForSingle, rend_base: str):

    # Basic arguments that will always be passed on:
    blender_args = [
        C.blender_executable_path,
        "--background",  # Run Blender in the background.
        os.path.normpath(C.path_project_root + "leafShader.blend"),  # Blender file to be run.
        "--python",  # Execute a python script with the Blender file.
        os.path.normpath(C.path_project_root + "testScript.py"),  # Python script file to be run.
        "--log-level", "0",

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
    # print(scirpt_args)

    with open(os.devnull, 'wb') as stream:
        subprocess.run(blender_args + scirpt_args, stdout=stream)

