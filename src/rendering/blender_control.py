"""
This script passes the rendering parameters to the Blender rendering scripts.
"""

import logging
import os
import subprocess
import time
from sys import platform

from src import constants as C
from src.data import path_handling as PH
from src.setup.runtime_environment import RuntimeEnvironment


def _get_blender_executable_path(runtime: RuntimeEnvironment):
    """Returns a path to a Blender executable file.

    :param runtime: The runtime environment object which contains the path to the
        Blender executable.
    :return: Path to Blender executable.
    :raises FileNotFoundError: if none of the paths actually contain the executable.
        This can happen if there is no Blender installed or none of the installed
        versions is compatible.
    """

    if runtime.blender_executable_path is not None:
        return runtime.blender_executable_path
    else:
        logging.warning(
            "Could not get Blender executable path from RuntimeEnvironment. "
            "Using default Blender executable path from constants.py file as a fallback."
        )

        bpath = C.blender_executable_path_win
        if not platform.startswith("win"):
            bpath = C.blender_executable_path_linux

        if not os.path.exists(bpath):
            raise FileNotFoundError(
                f"Could not find Blender executable from '{os.path.abspath(bpath)}'. "
                f"Check Blender installation and set correct path to 'constants.py'. "
            )
    return bpath


def _get_base_blender_args(
    script_name: str, scene_path: str, runtime: RuntimeEnvironment
):
    """Return basic arguments passed to Blender.

    :param script_name: Name of the Blender script to be run. These are found
        under `src/blender_scripts`.
    :param scene_path: Blend file to be run. For slab simulations this is found from the
        main project directory. For system_simulation scenes it is either the template found
        in `root/Internal` or a modified copy of it in `Slab simulation` directory.

    :return: List of basic arguments for Blender scripts. Add additional arguments after
        '--' that are passed to the script itself.

    :raises FileNotFoundError: if either the script or the Blender scene cannot be found.
    """

    if not script_name.endswith(".py"):
        script_name = script_name + ".py"

    script_path = PH.join(PH.directory_code_blender_scripts(), script_name)
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Cannot find script '{script_path}'.")

    if not os.path.exists(scene_path):
        raise FileNotFoundError(f"Cannot find scene '{scene_path}'.")

    blender_args = [
        _get_blender_executable_path(runtime=runtime),
        "--background",  # Run Blender in the background.
        "--python-exit-code",  # Tell Blender to set exit code
        "1",  # to 1 if the script does not execute properly.
        scene_path,  # Blender file to be run.
        "--python",  # Execute a python script with the Blender file.
        script_path,  # Python script file to be run.
        "--log-level",
        "0",
    ]
    return blender_args


def run_script(
    script_name: str,
    scene_path: str,
    runtime: RuntimeEnvironment,
    script_args: list[str],
    silent=True,
):
    """Runs a Blender script with given arguments.

    :param script_name: Name of the Blender script to be run.
    :param scene_path: Path to the Blender scene file that is used by the script.
    :param runtime: Runtime environment object that contains the Blender executable path.
    :param script_args: List of arguments to be passed to the Blender script.
    :param silent: If True, redirect Blender output to null stream to avoid cluttering of console.
    """

    blender_args = _get_base_blender_args(
        script_name=script_name,
        scene_path=scene_path,
        runtime=runtime,
    )

    full_args = blender_args + script_args

    logging.info(f"running Blender with following argument list:\n'{full_args}'")

    with open(os.devnull, "wb") as stream:
        # If silent is True, redirect Blender output to null stream to avoid cluttering of console.
        if silent:
            exit_code = subprocess.run(full_args, stdout=stream)
        else:
            exit_code = subprocess.run(full_args)

        if exit_code.returncode != 0:
            logging.fatal(
                f"Blender script '{script_name}' failed to run. Check the arguments passed to it."
            )
            exit(1)


def run_parallel_slab_wl_render(
    runtime: RuntimeEnvironment,
    rend_base_path: str,
    wl,
    ad,
    sd,
    ai,
    mf,
    clear_rend_folder=True,
    clear_references=True,
    render_references=True,
    dry_run=False,
    silent=True,
):
    """Runs a Blender script that renders a slab simulation for multiple wavelengths.

    This is used by the optimization solver :mod:`slab_model.opt`.

    :param runtime: Runtime environment object that contains the Blender executable path.
    :param rend_base_path: Base path for Blender renders. This is usually
        :func:`data.path_handling.directory_slab_optimization_working`
    :param wl: List of wavelengths to be used in the rendering.
    :param ad: List of absorption densities to be used in the rendering.
    :param sd: List of scattering densities to be used in the rendering.
    :param ai: List of scattering anisotropies to be used in the rendering.
    :param mf: List of mixing factors to be used in the rendering.
    :param clear_rend_folder: If True, the main render folder will be cleared before rendering.
    :param clear_references: If True, the reference folders will be cleared before rendering.
    :param render_references: If True, reference images will be rendered.
    :param dry_run: If True, nothing is rendered and just print out some debugging information.
    :silent: If True, Blender output is redirected to null stream to avoid cluttering of console.
    """

    scirpt_args = ["--"]
    p = os.path.abspath(rend_base_path)
    scirpt_args += ["-p", f"{p}"]
    if clear_rend_folder:
        scirpt_args += ["-c"]  # c for clearing main rend folder
    if clear_references:
        scirpt_args += ["-cr"]  # cr for clearing reference folders
    if render_references:
        scirpt_args += ["-r"]  # render refs
    if dry_run:
        scirpt_args += ["-y"]  # no render

    scirpt_args += ["-wl", f"{list(float(x) for x in wl)}"]  # wavelength to be used
    scirpt_args += ["-da", f"{list(float(x) for x in ad)}"]  # absorption density
    scirpt_args += ["-ds", f"{list(float(x) for x in sd)}"]  # scattering density
    scirpt_args += ["-ai", f"{list(float(x) for x in ai)}"]  # scattering anisotropy
    scirpt_args += ["-mf", f"{list(float(x) for x in mf)}"]  # mixing factor

    start = time.perf_counter()

    # Direct Blender logging info to `os.devnull` null stream to avoid cluttering of console.
    with open(os.devnull, "wb") as stream:
        try:
            run_script(
                script_name="bs_render_series.py",
                scene_path=PH.file_blend_slab_simulation_template(),
                runtime=runtime,
                script_args=scirpt_args,
                silent=silent,
            )
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Blender script argument string is too long for Windows to handle. Use less "
                f"wavelengths to reduce the amount of passed information. You can also try "
                f"running in separate batches."
            ) from e

    seconds = time.perf_counter() - start
    logging.info(f"Running parallel wavelength render took {seconds:.1f} seconds")


def run_slab_wl_render(
    runtime: RuntimeEnvironment,
    rend_base_path: str,
    wl: float,
    ad: float,
    sd: float,
    ai: float,
    mf: float,
    clear_rend_folder=True,
    clear_references=True,
    render_references=True,
    dry_run=False,
    silent=True,
):
    """Renders a single image of the slab simulation with given slab material parameters.

    This is used by the optimization solver :mod:`slab_model.opt`.

    :param runtime: Runtime environment object that contains the Blender executable path.
    :param rend_base_path: Base path for Blender renders. This is usually
        :func:`data.path_handling.directory_slab_optimization_working`
    :param wl: List of wavelengths to be used in the rendering.
    :param ad: List of absorption densities to be used in the rendering.
    :param sd: List of scattering densities to be used in the rendering.
    :param ai: List of scattering anisotropies to be used in the rendering.
    :param mf: List of mixing factors to be used in the rendering.
    :param clear_rend_folder: If True, the main render folder will be cleared before rendering.
    :param clear_references: If True, the reference folders will be cleared before rendering.
    :param render_references: If True, reference images will be rendered.
    :param dry_run: If True, nothing is rendered and just print out some debugging information.
    :param silent: If True, Blender output is redirected to null stream to avoid
        cluttering of console.
    """

    scirpt_args = ["--"]
    p = os.path.abspath(rend_base_path)
    scirpt_args += ["-p", f"{p}"]
    if clear_rend_folder:
        scirpt_args += ["-c"]  # clear rend
    if clear_references:
        scirpt_args += ["-cr"]  # clear refs
    if render_references:
        scirpt_args += ["-r"]  # render refs
    if dry_run:
        scirpt_args += ["-y"]  # no render

    scirpt_args += ["-wl", f"{wl:.10f}"]  # wavelength to be used
    scirpt_args += ["-da", f"{ad:.10f}"]  # absorption density
    scirpt_args += ["-ds", f"{sd:.10f}"]  # scattering density
    scirpt_args += ["-ai", f"{ai:.10f}"]  # scattering anisotropy
    scirpt_args += ["-mf", f"{mf:.10f}"]  # mixing factor

    logging.error(f"Running slab wavelength render with arguments:\n{scirpt_args}")

    run_script(
        script_name="bs_render_single.py",
        scene_path=PH.file_blend_slab_simulation_template(),
        runtime=runtime,
        script_args=scirpt_args,
        silent=silent,
    )


def run_reflectance_lab(
    runtime: RuntimeEnvironment,
    rend_base_path: str,
    dry_run=False,
    sun_power=None,
    silent=True,
):
    """Runs a Blender script that renders the reflectance lab scene.

    .. warning:: This method is very much not tested. Should test so it can be used
        for diffuse reflective surfaces [10.6.2025].

    :param runtime: Runtime environment object that contains the Blender executable path.
    :param rend_base_path: Base path for Blender renders.
    :param dry_run: If True, nothing is rendered and just print out some debugging information.
    :param sun_power: If not None, sets the sun power in the scene.
    :param silent: If True, Blender output is redirected to null stream to avoid cluttering of console.
    """

    scirpt_args = ["--"]
    p = os.path.abspath(rend_base_path)
    scirpt_args += ["-p", f"{p}"]
    if dry_run:
        scirpt_args += ["-y"]  # no render
    if sun_power is not None:
        scirpt_args += ["-s", f"{sun_power}"]  # no render

    run_script(
        script_name="bs_reflectance_lab.py",
        # scene_path=PH.file_blend_slab_simulation_template(), # this is the old line
        scene_path=PH.file_blend_reflectance_lab_template(),  # TODO check if this works as should
        runtime=runtime,
        script_args=scirpt_args,
        silent=silent,
    )


def generate_forest_control(
    runtime: RuntimeEnvironment,
    system_sim_name: str = None,
    global_master: bool = False,
):
    """Generates a system_simulation control file for forest simulation by reading
    parameters from a Blender file.

    .. note::
        Even if there are no usages for this method, do not remove it. It is used to
        generate the system_simulation control file from the scene template.

    :param runtime: Runtime environment object that contains the Blender executable path.
    :param system_sim_name: ID of the system_simulation to create the control file for.
        Can be None only if ``global_master == True``.
    :param global_master: If True, the global master control file is updated based on the parameters
        in system_simulation template file. The result is saved to the project root directory.
    :raises AttributeError: if either ``global_master == False`` and ``scene_id == None``,
        because there is nothing to be done OR if ``global_master == True`` and
        ``scene_id is not None``, because the caller might expect something else to
        happen than rewriting of the global master control.
    """

    if not global_master and system_sim_name is None:
        raise AttributeError(
            f"If global_master == False, a scene_id must be provided. Was None."
        )

    if global_master:
        system_sim_name_to_use = C.filename_system_sim_forest_template
        scene_path = PH.file_blend_system_simulation_template()
    else:
        if system_sim_name is None:
            raise AttributeError(
                f"If global_master == False, scene_id must be provided."
            )
        system_sim_name_to_use = system_sim_name
        scene_path = PH.file_blend_system_simulation(system_sim_name_to_use)

    scirpt_args = ["--"]
    scirpt_args += ["-id", f"{system_sim_name_to_use}"]
    if global_master:
        scirpt_args += ["-g"]

    run_script(
        script_name="bs_configuration.py",
        scene_path=scene_path,
        runtime=runtime,
        script_args=scirpt_args,
        silent=False,
    )


def setup_system_sim_scene(
    runtime: RuntimeEnvironment, system_sim_name: str, leaf_name_list=None
):
    """Set up the system_simulation for rendering.

    TODO: Refactor this when the material names in the system simulation scene are changed
        into more general slab material names and diffuse material names.

    :param runtime: Runtime environment object that contains the Blender executable path.
    :param system_sim_name: Name of the system simulation to be set up.
    :param leaf_name_list: Names of the leaf materials (must mach the ones used in the
        Blender file) as a list of strings like: ['Leaf material 1', 'Leaf material 2',...].
    """

    scirpt_args = ["--"]
    scirpt_args += ["-id", f"{system_sim_name}"]

    if leaf_name_list is not None and len(leaf_name_list) > 0:
        scirpt_args += ["-l_ids", f"{list(leaf_name_list)}"]  # available leaf indexes

    run_script(
        script_name="bs_setup_forest.py",
        scene_path=PH.file_blend_system_simulation(system_sim_name),
        runtime=runtime,
        script_args=scirpt_args,
        silent=False,
    )


def render_forest(
    runtime: RuntimeEnvironment, system_sim_name: str, render_mode: str, silent=True
):
    """Render different presentations of the forest scene.

    :param runtime: Runtime environment object that contains the Blender executable path.
    :param system_sim_name: System simulation name.
    :param render_mode: One of the following 'preview', 'spectral' or 'visibility'.
        'preview' renders only some preview images that can give an idea of the
        scene geometry without having to open the Blender file itself.
        'spectral' renders all spectral channels as a single image.
        'visibility' renders visibility maps that show which object is visible
        in each pixel.
    :param silent: If True, Blender output is redirected to null stream to avoid
        cluttering of console.
    """

    scirpt_args = ["--"]
    scirpt_args += ["-id", f"{system_sim_name}"]
    scirpt_args += ["-rm", render_mode]

    run_script(
        script_name="bs_render_forest.py",
        scene_path=PH.file_blend_system_simulation(system_sim_name),
        runtime=runtime,
        script_args=scirpt_args,
        silent=False,
    )
