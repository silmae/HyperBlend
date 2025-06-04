"""
Parsing and generating file names.
"""

from src import constants as C


def parse_target_signal_id(filename: str) -> int:
    """Parses target signal id of a slab simulation from a given filename.

    :param filename: Filename in the format where the target id is an int
        at the end of the name, e.g. `target_123.toml`.
    :return: Parsed sample id.
    """

    sample_id = int(filename.rstrip(C.postfix_text_data_format).split("_")[-1])
    return sample_id


def parse_wl_from_filename(filename: str):
    """Parse the wavelength from result signal toml or plot file's name.

    This is only used by the optimization slab simulation material parameter solver.

    :param filename: The filename must be formatted so that the wavelength is the
        last item in the name, separated by `_wl_` and the wavelength itself can
        be cast to float, such as `result_wl_1.00.toml`.
    :return: Wavelength as float.
    """

    tail = filename.split("_wl_", 1)[1]
    wl_s = tail.rsplit(".", 1)[0]
    return float(wl_s)


def filename_wl_result(wl: float, file_type="data") -> str:
    """Generate name of a wavelength result file of given wavelength.

    Used only by slab simulation **optimizer** material parameter solver.

    :param wl: Wavelength as float. Must be accurate to 2 decimals.
    :param file_type: Type of file to generate, either "data" for toml file
        or "plot" for png.
    :return: Filename as string in format `result_wl_1.00.toml` or `result_wl_1.00.png`.
    """

    if file_type == "data":
        filename = f"result_wl_{wl:.2f}" + C.postfix_text_data_format
    elif file_type == "plot":
        filename = f"result_wl_{wl:.2f}" + C.postfix_plot_image_format
    else:
        raise ValueError(f"Unknown file type: {file_type}. Use 'data' or 'plot'.")

    return filename


def filename_target_signal(signal_id: int, resampled=False) -> str:
    """Generate filename of a target signal toml file.

    :param signal_id: Signal id.
    :param resampled: If True, file name of corresponding resampled file is
        returned instead. Default is False.
    """

    if resampled:
        filename = (
            f"{C.file_opt_target}_resampled_{signal_id}{C.postfix_text_data_format}"
        )
    else:
        filename = f"{C.file_opt_target}_{signal_id}{C.postfix_text_data_format}"
    return filename


def filename_resample_plot(sample_id: int) -> str:
    """Generate filename for resampling plot."""

    filename = f"target_{sample_id}_resampling{C.postfix_plot_image_format}"
    return filename


def filename_slab_sim_render_refl_or_tran(imaging_type: str, wl: float) -> str:
    """Returns a name for a reflectance or transmittance image of slab optimizer.

    .. warning::
        This name must match the one the rendering script uses. So if any changes
        are made in rendering, they must be reflected here as well.

    :param imaging_type: String either 'refl' for reflectance or 'tran' for
        transmittance. Use the ones listed in :mod:`src.constants`.
    :param wl: Wavelength.
    :return: The name of the image.
    """

    image_name = f"{imaging_type}_wl_{wl:.2f}{C.postfix_render_image_format}"
    return image_name


def filename_sample_result(sample_id: int) -> str:
    """Filename of the sample result toml file."""

    filename = f"{C.filename_result_signal}_{sample_id}{C.postfix_text_data_format}"
    return filename


def filename_signal_result_plot(signal_id: int) -> str:
    """Filename of the signal result plot file."""

    filename = f"signal_{signal_id}_result_plot{C.postfix_plot_image_format}"
    return filename


def filename_system_sim_spectral_cube(system_sim_name: str, file_type="data") -> str:
    """Filename of the spectral image cube of a system simulation.

    :param system_sim_name: Scene id of the system simulation.
    :param file_type: Either "data" for the image data file or "header" for the header file.
    :return: Filename of the image cube itself or the name of the header file.
    """

    if file_type == "data":
        postfix = "img"
    elif file_type == "header":
        postfix = "hdr"
    else:
        raise ValueError(f"Unknown file type: {file_type}. Use 'data' or 'header'.")

    filename = f"spectral_cube_{system_sim_name}.{postfix}"
    return filename


def filename_slab_material_csv(slab_material_name: str) -> str:
    """Spectral slab material parameters csv file name."""

    filename = f"SM_{slab_material_name}.csv"
    return filename
