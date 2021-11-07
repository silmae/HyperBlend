"""
Parsing and generating file names.
"""

from src import constants as C


def parse_sample_id(filename: str) -> int:
    """Parse sample id from given filename as listed by os.listdir().

    :param filename:
        File name.
    :return:
        Parsed sample id.
    """

    sample_id = int(filename.rstrip(C.postfix_text_data_format).split('_')[-1])
    return sample_id


def parse_wl_from_filename(filename: str):
    """Parse the wavelength from result toml or plot file's name.

    The name must be formed [refl|tran]_wl[0-9]+.[0-9]+/.*
    """

    tail = filename.split("_wl_", 1)[1]
    wl_s = tail.rsplit(".", 1)[0]
    return float(wl_s)


def filename_wl_result(wl: float) -> str:
    """Generate name of a wavelength result toml file of given wavelength.

    :param wl:
        Wavelength as float. Must be accurate to 2 decimals.
    """

    filename = f"/result_wl_{wl:.2f}" + C.postfix_text_data_format
    return filename


def filename_wl_result_plot(wl:float, file_extension='png') -> str:
    """File name of wavelength result plot.

    :param wl:
        Wavelength.
    :param file_extension:
        File extension for automatic image type detection. Default is 'png'.
    :return:
        Filename as string.
    """

    filename = f"result_wl_{wl:.2f}.{file_extension}"
    return filename


def filename_target(sample_id: int) -> str:
    """Generate filename of a toml file where target measurements are stored. """

    filename = f'{C.file_opt_target}_{sample_id}{C.postfix_text_data_format}'
    return filename


def filename_starting_guess() -> str:
    """Generates the name of the default starting guess file."""

    filename = 'default_starting_guess' + C.postfix_text_data_format
    return filename


def filename_rendered_image(imaging_type: str, wl: float) -> str:
    """Generates a name for a rendered image based on given wavelength.

    :param imaging_type:
        String either 'refl' for reflectance or 'tran' for transmittance. Use the ones listed in constants.py.
    :param wl:
        Wavelength.
    :return:
        Image name in the format that other parts of the code can understand.
    """

    image_name = f"{imaging_type}_wl_{wl:.2f}{C.postfix_image_format}"
    return image_name


def filename_final_result() -> str:
    """Filename of the final result file."""

    filename = 'final_result' + C.postfix_text_data_format
    return filename


def filename_sample_result(sample_id: int) -> str:
    """Filename of the subresult file."""

    filename = f'{C.file_sample_result}_{sample_id}{C.postfix_text_data_format}'
    return filename

