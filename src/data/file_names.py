"""
Parsing and generating file names.
"""

from src import constants as C


def get_nn_save_name(layer_count: int, layer_width: int, batch_size: int, lr: float, split:float, training_set:str) -> str:
    """Generate filename for neural network.

    This is used for the NN models and training history plot with according postfix.

    :param layer_count:
        Hidden layer count.
    :param layer_width:
        Width of hidden layers.
    :param batch_size:
        Batch size
    :param lr:
        Learning rate
    :param split:
        Persentage of data reserved to testing.
    :return:
        Returns generated name.
    """

    name = f"lc{layer_count}_lw{layer_width}_b{batch_size}_lr{lr:.3f}_split{split:.2f}.pt"
    return name


def get_surface_model_save_name(training_set_name: str) -> str:

    file_name = f"surface_params_{training_set_name}{C.postfix_text_data_format}"
    return file_name


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


def filename_wl_result_plot(wl: float) -> str:
    """File name of wavelength result plot.

    :param wl:
        Wavelength.
    :return:
        Filename as string.
    """

    filename = f"result_wl_{wl:.2f}{C.postfix_plot_image_format}"
    return filename


def filename_target(sample_id: int, resampled=False) -> str:
    """Generate filename of a toml file where target measurements are stored.

    :param sample_id:
        Sample id.
    :param resampled:
         If True, file name of corresponding resampled file is returned instead. Default is False.
    """

    if resampled:
        filename = f'{C.file_opt_target}_resampled_{sample_id}{C.postfix_text_data_format}'
    else:
        filename = f'{C.file_opt_target}_{sample_id}{C.postfix_text_data_format}'
    return filename


def filename_resample_plot(sample_id: int) -> str:
    """Generate filename resampling plot.

    :param sample_id:
        Sample id.
    """

    filename = f'target_{sample_id}_resampling{C.postfix_plot_image_format}'
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

    image_name = f"{imaging_type}_wl_{wl:.2f}{C.postfix_render_image_format}"
    return image_name


def filename_sample_result(sample_id: int) -> str:
    """Filename of the sample result toml file."""

    filename = f'{C.file_sample_result}_{sample_id}{C.postfix_text_data_format}'
    return filename


def filename_sample_result_plot(sample_id: int) -> str:
    """Filename of the sample result plot file."""

    filename = f"sample_{sample_id}_result_plot{C.postfix_plot_image_format}"
    return filename


def filename_set_result() -> str:
    """Filename of the set result file."""

    filename = 'set_result' + C.postfix_text_data_format
    return filename


def filename_set_result_plot() -> str:
    """Returns filename of set result plot. """

    filename = f"set_result_plot{C.postfix_plot_image_format}"
    return filename


def filename_set_error_plot() -> str:
    """Returns filename of set error plot. """

    filename = f"set_error_plot{C.postfix_plot_image_format}"
    return filename


def filename_forest_scene(scene_id):
    """Name of the blend file of specific forest scene."""

    filename = f"scene_forest_{scene_id}.blend"
    return filename


def filename_forest_reflectance_cube(scene_id):

    filename = f"reflectance_cube_{scene_id}.img"
    return filename


def filename_forest_reflectance_header(scene_id):

    filename = f"reflectance_cube_{scene_id}.hdr"
    return filename

def filename_leaf_material_csv(leaf_material_name: str) -> str:
    """Spectral leaf material parameters csv file name."""

    filename = f"LM_{leaf_material_name}.csv"
    return filename
