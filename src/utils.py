"""
Utility functions for all scripts in the project.

"""

def parse_wl_from_image_name(image_name):
    """Parse the wavelength from an image name (filename).

    The name must be formed [refl|tran]_wl[0-9]+.[0-9]+/.*
    """

    tail = image_name.split("_wl", 1)[1]
    wl_s = tail.rsplit(".", 1)[0]
    return float(wl_s)
