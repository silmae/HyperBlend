

def parse_wl_from_image_name(image_name):
    tail = image_name.split("_wl", 1)[1]
    wl_s = tail.rsplit(".", 1)[0]
    return float(wl_s)
