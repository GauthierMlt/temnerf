
def get_n_channels(mode):
    return {
        "1": 1,  # 1-bit pixels, black and white
        "L": 1,  # 8-bit pixels, grayscale
        "P": 1,  # 8-bit pixels, mapped to any other mode using a color palette
        "RGB": 3,  # 3x8-bit pixels, true color
        "RGBA": 4,  # 4x8-bit pixels, true color with transparency mask
        "CMYK": 4,  # 4x8-bit pixels, color separation
        "YCbCr": 3,  # 3x8-bit pixels, color video format
        "I": 1,  # 32-bit signed integer pixels
        "F": 1,  # 32-bit floating point pixels
    }.get(mode, 1)  # default to 1 channel
