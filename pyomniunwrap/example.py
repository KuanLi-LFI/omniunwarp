#!/usr/bin/env python3

'''
This is an example of how to use pyomniunwrap.
An example omnidirectional image will be read. 
Two different models are used to unwrap the image.

Mei model will use cylinder unwrap to unwrap the image. 
The result will be saved in "mei.jpg".

Scara model will unwrap the image using two different methods.
The result of cylinder unwrap method will be saved in "scara.jpg".
The result of cuboid unwrap method will be saved in "cuboid.jpg" and "perspective_X.jpg".
'''

import pyomniunwrap
import cv2 as cv
import pkg_resources


def run_example():
    '''
    Example usage of how to use the pyomniunwrap package
    '''
    # File path for the yaml and image files including in the package.
    # You can replace with local file path
    example_image_path = pkg_resources.resource_filename(
        'pyomniunwrap', 'data/example.jpg')

    # Read the image
    original_img = cv.imread(example_image_path)

    # Prepare the calibration parameters
    scara_param = {
        # polynomial coefficients for the DIRECT mapping function (ocam_model.ss in MATLAB). These are used by cam2world
        "ss": [5, -1.864565e+02, 0.000000e+00, 2.919291e-03, -6.330598e-06, 8.678134e-09],
        # polynomial coefficients for the inverse mapping function (ocam_model.invpol in MATLAB). These are used by world2cam
        "invpol": [17, 323.712933, 300.441941, 91.808710, -51.905017, -80.453176, 63.731282, 130.714839, -23.557147, -133.764001, -14.408450, 91.162738, 31.999104, -32.157217, -18.885584, 3.285893, 4.131682, 0.746373],
        # center: "row" and "column", starting from 0 (C convention)
        "xy": [489.624522, 608.819613],
        # affine parameters "c", "d", "e"
        "cde": [0.998454, -0.008966, -0.008322],
        # image size: "height" and "width"
        "shape": [1000, 1230]
    }

    name = ("left", "front", "right", "back", "all")

    res, mask = pyomniunwrap.panoramic_rectify(original_img, param=scara_param)
    count = 0
    for img, msk in zip(res, mask):
        masked = cv.bitwise_and(img, img, mask=msk)
        cv.imwrite(f"{name[count]}.jpg", img)
        cv.imwrite(f"maksed_{name[count]}.jpg", masked)
        count += 1


if __name__ == "__main__":
    run_example()
