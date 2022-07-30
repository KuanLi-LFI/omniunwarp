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

from pyomniunwarp import OmniUnwarp
import cv2 as cv
import pkg_resources
from pathlib import Path


def run_example():
    '''
    Example usage of how to use the pyomniunwrap package
    '''
    # File path for the parameter and image files including in the package.
    # You can replace with local file path
    example_image_path = pkg_resources.resource_filename(
        'pyomniunwarp', 'data/example.jpg')
    calib_results_path = pkg_resources.resource_filename(
        'pyomniunwarp', 'data/calib_results.txt')

    # Read the image
    original_img = cv.imread(example_image_path)

    # Prepare the parameters and calib_results.txt
    # Define the mode here
    kwargs = {
        "mode": "cuboid",   # cuboid or cylinder
        "version": "0.2.1",  # https://pypi.org/project/pyomniunwarp/
        "calib_results_path": calib_results_path
    }

    # Initialize the model, this would take some time
    unwarper = OmniUnwarp(**kwargs)

    # New API
    imgs, masks, labels = unwarper.rectify(original_img)

    # Old API, call the rectify function seperately
    # Unwarp using cylinder and cuboid projection
    # Return list of img, mask, and label
    cyl, cyl_mask, cyl_label = unwarper.cylinder_rectify(original_img)
    cub, cub_mask, cub_label = unwarper.cuboid_rectify(original_img)

    # Save the images
    for index, img in enumerate(imgs):
        masked = cv.bitwise_and(img, img, mask=masks[index])
        cv.imwrite(f"{labels[index]}.jpg", masked)


if __name__ == "__main__":
    run_example()
