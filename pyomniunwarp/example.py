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

from pyomniunwarp import OmniUnwrap
import cv2 as cv
import pkg_resources


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

    # Initialize the model, this would take some time
    # Input the path to calib_results.txt
    unwarper = OmniUnwrap(calib_results_path)
    '''
    Input:
            left_cropped_pixel (int)    : pixel cropped on the left side from the original image
            right_cropped_pixel (int)   : pixel cropped on the right side from the original image
            Rmax (int)                  : The maximum radius from the image center in calib_results
            Rmin (int)                  : The minimum radius from the image center in calib_results
    '''
    unwarper.initialize(300, 320, 600, 235)

    # Unwarp using cylinder and cuboid projection
    # Return list of img and mask
    cyl, cyl_mask = unwarper.cylinder_rectify(original_img)
    cub, cub_mask = unwarper.cuboid_rectify(original_img)

    # Save the images
    for index, img in enumerate(cyl):
        masked = cv.bitwise_and(img, img, mask=cyl_mask[index])
        cv.imwrite(f"cylinder.jpg", img)
        cv.imwrite(f"cylinder_maksed.jpg", masked)

    name = ("left", "front", "right", "back", "full")
    for index, img in enumerate(cub):
        masked = cv.bitwise_and(img, img, mask=cub_mask[index])
        cv.imwrite(f"cuboid_{name[index]}.jpg", img)
        cv.imwrite(f"cuboid_maksed_{name[index]}.jpg", masked)


if __name__ == "__main__":
    run_example()
