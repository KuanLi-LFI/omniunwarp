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
    scara_yaml_path = pkg_resources.resource_filename(
        'pyomniunwrap', 'data/scara.yaml')
    mei_yaml_path = pkg_resources.resource_filename(
        'pyomniunwrap', 'data/mei.yaml')
    example_image_path = pkg_resources.resource_filename(
        'pyomniunwrap', 'data/example.jpg')

    # Read the image
    original_img = cv.imread(example_image_path)
    print(f'Origianl Image has size of {original_img.shape}')

    # Crop the image to select the region of interest
    cropped = pyomniunwrap.preprocess_img(original_img)
    print(f'Cropped Image has size of {cropped.shape}')

    # Read the model parameters
    scara = pyomniunwrap.SCARA_OCAM_MODEL(scara_yaml_path)
    mei = pyomniunwrap.MEI_OCAM_MODEL(mei_yaml_path)

    # Scara model cylinder unwrap
    res_scara = scara.panoramic_rectify(cropped, 540, 200, (400, 1800))
    # Scara model cuboid unwrap
    per_images, res_scara_cuboid = scara.cuboid_rectify(cropped)
    # Mei model cylinder unwrap
    res_mei = mei.panoramic_rectify(cropped, (2900, 800))

    # Save the unwrapped images
    cv.imwrite("scara.jpg", res_scara)
    cv.imwrite("cuboid.jpg", res_scara_cuboid)
    for index, img in enumerate(per_images):
        cv.imwrite(f"perspective_{index}.jpg", img)
    cv.imwrite("mei.jpg", res_mei)


if __name__ == "__main__":
    run_example()
