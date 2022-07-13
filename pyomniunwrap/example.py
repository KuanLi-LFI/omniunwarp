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
import yaml


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

    # Read the model parameters
    with open(mei_yaml_path, 'r') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    K = data['K']
    D = data['D']
    Xi = data['Xi']

    with open(scara_yaml_path, 'r') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    ss = data['ss']
    invpol = data['invpol']
    shape = data['shape']
    cde = data['cde']
    xy = data['xy']

    # Unwrap the images
    res_scara, mask_scara = pyomniunwrap.panoramic_rectify(
        original_img, model='scara', ss=ss, invpol=invpol, shape=shape, cde=cde, xy=xy)
    masked = cv.bitwise_and(res_scara, res_scara, mask=mask_scara)
    cv.imwrite("scara.jpg", res_scara)
    cv.imwrite("scara_masked.jpg", masked)

    res_mei, mask_mei = pyomniunwrap.panoramic_rectify(
        original_img, model='mei', K=K, D=D, Xi=Xi)
    masked = cv.bitwise_and(res_mei, res_mei, mask=mask_mei)
    cv.imwrite("mei.jpg", res_mei)
    cv.imwrite("mei_masked.jpg", masked)


if __name__ == "__main__":
    run_example()
