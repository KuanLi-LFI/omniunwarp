# Overview
This is a python package using Scaramuzza model to rectify omnidirectional images

## Prerequisite

```
$ pip install opencv-python
```

## Installation

```
$ pip install pyomniunwarp
```

## Parameters

The initialize function take 4 parameters

left_cropped_pixel (int)    : pixel cropped on the left side from the original image

right_cropped_pixel (int)   : pixel cropped on the right side from the original image

Rmax (int)                  : The maximum radius from the image center in calib_results

Rmin (int)                  : The minimum radius from the image center in calib_results

## Usage

To run the example in python
```
import pyomniunwarp.example

pyomniunwarp.example.run_example()
```

To use the calibrated model in python, prepare `calib_results.txt` from [OCamCalib toolbox](https://sites.google.com/site/scarabotix/ocamcalib-omnidirectional-camera-calibration-toolbox-for-matlab)

Put `calib_results.txt` under the same folder with the python script

Then in python, import as
```
from pyomniunwarp import OmniUnwrap

unwarper = OmniUnwrap()
unwarper.initialize(left_cropped_pixel, right_cropped_pixel, Rmax, Rmin)
```

Initialize will take several seconds. After the initializtion, perform unwarping by

```
cyl, cyl_mask = unwarper.cylinder_rectify(original_img)
cub, cub_mask = unwarper.cuboid_rectify(original_img)
```