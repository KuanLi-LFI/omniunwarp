# Overview
This is a python package using Scaramuzza and Mei's method to rectify omnidirectional images

## Prerequisite

```
$ pip install opencv-python
```

If you want to test the Mei's model, also install

```
$ pip install opencv-contrib-python
```

Make sure opencv-python and opencv-contrib-python have the same version

## Installation

```
$ pip install pyomniunwrap
```

## Usage

To run the example in the terminal
```
$ python example.py
```
or in python
```
import pyomniunwrap.example

pyomniunwrap.example.run_example()
```

To use the calibrated model in python, prepare the calibration parameters as the following format. Calibration result can be obtained by [OCamCalib toolbox](https://sites.google.com/site/scarabotix/ocamcalib-omnidirectional-camera-calibration-toolbox-for-matlab)

```
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
```

Then in python, use as
```
import pyomniunwrap

res_scara, mask_scara = pyomniunwrap.panoramic_rectify(
        original_img, param=scara_param)
```

The returns are perspective images (left, front, right, back, all) and corresponding masks.
The first call to the unwrap operation would take longer time for computing the mapping matrices, the following calls would be faster reusing the maps.