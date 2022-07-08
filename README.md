# Overview
This is a python package using Scaramuzza and Mei's method to rectify omnidirectional images

## Prerequisite

```
$ pip install opencv-python
$ pip install opencv-contrib-python
$ pip install pyyaml
```

Make sure opencv-python and opencv-contrib-python have the same version

## Installation

```
$ pip install pyomniunwrap
```

## Usage

To see and run the example in the terminal
```
$ python example.py
```
or in python
```
import pyomniunwrap.example

pyomniunwrap.example.run_example()
```

To use the calibrated model in python
```
import pyomniunwrap

scara = pyomniunwrap.SCARA_OCAM_MODEL("path to scara.yaml")
perspective_images, full_image = scara.cuboid_rectify(omni_image)
```