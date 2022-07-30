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

Prepare the parameters as the following format
```
kwargs = {
    "mode": "cuboid",   # cuboid or cylinder
    "version": "0.2.1",  # https://pypi.org/project/pyomniunwarp/
    "calib_results_path": calib_results_path
}
```

## Usage

To run the example in python
```
import pyomniunwarp.example

pyomniunwarp.example.run_example()
```

To use the calibrated model in python, prepare `calib_results.txt` from [OCamCalib toolbox](https://sites.google.com/site/scarabotix/ocamcalib-omnidirectional-camera-calibration-toolbox-for-matlab). Edit `calib_results.txt` as the following example [here](/pyomniunwarp/data/calib_results.txt)

Put `calib_results.txt` under the same folder with the python script

Then in python, import as
```
from pyomniunwarp import OmniUnwarp

unwarper = OmniUnwrap(**kwargs)
```

Initialize will take several seconds. After the initializtion, perform unwarping by

```
imgs, masks, labels = unwarper.rectify(original_img)
```
