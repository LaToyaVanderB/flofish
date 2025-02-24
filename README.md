# Description
This module is a light wrapper around the Big-FISH smRA FISH spot detection module:

https://big-fish.readthedocs.io/

The goal is to automate processing and record processing parameters.
# Expected inputs
The module expects a directory with:
- for the fluorescent channels: .vsi microscopy files and the corresponding vsi directories
- for the cell outlines: 2D .tif files (e.g. DIC images)
- a json configuration file containing channel and processing parameters
# Typical workflow
1. Batch process inputs to TIF files using flofish (see workflow.ipynb)
2. Find good segmentation parameters using Omnipose GUI
3. Batch segment DIC and DAPI pictures (see workflow.ipynb)
4. Find good spot detection parameters using napari-flofish plugin
   1. Load an img.json file in napari
   2. Tweak spot detection parameters
5. Batch detect spots (see workflow.ipynb)