# Description
This module is a light wrapper around the Big-FISH smRA FISH spot detection module:

https://big-fish.readthedocs.io/

Goal is to automate processing and record processing parameters.
# Expected inputs
The module expects a directory containing the following files:
- Fluorescent channels: .vsi microscopy files and the corresponding vsi directories
- Cell outlines: 2D .tif files (e.g. DIC images)
- Configuration: a json file containing channel and processing parameters
# Test data
Available at: https://zenodo.org/records/14879324
Can be dowloaded using 
# Typical workflow
1. Batch process inputs to TIF files: `flofish/workflow.ipynb`
2. Find good segmentation parameters using Omnipose GUI
3. Batch segment DIC and DAPI pictures: `flofish/workflow.ipynb`
4. Find good spot detection parameters using napari-flofish plugin
   1. Load an img.json file in napari
   2. Tweak spot detection parameters
5. Batch detect spots: `flofish/workflow.ipynb`