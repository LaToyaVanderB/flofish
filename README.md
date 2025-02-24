# Workflow
The module expects a directory with:
- .vsi files
- the corresponding vsi directories
- DIC .tif files
# Typical workflow
1. Batch process inputs to TIF files using flofish (see workflow.ipynb)
2. Find good segmentation parameters using Omnipose GUI
3. Batch segment DIC and DAPI pictures (see workflow.ipynb)
4. Find good spot detection parameters using napari-flofish plugin
   1. Load an img.json file in napari
   2. Tweak spot detection parameters
5. Batch detect spots (see workflow.ipynb)