# Brain CT Ventricle Segmentation (OpenCV)

A lightweight rule-based pipeline for brain CT ventricle segmentation using classical computer vision techniques.

This project was implemented as a simple baseline before exploring deep learning-based medical image segmentation models such as MedSAM.

## Overview

This pipeline processes a DICOM CT volume and extracts a rough ventricle mask using Hounsfield Unit (HU) thresholding and morphological image processing.

### Pipeline
<img width="1547" height="943" alt="pipeline" src="https://github.com/user-attachments/assets/adb47b9b-77e2-4ecb-b0dc-b3109279c997" />

1. Load DICOM CT series
2. Convert pixel values to Hounsfield Units (HU)
3. Apply anisotropic diffusion filtering
4. Threshold tissues using HU ranges
5. Morphological operations
6. Connected component analysis
7. 3D connected component cleanup
8. Generate ventricle mask


## Example Result
Original CT	Segmentation
<img width="2209" height="458" alt="opencv_108" src="https://github.com/user-attachments/assets/6fb91be7-8aed-43af-a994-1ce2537ffdf0" />
<img width="2209" height="458" alt="opencv_137" src="https://github.com/user-attachments/assets/d02996e6-ddf5-4681-8d37-1ccd8101ca3c" />

MedSAM
<img width="1000" height="500" alt="MedSAM_108" src="https://github.com/user-attachments/assets/f1fc396b-aa52-4444-8fb5-1830c88845a3" />
<img width="1000" height="500" alt="MedSAM_137" src="https://github.com/user-attachments/assets/69de3fdf-cb30-4337-84fe-1705eb2ed6a4" />


## Comparison

This repository includes both:

- Classical OpenCV pipeline
- MedSAM foundation model inference

to compare rule-based and learning-based medical image segmentation approaches.
