# Brain CT Ventricle Segmentation (OpenCV)

A lightweight rule-based pipeline for brain CT ventricle segmentation using classical computer vision techniques.

This project was implemented as a simple baseline before exploring deep learning-based medical image segmentation models such as MedSAM.

## Overview

This pipeline processes a DICOM CT volume and extracts a rough ventricle mask using Hounsfield Unit (HU) thresholding and morphological image processing.

### Pipeline
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



## Comparison

This repository includes both:

- Classical OpenCV pipeline
- MedSAM foundation model inference

to compare rule-based and learning-based medical image segmentation approaches.