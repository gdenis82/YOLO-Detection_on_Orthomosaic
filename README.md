# YOLO Object Detection for Agisoft Metashape

A Python module for Agisoft Metashape Professional that enables YOLO-based object detection on orthomosaic images, shape conversion, and YOLO dataset creation.

## Overview

This module integrates the YOLO (You Only Look Once) object detection framework with Agisoft Metashape Professional, allowing users to:

1. Detect objects on orthomosaic images using pre-trained or custom YOLO models
2. Convert between different shape representations (boxes to points and vice versa)
3. Create YOLO-format datasets from Metashape data for training custom models

The module is designed to work with Agisoft Metashape Professional 2.2.0 and above, utilizing Python 3.9 and CUDA 11.8 for GPU acceleration.

## Requirements

- Agisoft Metashape Professional 2.2.0 or higher
- Python 3.9
- CUDA 11.8 (for GPU acceleration)
- The following Python packages:
  - numpy==2.0.2
  - pandas==2.2.3
  - opencv-python==4.11.0.86
  - shapely==2.0.7
  - pathlib==1.0.1
  - Rtree==1.3.0
  - tqdm==4.67.1
  - ultralytics==8.3.84
  - torch==2.6.0+cu118
  - torchvision==0.21.0+cu118
  - scikit-learn==1.6.1

## Installation

### Windows Installation

1. Update pip in the Agisoft Python environment:
   ```
   cd %programfiles%\Agisoft\python
   python.exe -m pip install --upgrade pip
   ```

2. Copy the module to the Agisoft modules directory:
   - Copy the `yolo11_detected` folder to `%programfiles%\Agisoft\modules`
   - Copy the `run_scripts.py` script to `C:/Users/<username>/AppData/Local/Agisoft/Metashape Pro/scripts/`

3. Restart Metashape and wait for the automatic installation of required packages.

4. Install CUDA-enabled PyTorch (for GPU acceleration):
   ```
   cd %programfiles%\Agisoft\python
   python.exe -m pip uninstall -y torch torchvision
   python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

5. Restart Metashape.

## Usage

After installation, three new menu items will be available in Metashape under the "Scripts" menu:

### 1. YOLO Object Detection

Access via: `Scripts > YOLO`

This tool allows you to detect objects on orthomosaic images using YOLO models. Features include:
- Detection using pre-trained or custom models
- Option to detect in specific zones or the entire orthomosaic
- Adjustable detection parameters (confidence threshold, resolution, etc.)
- Results are saved as shapes in the Metashape project

Requirements:
- An active orthomosaic with resolution ≤ 10 cm/pixel

### 2. Convert Shapes

Access via: `Scripts > Convert shapes`

This tool allows you to convert between different shape representations:
- Convert boxes to center points
- Create boxes from center points
- Adjust box size and other parameters

### 3. Create YOLO Dataset

Access via: `Scripts > Create yolo dataset`

This tool helps you create datasets in YOLO format for training custom models:
- Export orthomosaic tiles with annotations
- Support for data augmentation
- Split data into training, validation, and test sets
- Generate YAML configuration files for YOLO training

## Configuration Options

The module provides several configuration options:

- **Working Directory**: Directory for temporary files and results
- **Model Path**: Path to the YOLO model file
- **Detection Parameters**:
  - Max Image Size: Maximum size of images for processing
  - Patch Size: Size of image patches for processing
  - Resolution: Preferred resolution for detection
  - Score Threshold: Minimum confidence score for detections
- **Debug Mode**: Enable/disable debug information
- **Data Augmentation**: Options for dataset creation

## Notes

- The orthomosaic should have a resolution of 10 cm/pixel or better for optimal results
- GPU acceleration is recommended for faster processing
- Custom models can be trained using the dataset creation tool and the Ultralytics YOLO framework

## Credits

This module is based on:

- [Agisoft Metashape Scripts](https://github.com/agisoft-llc/metashape-scripts/blob/master/src/detect_objects.py)
- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)