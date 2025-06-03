# Automated Solution for Detection, Segmentation, and Classification of Objects on Orthophotoplans Using YOLO Neural Network in Agisoft Metashape

## Introduction

This project is dedicated to developing a solution for automated detection, segmentation, and classification of animals on orthophotoplans obtained from aerial photography of northern sea lion and northern fur seal rookeries. The task of object detection is universal for various types of images; differences are manifested in recognition accuracy and technical means used at different stages of data processing.

The project uses aerial photographic materials obtained during surveys with unmanned aerial vehicles. Processing of images, including combining overlapping images to build orthophotoplans, is a labor-intensive and resource-intensive process, the automation of which is currently only partially implemented.

As one of the key components, a built-in module for the Agisoft Metashape software product has been developed, implementing functions for automatic search, segmentation, and classification of animals. To solve the problem, the architecture of the convolutional neural network YOLO (You Only Look Once) is applied, providing high performance and acceptable accuracy with limited computational resources.

## Operating Principle of the Main Module Functions

### 1. Function detect_objects()

The `detect_objects()` function is the main interface for starting the process of detecting objects on an orthophotoplan. The principle of operation of the function:

1. **Input Data Verification**:
   - Checks for the presence of an active orthophotoplan in the current Metashape project.
   - Checks the resolution of the orthophotoplan (should be no more than 10 cm/pixel).

2. **Creating a User Interface**:
   - Creates an instance of the `MainWindowDetect` class, which provides a graphical interface for configuring detection parameters.

3. **Detection Process** (method `detect()` of the `MainWindowDetect` class):
   - **Dividing the orthophotoplan into tiles**: The orthophotoplan is divided into large tiles, which are then divided into smaller subtiles for processing. This allows processing large orthophotoplans without loading them entirely into memory.
   - **Processing each subtile using YOLO**: Each subtile is processed by the YOLO model, which returns detected objects with their bounding boxes and, if available, segmentation masks.
   - **Processing overlapping areas**: A non-maximum suppression (NMS) algorithm is applied to eliminate duplicate detections in overlapping areas of adjacent tiles.
   - **Coordinate transformation**: The coordinates of detected objects are transformed from pixel coordinates of the tile to world coordinates of the orthophotoplan.
   - **Creating vector objects**: For each detected object, a vector object (polygon) is created in Metashape with corresponding attributes (class, confidence).

4. **Visualization of Results**:
   - Detected objects are displayed as vector layers in Metashape.
   - Two layers are created: one for bounding boxes and one for segmentation contours.

### 2. Function convert_shapes()

The `convert_shapes()` function is designed to convert between different representations of vector objects. The principle of operation of the function:

1. **Input Data Verification**:
   - Checks for the presence of an active orthophotoplan in the current Metashape project.
   - Checks the resolution of the orthophotoplan (should be no more than 10 cm/pixel).

2. **Creating a User Interface**:
   - Creates an instance of the `WindowConvertShapes` class, which provides a graphical interface for configuring conversion parameters.

3. **Conversion Process** (method `run_convert_shapes()` of the `WindowConvertShapes` class):
   - Depending on the selected mode, either `create_boxes_from_points()` or `create_points_from_boxes()` is called.

4. **Converting Points to Bounding Boxes** (method `create_boxes_from_points()`):
   - For each point from the source layer:
     - Extracts the label (class) of the point.
     - Converts the point coordinates to the orthophotoplan coordinate system.
     - If zones are specified, checks if the point is inside any zone.
   - Creates bounding boxes around points with a given size.
   - Bounding boxes are added as polygonal objects to a new Metashape layer.

5. **Converting Bounding Boxes to Points** (method `create_points_from_boxes()`):
   - For each bounding box from the source layer:
     - Extracts the label (class) of the box.
     - Calculates the center of the box.
     - If zones are specified, checks if the center is inside any zone.
   - The centers of the boxes are added as point objects to a new Metashape layer.

### 3. Function create_yolo_dataset()

The `create_yolo_dataset()` function is designed to create a dataset for training YOLO models based on an orthophotoplan and vector objects. The principle of operation of the function:

1. **Input Data Verification**:
   - Checks for the presence of an active orthophotoplan in the current Metashape project.
   - Checks the resolution of the orthophotoplan (should be no more than 10 cm/pixel).

2. **Creating a User Interface**:
   - Creates an instance of the `WindowCreateYoloDataset` class, which provides a graphical interface for configuring dataset creation parameters.

3. **Dataset Creation Process** (method `create_on_user_data()` of the `WindowCreateYoloDataset` class):
   - **Processing Training Zones**:
     - For each training zone, the optimal transformation between world coordinates and pixel coordinates is found.
     - The bounding box of each zone in pixel coordinates is calculated.
     - Checks if the zone is large enough for training.

   - **Processing Annotations**:
     - For each annotation within the training zones:
       - Converts the coordinates of the annotation vertices to orthophotoplan coordinates.
       - Checks if the annotation overlaps sufficiently with the zone.
       - Assigns a class identifier for each unique label.

   - **Dividing Zones into Tiles**:
     - Each zone is divided into tiles of fixed size.
     - For each tile:
       - Image data is extracted from the orthophotoplan.
       - Tiles that are outside the orthophotoplan are skipped.
       - Annotations that overlap with the tile are found.

   - **Applying Data Augmentation** (if enabled):
     - Various transformations (rotations, reflections) are applied to each tile.
     - For each transformed tile, the coordinates of annotations are recalculated.

   - **Saving Data in YOLO Format**:
     - Images are saved in JPEG format.
     - Annotations are saved in YOLO format (normalized coordinates).
     - The directory structure required for YOLO is created.
     - A data.yaml file with class information is created.

   - **Splitting into Training and Validation Sets**:
     - The dataset is randomly split into training and validation sets according to the specified ratio.

   - **Visualization of Annotations** (in debug mode):
     - Visualizations of annotations are created to check the correctness of the transformation.

## Technical Implementation Details

### Processing Large Orthophotoplans

Orthophotoplans created from aerial photography materials can be very large (tens of thousands of pixels in each dimension). For efficient processing of such images, the module uses the following approaches:

1. **Division into Tiles**: The orthophotoplan is divided into tiles of fixed size, which are processed separately.
2. **Tile Overlap**: Tiles have overlap so that objects located at the boundaries of tiles are fully visible in at least one tile.
3. **Processing Overlapping Detections**: A non-maximum suppression algorithm is applied to eliminate duplicate detections in overlapping areas.

### Coordinate Transformation

The module performs several coordinate transformations:

1. **From World Coordinates to Pixel Coordinates**: For extracting tiles from the orthophotoplan.
2. **From Pixel Coordinates of the Tile to Pixel Coordinates of the Orthophotoplan**: For combining detection results from different tiles.
3. **From Pixel Coordinates of the Orthophotoplan to World Coordinates**: For creating vector objects in Metashape.

### Data Augmentation

To improve the training of the YOLO model, the module supports various data augmentation methods:

1. **Geometric Transformations**:
   - Rotations (by 90, 180, 270 degrees)
   - Mirror reflections
   - Combinations of rotations and reflections

2. **Color Transformations**:
   - Brightness changes
   - Contrast changes
   - Saturation changes

### YOLO Data Format

The module creates a dataset in the format required for training YOLO models:

1. **Directory Structure**:
   ```
   dataset_yolo/
   ├── data.yaml
   ├── train/
   │   ├── images/
   │   └── labels/
   └── val/
       ├── images/
       └── labels/
   ```

2. **Annotation Format**:
   - For object detection:
     ```
     <class_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm>
     ```
   - For segmentation:
     ```
     <class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
     ```
   where `class_id` is the class identifier, and coordinates are normalized by image dimensions.

3. **data.yaml File**:
   ```yaml
   train: train/images
   val: val/images
   nc: <number_of_classes>
   names: [<class_name_1>, <class_name_2>, ...]
   ```

## Conclusion

The developed module for Agisoft Metashape provides a comprehensive solution for automating the process of detecting, segmenting, and classifying objects on orthophotoplans using the YOLO neural network. The module integrates into the Metashape workflow, allowing users to perform all processing stages in one program, without the need to use third-party tools.

The main advantages of the module:
- Automation of the labor-intensive process of manual marking of objects on orthophotoplans
- High performance due to efficient processing of large images
- Support for both object detection (bounding boxes) and segmentation (contours)
- Ability to create datasets for training custom YOLO models
- Integration with Metashape, providing a convenient workflow

The module can be used for various tasks related to orthophotoplan processing, including animal population monitoring, mapping, agriculture, forestry, and other areas where automatic detection and classification of objects in aerial photographs is required.