# ## For Agisoft Metashape Professional 2.2.0
# - python 3.9
# - cuda 11.8
#
# #### Based on:
# - https://github.com/agisoft-llc/metashape-scripts/blob/master/src/detect_objects.py
# - https://docs.ultralytics.com/
#
# ## How to install (Windows):
# How to install external Python module to Metashape Professional package https://agisoft.freshdesk.com/support/solutions/articles/31000136860-how-to-install-external-python-module-to-metashape-professional-package


# ## Install packages
# 1. Update pip in the command line run
# ```
#  - cd \d %programfiles%\Agisoft\python
#  - python.exe -m pip install --upgrade pip
# ```
# 2. Add this script to auto-launch, copy script to folder yolo11_detected to %programfiles%\Agisoft\modules and copy script run_scripts.py to C:/Users/<username>/AppData/Local/Agisoft/Metashape Pro/scripts/
# 3. Restart Metashape.
# 4. Wait for the end of the installation packages.
# 5. Uninstall torch, torchvision ( for cpu) and install torch+cuda torchvision+cuda (for gpu) from https://download.pytorch.org/whl/cu118
#
# ```
#  - cd \d %programfiles%\Agisoft\python
#  - python.exe -m pip uninstall -y torch torchvision
#  - python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# ```
# 6. Restart Metashape.

# ## Script use:
# """
# numpy==2.0.2
# pandas==2.2.3
# opencv-python==4.11.0.86
# shapely==2.0.7
# pathlib==1.0.1
# Rtree==1.3.0
# tqdm==4.67.1
# ultralytics==8.3.84
# torch==2.6.0+cu118
# torchvision==0.21.0+cu118
# scikit-learn==1.6.1
# """
# The script has additional modules:
# - shape transformations - to transform rectangles into the center points of these rectangles, as well as to create boxes using the starting point as the center.
# - Creating a dataset in yolo format.

import Metashape

from PySide2 import QtCore, QtWidgets
import pathlib, os, time, sys, subprocess, pkg_resources


requirements = """
numpy==2.0.2
pandas==2.2.3
opencv-python==4.11.0.86
shapely==2.0.7
pathlib==1.0.1
Rtree==1.3.0
tqdm==4.67.1
ultralytics==8.3.84
torch==2.6.0
torchvision==0.21.0
scikit-learn==1.6.1
"""


def is_package_installed(package_name, version=None):
    """
    Checks whether the package is installed with the specified version.

    :param package_name: The name of the package to check.
    ::type package_name: string
    :param version: The version of the package to check (optional).
    :type version: str
    :return: Package version if the specified version is installed, or False in case of an error.
    ::type: str | bool
    """
    try:
        if version:
            package_str = f"{package_name}=={version}"
            pkg_resources.require(package_str)

        installed_version = pkg_resources.get_distribution(package_name).version
        return installed_version
    except pkg_resources.DistributionNotFound:
        return False
    except pkg_resources.VersionConflict:
        return False

def check_package_installed(txt_requirements):
    requirements_dict = {
        line.split("==")[0]: line.split("==")[1]
        for line in txt_requirements.strip().split("\n")
    }

    # Checking each package and displaying the results
    packages_to_install = []
    for package, version in requirements_dict.items():
        installed_version = is_package_installed(package, version)
        if installed_version:
            print(f"{package} {installed_version} installed")

        else:
            if version:
                packages_to_install.append("{}=={}".format(package, version))
            else:
                packages_to_install.append("{}".format(package))
    return packages_to_install

def install_packages():
    """
    Checks and installs a predefined list of packages with specific versions if they are not already installed.
    It ensures the required versions of the packages are installed.
    If the required versions of the packages are not present, it installs them using pip and provides a link to a page where the correct versions can be found.

    :return: None
    """
    upgrade_pip_result = subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
                                        capture_output=True, text=True)
    print("STDOUT:", upgrade_pip_result.stdout) if upgrade_pip_result.stdout else None
    print("STDERR:", upgrade_pip_result.stderr) if upgrade_pip_result.stderr else None

    packages_to_install = check_package_installed(requirements)
    if packages_to_install:

        # Forming a command for pip install
        command = [sys.executable, "-m", "pip", "install", *packages_to_install]

        # Run command in the terminal
        result = subprocess.run(command, capture_output=True, text=True)

        # Checking the result of the command execution
        if result.returncode != 0:
            print(f"Error installing the package: ")
            for pk in packages_to_install:
                print(pk)
            print(result.stderr)
        else:
            print(result.stdout)

install_packages()

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import json
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import shutil
import yaml
import math
import random
from collections import OrderedDict
from shapely.errors import GEOSException
from shapely.geometry import Polygon, MultiPolygon, box
from shapely.ops import unary_union
from rtree import index
import torch
import pandas as pd
from ultralytics import YOLO

def pandas_append(df, row, ignore_index=False):
    """
    Appends a row or multiple rows to an existing pandas DataFrame.

    This function provides a flexible utility to append data to a pandas
    DataFrame. The input row can be either another DataFrame, a pandas Series,
    or a dictionary. Depending on the type of the input, it concatenates
    or integrates the data appropriately to the DataFrame. If a DataFrame or
    Series with all NA (Not Available) values is passed and the existing
    DataFrame is not empty, the function ensures that such input rows are
    not appended.

    Parameters:
        df (pd.DataFrame): The DataFrame to which the row data will be appended.
        row (Union[pd.DataFrame, pd.Series, dict]): The row or rows to append,
            which can be a DataFrame, Series, or dictionary. Its content will determine
            how the data is appended.
        ignore_index (bool): Whether to ignore the index values in the resulting
            DataFrame during concatenation. Defaults to False.

    Returns:
        pd.DataFrame: The resulting DataFrame after appending the given row(s). It
            reflects the combined structure of the original DataFrame and the
            appended data.

    Raises:
        RuntimeError: If the input row type is unsupported or not one of
            DataFrame, Series, or dictionary.
    """

    if isinstance(row, pd.DataFrame):

        if not df.empty and not row.isna().all().all():  # Доп проверка на NA
            result = pd.concat([df, row], ignore_index=True)
        else:
            result = row if df.empty else df

        # result = pd.concat([df, row], ignore_index=ignore_index) # old
    elif isinstance(row, pd.core.series.Series):
        result = pd.concat([df, row.to_frame().T], ignore_index=ignore_index)
    elif isinstance(row, dict):
        result = pd.concat([df, pd.DataFrame(row, index=[0], columns=df.columns)])
    else:
        raise RuntimeError("pandas_append: unsupported row type - {}".format(type(row)))
    return result

def getShapeVertices(shape):
    """
    Gets the vertices of the given shape.

    This function computes and returns a list of vertices for the specified shape. It retrieves marker positions for attached
    shapes or directly uses coordinate values for detached shapes. Transformations are applied to convert marker positions
    to the desired coordinate system when working with attached shapes.

    Parameters:
    shape: Metashape.Shape
        The shape from which vertices are to be extracted. The shape can either be attached or detached.

    Returns:
    list
        A list of vertex points representing the shape's geometry. The points are either transformed marker positions
        (for attached shapes) or coordinate values directly extracted from the shape (for detached shapes).

    Raises:
    Exception
        If the active chunk is null.
    Exception
        If any marker position is invalid within the given shape.
    """
    chunk = Metashape.app.document.chunk
    if chunk == None:
        raise Exception("Null chunk")

    T = chunk.transform.matrix
    result = []

    if shape.is_attached:
        assert (len(shape.geometry.coordinates) == 1)
        for key in shape.geometry.coordinates[0]:
            for marker in chunk.markers:
                if marker.key == key:
                    if (not marker.position):
                        raise Exception("Invalid shape vertex")

                    point = T.mulp(marker.position)
                    point = Metashape.CoordinateSystem.transform(point, chunk.world_crs, chunk.shapes.crs)
                    result.append(point)
    else:
        assert (len(shape.geometry.coordinates) == 1)
        for coord in shape.geometry.coordinates[0]:
            result.append(coord)

    return result

def get_device():
    """
    Determines the appropriate device for computation based on the availability of a
    CUDA-compatible GPU.

    Checks if a CUDA-compatible GPU is available and sets the computation device
    to GPU if available, otherwise defaults to the CPU.

    Returns:
        int or str: Returns 0 if a CUDA-compatible GPU is available, indicating
        the use of the first GPU. Otherwise, returns the string 'cpu' indicating
        that the computations will run on the CPU.
    """

    device = 0 if torch.cuda.is_available() else 'cpu'
    return device

def ensure_unique_directory(base_dir):
    """
    Generates a unique directory name by appending a numeric suffix to the provided base directory
    name if a directory with the same base name already exists. If the base directory does not exist,
    it is returned unchanged.

    Args:
        base_dir (str): The base directory name to check and ensure uniqueness for.

    Returns:
        str: A unique directory name. If the base directory does not exist, the same directory name is
        returned. If it does exist, a unique name with an appended numeric suffix is returned.
    """
    if not os.path.exists(base_dir):
        return base_dir

    counter = 1
    new_dir = f"{base_dir}_{counter}"  # Формируем начальное имя (с постфиксом `1`)
    while os.path.exists(new_dir):  # Пока директория существует, увеличиваем счетчик
        counter += 1
        new_dir = f"{base_dir}_{counter}"

    return new_dir

def remove_directory(directory_path):
    """
    Removes the specified directory and its contents if it exists. If the directory
    doesn't exist, the function does nothing.

    Args:
        directory_path (str): The path to the directory to remove.

    Raises:
        Exception: If an unexpected error occurs while attempting to remove the
        directory.
    """
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
    else:
        pass


def merge_contours_shapely(contours, iou_threshold=0.7):
    # Преобразуем входные контуры в полигоны
    polygons = []
    for contour in contours:
        try:
            poly = Polygon(contour).buffer(0)  # Исправляем геометрию через buffer(0)
            if poly.is_valid and not poly.is_empty:
                polygons.append(poly)
            else:
                print(f"Skipping invalid geometry: {contour}")
        except GEOSException as e:
            print(f"Error creating polygon: {e}, skipping this geometry.")

    # Итеративное объединение
    while len(polygons) > 1:  # Пока есть больше одного полигона, можно пытаться объединить
        has_changes = False
        merged_polygons = []
        visited = [False] * len(polygons)  # Массив для отметки объединённых полигонов

        # Индексируем полигоны для оптимального поиска
        idx = index.Index()
        for pos, poly in enumerate(polygons):
            if not poly.is_empty and poly.is_valid:  # Пропускаем пустые/невалидные полигоны
                idx.insert(pos, poly.bounds)

        for pos, poly in enumerate(polygons):
            if visited[pos] or poly.is_empty or not poly.is_valid:  # Если уже обработали или полигон некорректен
                continue

            merge_queue = [pos]
            candidate_merged_poly = poly
            visited[pos] = True

            # Перебор всех соседних полигонов
            while merge_queue:
                current_idx = merge_queue.pop()
                current_poly = polygons[current_idx]

                for merge_index in idx.intersection(current_poly.bounds):  # Проверяем пересекающиеся полигоны
                    if visited[merge_index] or merge_index == current_idx:
                        continue

                    neighbor_poly = polygons[merge_index]
                    try:
                        # Проверка пересечения и вычисление IoU (по отношению к минимальной площади)
                        if current_poly.intersects(neighbor_poly):
                            intersection_area = current_poly.intersection(neighbor_poly).area
                            min_area = min(current_poly.area, neighbor_poly.area)

                            if intersection_area / min_area >= iou_threshold:
                                # Объединяем текущий полигон с соседним
                                candidate_merged_poly = unary_union([candidate_merged_poly, neighbor_poly]).buffer(0)

                                if candidate_merged_poly.is_valid and not candidate_merged_poly.is_empty:
                                    merge_queue.append(merge_index)
                                    visited[merge_index] = True
                                    has_changes = True
                    except GEOSException as e:
                        print(f"Intersection check failed: {e}")
                        continue

            # Добавляем объединённый полигон в список
            if candidate_merged_poly.is_valid and not candidate_merged_poly.is_empty:
                merged_polygons.append(candidate_merged_poly)

        # Если за итерацию ничего не изменилось, объединение завершено
        if not has_changes:
            break

        # Обновляем список полигонов после объединения
        polygons = merged_polygons

    # Переводим полигоны обратно в контуры
    result_contours = []
    for poly in polygons:
        if isinstance(poly, Polygon):
            result_contours.append(np.array(poly.exterior.coords))
        elif isinstance(poly, MultiPolygon):
            multi_poly_contours = []
            for sub_poly in poly.geoms:
                if sub_poly.is_valid and not sub_poly.is_empty:  # Пропускаем невалидные или пустые
                    multi_poly_contours.append(np.array(sub_poly.exterior.coords))

            # Вычисление площади каждого контура
            areas = []
            for contour in multi_poly_contours:
                if len(contour) >= 4:  # Убеждаемся, что контур состоит как минимум из 4 точек
                    areas.append(cv2.contourArea(np.array(contour, dtype=np.float32)))

            # Нахождение индекса контура с наибольшей площадью
            if areas:
                largest_contour_index = np.argmax(areas)
                # Выбор контура с наибольшей площадью
                largest_contour = multi_poly_contours[largest_contour_index]
                result_contours.append(largest_contour)

    return result_contours

def calculate_iou(box1, box2):
    """
    Вычислить IoU (Intersection over Union) для двух боксов: box1 и box2.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Площадь пересечения
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    # Площади каждого бокса
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Площадь объединения
    union_area = box1_area + box2_area - intersection_area

    # IoU
    iou = intersection_area / union_area if union_area != 0 else 0
    return iou

def is_inside(box1, box2, threshold=0.9):
    """
    Проверяет, находится ли box1 внутри box2 с заданным процентом перекрытия.
    :param box1: Координаты первого бокса [x1, y1, x2, y2].
    :param box2: Координаты второго бокса [x1, y1, x2, y2].
    :param threshold: Порог перекрытия (от 0 до 1), 1.0 для полного пересечения, 0.9 для 90%.
    :return: True, если box1 входит в box2 на 'threshold', иначе False.
    """
    # Вычисляем координаты пересечения
    intersect_x1 = max(box1[0], box2[0])
    intersect_y1 = max(box1[1], box2[1])
    intersect_x2 = min(box1[2], box2[2])
    intersect_y2 = min(box1[3], box2[3])

    # Вычисляем площадь пересечения
    intersect_area = max(0, intersect_x2 - intersect_x1) * max(0, intersect_y2 - intersect_y1)

    # Вычисляем площадь первого бокса (box1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])

    # Проверяем условие перекрытия
    return (intersect_area / box1_area) >= threshold

def merge_boxes(box1, box2):
    """
    Объединяет два бокса в один, охватывающий оба.
    """
    x1 = min(box1[0], box2[0])
    y1 = min(box1[1], box2[1])
    x2 = max(box1[2], box2[2])
    y2 = max(box1[3], box2[3])
    return [x1, y1, x2, y2]

def merge_all_boxes(boxes, iou_threshold=0.9):
    """
    Объединяет все пересекающиеся боксы или полностью вложенные, пока они существуют.
    """
    merged_boxes = boxes.copy()  # Создаем копию оригинального списка

    while True:
        new_boxes = []
        skip_indices = set()  # Индексы, которые были добавлены/объединены в этой итерации
        merged = False  # Флаг для контроля, были ли объединения

        for i in range(len(merged_boxes)):
            if i in skip_indices:
                continue  # Пропускаем уже обработанные боксы

            current_box = merged_boxes[i]  # Текущий бокс, который мы пытаемся объединить

            for j in range(len(merged_boxes)):
                if i == j or j in skip_indices:
                    continue  # Пропускаем самого себя или уже обработанный бокс

                # Проверяем условия объединения:
                iou = calculate_iou(current_box, merged_boxes[j])
                if (
                        iou >= iou_threshold or
                        is_inside(current_box, merged_boxes[j], iou_threshold) or
                        is_inside(merged_boxes[j], current_box, iou_threshold)
                ):
                    # Объединяем два бокса
                    current_box = merge_boxes(current_box, merged_boxes[j])
                    skip_indices.add(j)  # Помечаем, что этот бокс был обработан
                    merged = True  # Указываем, что произошло объединение

            # Сохраняем объединённый бокс
            new_boxes.append(current_box)
            skip_indices.add(i)

        # Если ни одно объединение не произошло, завершить
        if not merged:
            break

        # Обновляем список боксов для следующей итерации
        merged_boxes = new_boxes

    return merged_boxes


class MainWindowDetect(QtWidgets.QDialog):

    def __init__(self, parent):
        QtWidgets.QDialog.__init__(self, parent)

        self.results_time_total = None
        self.stopped = False
        self.model = None
        self.classes = None
        self.force_small_patch_size = True
        self.isLoadTiles = False
        self.expected_layer_name_train_zones = "Zone"
        self.expected_layer_name_train_data = "Train data"
        self.layer_name_detection_data = ""

        if len(Metashape.app.document.path) > 0:
            self.working_dir = str(pathlib.Path(Metashape.app.document.path).parent / "objects_detection")
        else:
            self.working_dir = ""

        self.load_model_path = self.read_model_path_from_settings()
        max_image_size = Metashape.app.settings.value("scripts/yolo/max_image_size")
        self.max_image_size = int(max_image_size) if max_image_size else 640
        self.cleanup_working_dir = False
        self.isDebugMode = False  #
        self.train_on_user_data_enabled = False
        self.preferred_patch_size = 640  # 640 pixels
        self.preferred_resolution = 0.005  # 0,5 cm/pix
        self.detection_score_threshold = 0.90
        self.prefer_original_resolution = False

        self.setWindowTitle("YOLO objects detection on orthomosaic")
        self.chunk = Metashape.app.document.chunk
        self.create_gui()
        self.exec()

    def stop(self):
        """
        Stops the running process by setting the stopped state to True.

        This method updates the state of the `stopped` attribute, effectively causing any
        process depending on this attribute to halt its execution. It provides a mechanism
        to control and terminate activities gracefully.

        Attributes:
            stopped (bool): Represents whether the process is stopped. Initially set to
                            False and updated to True when this method is called.
        """
        self.stopped = True

    def check_stopped(self):
        """
        Checks whether the 'stopped' attribute is True and raises an exception if it is.

        This method is used to determine if a stop signal has been triggered. If the
        'stopped' attribute evaluates to True, it raises an InterruptedError, which
        signals that the operation should be halted.

        Raises:
            InterruptedError: If the 'stopped' attribute is True, indicating a
            stop request.
        """
        if self.stopped:
            raise InterruptedError("Stop was pressed")

    def run_detect(self):
        """
        Runs the detection process and controls the workflow of specific tasks such as loading
        parameters, preparing the environment, creating a neural network, and exporting results.

        This method also tracks processing time, manages user interactions via enabling/disabling
        appropriate UI buttons, and handles post-processing cleanup. The method detects zones
        using specified layers if available, or generic detection if none is provided.

        """
        try:
            self.stopped = False
            self.btnDetect.setEnabled(False)
            self.btnStop.setEnabled(True)

            time_start = time.time()

            self.load_params()
            self.prepare()

            print("Script started...")

            self.create_neural_network()
            self.export_orthomosaic()

            if self.chunk.shapes is None:
                self.chunk.shapes = Metashape.Shapes()
                self.chunk.shapes.crs = self.chunk.crs

            detectZonesLayer = self.layers[self.detectZonesLayer.currentIndex()]
            if detectZonesLayer == self.noDataChoice:
                self.detect()
            else:
                self.detect_tiles_zones()

            self.results_time_total = time.time() - time_start
            self.show_results_dialog()
        except Exception as e:
            if self.stopped:
                Metashape.app.messageBox("Processing was stopped.")
            else:
                Metashape.app.messageBox("Something gone wrong.\n"
                                         "Please check the console.")

                raise
        finally:
            if self.cleanup_working_dir:
                # shutil.rmtree(self.working_dir, ignore_errors=True)
                pass
            self.reject()

        print("Script finished.")
        return True

    def prepare(self):
        """
        Prepares the working directory and its subdirectories for the project. Ensures the necessary
        directory structure is created and sets configurations for multiprocessing if the operating system
        is Windows.

        Raises
        ------
        Exception
            If the working directory is not specified.
        """

        import multiprocessing as mp

        if self.working_dir == "":
            raise Exception("You should specify working directory (or save .psx project)")

        os.makedirs(self.working_dir, exist_ok=True)
        print("Working dir: {}".format(self.working_dir))

        self.dir_tiles = self.working_dir + "/tiles/"
        self.dir_detection_results = self.working_dir + "/detection/"
        self.dir_subtiles_results = self.dir_detection_results + "inner/"

        for subdir in [self.dir_tiles,
                       self.dir_detection_results, self.dir_subtiles_results,
                       ]:
            os.makedirs(subdir, exist_ok=True)

        if os.name == 'nt':  # if Windows
            mp.set_executable(os.path.join(sys.exec_prefix, 'python.exe'))
            print(f"multiprocessing set_executable: {os.path.join(sys.exec_prefix, 'python.exe')}")

    def create_neural_network(self):
        """
        Loads and initializes a neural network model using the Ultralytics YOLO framework. This function
        either loads a neural network from a pre-specified path or raises an error in the
        absence of a specified model.

        Parameters: None

        Raises:
            FileExistsError: If no model path is specified upon invocation.
        """
        print("Neural network loading...")


        if self.load_model_path:
            print("Using the neural network loaded from '{}'...".format(self.load_model_path))
            self.save_model_path_to_settings(self.load_model_path)

            self.model = YOLO(self.load_model_path)
            self.classes = self.model.names

        else:
            raise FileExistsError("No neural network was specified")

    def export_orthomosaic(self):
        """
        Prepares and exports an orthomosaic image to a set of tiles, then processes
        the tiles to map their paths and spatial transformation data. It ensures
        the tiles are stored with proper metadata required for further processing.
        Handles resolution preferences and manages data accordingly.

        Raises:
            Exception: If no tiles are found in the specified directory.

        """

        print("Preparing orthomosaic...")

        kwargs = {}
        if not self.prefer_original_resolution:  # and (self.chunk.orthomosaic.resolution < self.preferred_resolution * 0.90):
            kwargs["resolution"] = self.preferred_resolution
        else:
            print("no resolution downscaling required")

        if not self.isLoadTiles:
            self.chunk.exportRaster(path=self.dir_tiles + "tile.jpg",
                                    source_data=Metashape.OrthomosaicData,
                                    image_format=Metashape.ImageFormat.ImageFormatJPEG,
                                    save_alpha=False,
                                    white_background=True,
                                    save_world=True,
                                    split_in_blocks=True,
                                    block_width=self.patch_size,
                                    block_height=self.patch_size,
                                    **kwargs)

        tiles = os.listdir(self.dir_tiles)
        if not tiles:
            raise Exception("No tiles found in the directory.")

        self.tiles_paths = {}
        self.tiles_to_world = {}

        for tile in sorted(tiles):

            if not tile.startswith("tile-"):
                continue

            _, tile_x, tile_y = tile.split(".")[0].split("-")
            tile_x, tile_y = map(int, [tile_x, tile_y])
            if tile.endswith(".jgw") or tile.endswith(".pgw"):  # https://en.wikipedia.org/wiki/World_file
                with open(self.dir_tiles + tile, "r") as file:
                    matrix2x3 = list(map(float, file.readlines()))
                matrix2x3 = np.array(matrix2x3).reshape(3, 2).T
                self.tiles_to_world[tile_x, tile_y] = matrix2x3
            elif tile.endswith(".jpg"):
                self.tiles_paths[tile_x, tile_y] = self.dir_tiles + tile

        assert (len(self.tiles_paths) == len(self.tiles_to_world))
        assert (self.tiles_paths.keys() == self.tiles_to_world.keys())

        self.tile_min_x = min([key[0] for key in self.tiles_paths.keys()])
        self.tile_max_x = max([key[0] for key in self.tiles_paths.keys()])
        self.tile_min_y = min([key[1] for key in self.tiles_paths.keys()])
        self.tile_max_y = max([key[1] for key in self.tiles_paths.keys()])
        print("{} tiles, tile_x in [{}; {}], tile_y in [{}; {}]".format(len(self.tiles_paths), self.tile_min_x,
                                                                        self.tile_max_x, self.tile_min_y,
                                                                        self.tile_max_y))

    def read_part(self, res_from, res_to):
        """
        Reads a specified region of an image by combining multiple tiles into a single output.

        This method processes a specified rectangular region of an image, defined by the
        coordinates `res_from` and `res_to`, by stitching together multiple smaller tiles.
        The output is a complete section of the image with padding applied where necessary
        to fill missing parts.

        Parameters:
            res_from (numpy.ndarray): A 2-element array specifying the top-left corner
                (x, y) of the region to be read.
            res_to (numpy.ndarray): A 2-element array specifying the bottom-right corner
                (x, y) of the region to be read.

        Returns:
            numpy.ndarray: A 3D array representing the specified region of the image in RGB
                format, with missing areas filled with white.

        Raises:
            AssertionError: If the size of the requested region is smaller than the patch
                size or if an invalid region range is specified.
        """

        res_size = res_to - res_from
        assert np.all(res_size >= [self.patch_size, self.patch_size])
        res = np.zeros((res_size[1], res_size[0], 3), np.uint8)
        res[:, :, :] = 255

        tile_xy_from = np.int32(res_from // self.patch_size)
        tile_xy_upto = np.int32((res_to - 1) // self.patch_size)
        assert np.all(tile_xy_from <= tile_xy_upto)
        for tile_x in range(tile_xy_from[0], tile_xy_upto[0] + 1):
            for tile_y in range(tile_xy_from[1], tile_xy_upto[1] + 1):
                if (tile_x, tile_y) not in self.tiles_paths:
                    continue
                part = cv2.imread(self.tiles_paths[tile_x, tile_y])
                part = cv2.copyMakeBorder(part, 0, self.patch_size - part.shape[0], 0, self.patch_size - part.shape[1],
                                          cv2.BORDER_CONSTANT, value=[255, 255, 255])
                part_from = np.int32([tile_x, tile_y]) * self.patch_size - res_from
                part_to = part_from + self.patch_size

                res_inner_from = np.int32([max(0, part_from[0]), max(0, part_from[1])])
                res_inner_to = np.int32([min(part_to[0], res_size[0]), min(part_to[1], res_size[1])])

                part_inner_from = res_inner_from - part_from
                part_inner_to = part_inner_from + res_inner_to - res_inner_from

                res[res_inner_from[1]:res_inner_to[1], res_inner_from[0]:res_inner_to[0], :] = part[part_inner_from[1]:
                                                                                                    part_inner_to[1],
                                                                                               part_inner_from[0]:
                                                                                               part_inner_to[0], :]

        return res

    @staticmethod
    def add_pixel_shift(to_world, dx, dy):
        """
        Adds a pixel shift to a given transformation matrix by updating its
        translation components based on the specified shifts in the x and y
        directions. This operation modifies only the translation components
        of the matrix while keeping other transformations intact.

        Args:
            to_world (numpy.ndarray): The transformation matrix to modify.
                Expected to be a 3x3 matrix representing an affine transformation
                in homogeneous coordinates.
            dx (float): The horizontal pixel shift to apply to the transformation.
            dy (float): The vertical pixel shift to apply to the transformation.

        Returns:
            numpy.ndarray: A new transformation matrix with the updated translation
            components that include the specified pixel shifts.
        """
        to_world = to_world.copy()
        to_world[0, 2] = to_world[0, :] @ [dx, dy, 1]
        to_world[1, 2] = to_world[1, :] @ [dx, dy, 1]
        return to_world

    @staticmethod
    def invert_matrix_2x3(to_world):
        """
        Invert a 2x3 transformation matrix.

        This static method takes a 2x3 transformation matrix and calculates its inverse.
        The input matrix is extended into a 3x3 matrix by appending a row of [0, 0, 1]
        to make it suitable for inversion. The result is a 2x3 matrix obtained by
        removing the last row after inversion.

        Parameters:
            to_world: numpy.ndarray
                A 2x3 transformation matrix to be inverted.

        Returns:
            numpy.ndarray
                The inverted 2x3 transformation matrix.

        Raises:
            AssertionError
                If the extended 3x3 matrix does not fulfill the expected
                constraints after inversion.
        """


        to_world33 = np.vstack([to_world, [0, 0, 1]])
        from_world = np.linalg.inv(to_world33)

        assert (from_world[2, 0] == from_world[2, 1] == 0)
        assert (from_world[2, 2] == 1)
        from_world = from_world[:2, :]

        return from_world

    @staticmethod
    def read_model_path_from_settings():
        """
        Reads the model path from the application settings.

        This static method retrieves the model load path specified in the application's settings.
        If no path is found, it defaults to an empty string.

        Returns:
            str: The model load path from the application's settings or an empty string if no path is set.
        """
        load_path = Metashape.app.settings.value("scripts/yolo/model_load_path")
        if load_path is None:
            load_path = ""
        return load_path

    @staticmethod
    def save_model_path_to_settings(load_path):
        """
        Sets and saves the model load path to Metashape application settings.

        This static method is used to save the specified model load path into the
        Metashape application settings under the specified key. It modifies
        persistent application settings to store the given path for future use.

        Args:
            load_path (str): The file system path to the model that needs to be saved
            into the Metashape application settings.
        """
        Metashape.app.settings.setValue("scripts/yolo/model_load_path", load_path)


    def draw_boxes_zone_tiles(self, tiles_data):
        """
        Draws boxes on zone tiles based on provided tile data and adds them as polygon shapes to the
        project's chunk.

        This function processes tile data containing bounding box coordinates and transforms them from
        zone-based coordinates to the appropriate coordinate system used within the Metashape project.
        The resultant polygons are labeled and grouped under a new shapes group.

        Args:
            tiles_data (list[dict]): A list of dictionaries where each dictionary represents a tile's
                                      data. Each dictionary must contain:
                                      - 'x_tile': int, X coordinate of the tile's bottom-left corner.
                                      - 'y_tile': int, Y coordinate of the tile's bottom-left corner.
                                      - 'x_max': int, X coordinate of the tile's top-right corner.
                                      - 'y_max': int, Y coordinate of the tile's top-right corner.
                                      - 'label': str, Label for the polygon shape.
                                      - 'zone_to_world': numpy.ndarray, Transformation matrix from zone
                                        to world coordinates.
        """

        shapes_group = self.chunk.shapes.addGroup()
        shapes_group.label = "Tiles Boxes"
        shapes_group.show_labels = False

        for row in tiles_data:
            xmin = int(row["x_tile"])
            ymin = int(row["y_tile"])
            xmax = int(row["x_max"])
            ymax = int(row["y_max"])
            label = row["label"]
            zone_to_world = row["zone_to_world"]

            corners = []
            for x, y in [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]:
                x, y = zone_to_world @ np.array([x + 0.5, y + 0.5, 1]).reshape(3, 1)
                p = Metashape.Vector([x, y])
                p = Metashape.CoordinateSystem.transform(p, self.chunk.orthomosaic.crs, self.chunk.shapes.crs)
                corners.append([p.x, p.y])

            shape = self.chunk.shapes.addShape()
            shape.group = shapes_group
            shape.geometry = Metashape.Geometry.Polygon(corners)
            shape.label = label

    def get_tiles_from_zones(self):
        """
        Extracts and processes tiles from zones based on detected shapes and spatial relationships.

        This method identifies tiles from zones defined by shapes, applying coordinate transformations
        to determine the zones' presence and placement within the orthomosaic project. It further calculates
        dimensions, verifies permissible sizes, and processes tiles accordingly while noting tiles outside
        the orthomosaic or with excessive white pixels.

        Parameters
        ----------
        self : class
            An instance of the class containing detected zones, tiles' spatial metadata, and necessary
            attributes for tile extraction and processing.

        Returns
        -------
        List[Dict[str, Any]]
            A list of dictionaries where each dictionary represents a processed tile. Each dictionary
            contains the tile's data, position information, its label, and a transformation matrix.
        """

        all_tiles_zones = []
        zones_on_ortho = []

        for zone_i, shape in enumerate(self.detected_zones):
            shape_vertices = getShapeVertices(shape)
            zone_from_world = None
            zone_from_world_best = None
            zone_to_world = None

            for tile_x in range(self.tile_min_x, self.tile_max_x + 1):
                for tile_y in range(self.tile_min_y, self.tile_max_y + 1):
                    if (tile_x, tile_y) not in self.tiles_paths:
                        continue
                    to_world = self.tiles_to_world[tile_x, tile_y]
                    from_world = self.invert_matrix_2x3(to_world)
                    for v_p in shape_vertices:
                        p = Metashape.CoordinateSystem.transform(v_p, self.chunk.shapes.crs,
                                                                 self.chunk.orthomosaic.crs)
                        p_in_tile = from_world @ [p.x, p.y, 1]
                        distance2_to_tile_center = np.linalg.norm(
                            p_in_tile - [self.patch_size / 2, self.patch_size / 2])
                        if zone_from_world_best is None or distance2_to_tile_center < zone_from_world_best:
                            zone_from_world_best = distance2_to_tile_center
                            zone_from_world = self.invert_matrix_2x3(
                                self.add_pixel_shift(to_world, -tile_x * self.patch_size, -tile_y * self.patch_size))
                            zone_to_world = self.add_pixel_shift(to_world, -tile_x * self.patch_size,
                                                                 -tile_y * self.patch_size)

            zone_from = None
            zone_to = None
            for v_p in shape_vertices:
                p = Metashape.CoordinateSystem.transform(v_p, self.chunk.shapes.crs, self.chunk.orthomosaic.crs)
                p_in_ortho = np.int32(np.round(zone_from_world @ [p.x, p.y, 1]))
                if zone_from is None:
                    zone_from = p_in_ortho
                if zone_to is None:
                    zone_to = p_in_ortho
                zone_from = np.minimum(zone_from, p_in_ortho)
                zone_to = np.maximum(zone_to, p_in_ortho)
            train_size = zone_to - zone_from
            train_size_m = np.int32(np.round(train_size * self.orthomosaic_resolution))
            if np.any(train_size < self.patch_size):
                print("Zone #{} {}x{} pixels ({}x{} meters) is too small - each side should be at least {} meters"
                      .format(zone_i + 1, train_size[0], train_size[1], train_size_m[0], train_size_m[1],
                              self.patch_size * self.orthomosaic_resolution), file=sys.stderr)
                zones_on_ortho.append(None)
            else:
                print("Zone #{}: {}x{} orthomosaic pixels, {}x{} meters".format(zone_i + 1, train_size[0],
                                                                                train_size[1], train_size_m[0],
                                                                                train_size_m[1]))

                self.check_stopped()

                border = self.patch_inner_border
                inner_path_size = self.patch_size - 2 * border

                zone_size = zone_to - zone_from
                assert np.all(zone_size >= self.patch_size)
                nx_tiles, ny_tiles = np.int32((zone_size - 2 * border + inner_path_size - 1) // inner_path_size)
                assert nx_tiles >= 1 and ny_tiles >= 1
                xy_step = np.int32(np.round((zone_size + [nx_tiles, ny_tiles] - 1) // [nx_tiles, ny_tiles]))

                out_of_orthomosaic_train_tile = 0

                for x_tile in range(0, nx_tiles):
                    for y_tile in range(0, ny_tiles):
                        tile_to = zone_from + self.patch_size + xy_step * [x_tile, y_tile]
                        if x_tile == nx_tiles - 1 and y_tile == ny_tiles - 1:
                            assert np.all(tile_to >= zone_to)
                        tile_to = np.minimum(tile_to, zone_to)
                        tile_from = tile_to - self.patch_size
                        if x_tile == 0 and y_tile == 0:
                            assert np.all(tile_from == zone_from)
                        assert np.all(tile_from >= zone_from)

                        tile = self.read_part(tile_from, tile_to)
                        assert tile.shape == (self.patch_size, self.patch_size, 3)

                        white_pixels_fraction = np.sum(np.all(tile == 255, axis=-1)) / (
                                tile.shape[0] * tile.shape[1])
                        if np.all(tile == 255) or white_pixels_fraction >= 0.90:
                            out_of_orthomosaic_train_tile += 1
                            continue

                        label_tile = f"{(zone_i + 1)}-{x_tile}-{y_tile}"

                        all_tiles_zones.append({"tile": tile, "x_tile": tile_from[0], "y_tile": tile_from[1],
                                                "x_max": tile_to[0], "y_max": tile_to[1], "label": label_tile,
                                                "zone_to_world": zone_to_world})


        print(f"Tiles: {len(all_tiles_zones)}")
        return all_tiles_zones

    def detect_tiles_zones(self):

        app = QtWidgets.QApplication.instance()

        print("Detection selected zones...")

        if not self.model:
            print("No init model!")
            return

        print(f"Zones: {len(self.detected_zones)}")
        print(f"Classes: {self.classes}")
        print(f"Size tile: {self.max_image_size}")
        # imgsz = self.model.ckpt['train_args']['imgsz']
        # print(f"Size img model: {imgsz}")

        # Напечатать устройство для проверки
        device = get_device()
        print(f"Using device: {device}")

        tiles_from_zones = self.get_tiles_from_zones()

        self.draw_boxes_zone_tiles(tiles_from_zones)

        subtile_inner_preds = pd.DataFrame(
            columns=['zone_to_world', 'xmin', 'ymin', 'xmax', 'ymax', 'label', 'score', 'mask'])

        for tile_index, tile_data in enumerate(tiles_from_zones):
            self.txtDetectionPBar.setText(f"Detection progress: ({tile_index + 1} of {len(tiles_from_zones)})")
            self.detectionPBar.setValue(int((tile_index + 1) * 100 / len(tiles_from_zones)))

            subtile = tile_data["tile"]
            x_tile = tile_data["x_tile"]
            y_tile = tile_data["y_tile"]
            zone_to_world = tile_data["zone_to_world"]

            save = False
            save_conf = False
            save_txt = False
            save_crop = False

            if self.isDebugMode:
                save = True
                save_conf = True
                save_txt = True
                save_crop = True

            # Выполнение предсказания
            with torch.no_grad():
                subtile_prediction = self.model.predict(subtile, imgsz=self.max_image_size, device=device,
                                                   conf=self.detection_score_threshold, iou=0.45,
                                                   project=self.dir_subtiles_results, save=save, save_conf=save_conf,
                                                   save_txt=save_txt, save_crop=save_crop, half=True, )

            Metashape.app.update()
            app.processEvents()
            self.check_stopped()

            subtile_prediction = subtile_prediction[0].cpu()

            if subtile_prediction.boxes is not None:
                # Получение размеров изображения (из results, оригинальные размеры фото)
                original_shape = subtile_prediction.orig_shape  # Исходные размеры фото (высота, ширина)
                orig_h, orig_w = original_shape[:2]

                masks = None
                # Получаем данные из предсказания
                if subtile_prediction.masks is not None:
                    masks = subtile_prediction.masks.xyn

                boxes = subtile_prediction.boxes.xyxyn
                # labels = subtile_prediction.boxes.cls.data.numpy()

                for idx, bbox in enumerate(boxes):
                    box = subtile_prediction.boxes[idx]
                    score = box.conf.numpy()[0]
                    b_class = box.cls.numpy()
                    label = b_class[0]

                    xmin, ymin, xmax, ymax = bbox
                    xmin, ymin, xmax, ymax = int(xmin * orig_w), int(ymin * orig_h), int(xmax * orig_w), int(
                        ymax * orig_h)

                    xmin, xmax = map(lambda x: x_tile + x, [xmin, xmax])
                    ymin, ymax = map(lambda y: y_tile + y, [ymin, ymax])

                    row = {"zone_to_world": zone_to_world, "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
                           "label": label, "score": score, 'mask': []}

                    if masks is not None and len(masks) > 0:
                        # Добавление маски
                        mask = masks[idx]

                        # Преобразуем нормализованные координаты в пиксели
                        pixel_coords = np.array([[int(y * orig_h), int(x * orig_w)] for y, x in mask],
                                                dtype=np.int32)

                        # Создаем цветную маску
                        color_mask = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)

                        # Заполняем область красным цветом (BGR)
                        cv2.fillPoly(color_mask, [pixel_coords], color=(0, 0, 255))

                        # Ищем контуры на основе красного канала
                        contours, _ = cv2.findContours(color_mask[:, :, 2], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                        mask_contour = []
                        if contours:
                            # Вычисление площади каждого контура
                            areas = [cv2.contourArea(contour) for contour in contours]

                            # Нахождение индекса контура с наибольшей площадью
                            largest_contour_index = np.argmax(areas)

                            # Выбор контура с наибольшей площадью
                            largest_contour = contours[largest_contour_index]

                            # Сдвигаем координаты точек контура на значения fromx и fromy
                            shifted_contour = [[x + x_tile, y + y_tile] for [[x, y]] in largest_contour]

                            # Добавляем смещенные точки в mask_contour
                            mask_contour.extend(shifted_contour)
                            row['mask'] = mask_contour

                    subtile_inner_preds = pandas_append(subtile_inner_preds, pd.DataFrame([row]),
                                                        ignore_index=True)

        # Создаем новые слои для результатов

        box_detected_label = "box_detected ({:.2f} cm/pix, ".format(
            100.0 * self.orthomosaic_resolution)
        box_detected_label += f"size img: {self.max_image_size}, threshold: {self.detection_score_threshold})"

        detected_shapes_layer = self.chunk.shapes.addGroup()
        detected_shapes_layer.label = box_detected_label
        detected_shapes_layer.show_labels = False

        boxes_shapes = self.add_boxes_from_pred_zones(subtile_inner_preds, detected_shapes_layer)

        outline_shapes = []
        if subtile_inner_preds['mask'].notna().any():
            detected_mask_label = self.layer_name_detection_data + "outline_detected ({:.2f} cm/pix, ".format(
                100.0 * self.orthomosaic_resolution)
            detected_mask_label += f"size img: {self.max_image_size}, threshold: {self.detection_score_threshold})"
            detected_mask_shapes_layer = self.chunk.shapes.addGroup()
            detected_mask_shapes_layer.label = detected_mask_label
            detected_mask_shapes_layer.show_labels = False
            outline_shapes = self.add_masks_from_pred_zones(subtile_inner_preds, detected_mask_shapes_layer)

        print(f"Обработка пересекающихся контуров...")

        coords_outline_shapes = []

        for shape in outline_shapes:
            coords_outline_shapes.append(np.array(getShapeVertices(shape)))

        coords_boxes_shapes = []

        for shape in boxes_shapes:
            coords_boxes_shapes.append(np.array(getShapeVertices(shape)).tolist())

        coords_boxes_xyxy = [
            [
                min(point[0] for point in box),  # x_min
                min(point[1] for point in box),  # y_min
                max(point[0] for point in box),  # x_max
                max(point[1] for point in box)  # y_max
            ]
            for box in coords_boxes_shapes
        ]

        # Объединяем пересекающиеся фигуры
        start_time = time.time()

        merged_shapes = merge_contours_shapely(coords_outline_shapes, iou_threshold=0.7)
        merged_boxes = merge_all_boxes(coords_boxes_xyxy, iou_threshold=0.7)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(
            f"Обработка {len(coords_outline_shapes)} контуров, результат {len(merged_shapes)} контуров, продолжительность {elapsed_time / 60:.2f} минут")

        # Применяем новые объединенные фигуры обратно в проект
        union_outline_detected_label = "union_outline_detected ({:.2f} cm/pix, ".format(
            100.0 * self.orthomosaic_resolution)
        union_outline_detected_label += f"size img: {self.max_image_size}, threshold: {self.detection_score_threshold})"

        if outline_shapes:
            self.apply_union_mask_shapes(self.chunk, merged_shapes, union_outline_detected_label)

        # Применяем новые объединенные фигуры обратно в проект
        union_outline_detected_label = "union_boxes_detected ({:.2f} cm/pix, ".format(
            100.0 * self.orthomosaic_resolution)
        union_outline_detected_label += f"size img: {self.max_image_size}, threshold: {self.detection_score_threshold})"
        if merged_boxes:
            self.apply_union_boxes_shapes(self.chunk, merged_boxes, union_outline_detected_label)

        Metashape.app.update()

    def add_boxes_from_pred_zones(self, data, shapes_group):
        """
        Adds bounding boxes as shapes to the Metashape model from predictions within specified zones.

        This function processes a set of bounding box predictions, transforms their coordinates from a local
        coordinate system to the world coordinate system, and creates corresponding polygonal shapes in the
        Metashape chunk. Each shape is assigned a label, a score, and associated with a given shapes group.
        The function returns a list of created shapes for further usage.

        Parameters:
        data : DataFrame
            A pandas DataFrame containing the bounding box data. Each row must represent a single bounding
            box and include the following columns: xmin, ymin, xmax, ymax, label, score, and zone_to_world.
        shapes_group : object
            The group object to which the newly created shapes will belong.

        Returns:
        list
            A list of created shape objects corresponding to the bounding boxes.
        """

        boxes_shapes = []
        for row in data.itertuples():
            xmin, ymin, xmax, ymax, label, score, zone_to_world = int(row.xmin), int(row.ymin), int(row.xmax), int(
                row.ymax), row.label, row.score, row.zone_to_world

            corners = []
            for x, y in [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]:
                x, y = zone_to_world @ np.array([x, y, 1]).reshape(3, 1)
                p = Metashape.Vector([x, y])
                p = Metashape.CoordinateSystem.transform(p, self.chunk.orthomosaic.crs, self.chunk.shapes.crs)
                corners.append([p.x, p.y])

            shape = self.chunk.shapes.addShape()
            shape.group = shapes_group
            shape.geometry = Metashape.Geometry.Polygon(corners)
            shape.label = self.classes[int(label)]
            shape.attributes["Score"] = str(score)

            boxes_shapes.append(shape)
        return boxes_shapes

    def add_masks_from_pred_zones(self, data, shapes_group):
        """
        Transforms the coordinates of masks to world pixel coordinates and adds them as shapes
        to a provided group. Requires a minimum of three coordinates per polygon.

        Parameters:
            data:
                A pandas DataFrame-like object containing the following required attributes:
                - mask: Iterable object representing polygonal mask coordinates in relative or local
                  space.
                - label: Integer or categorical identifier for the class associated with the mask.
                - score: Float indicating confidence or accuracy associated with the predicted label.
                - zone_to_world: Transformation matrix for converting local mask coordinates to world
                  space.
            shapes_group:
                The group within which to add the created polygon shapes.

        Returns:
            list:
                A list of created shape objects corresponding to valid polygons derived from the mask
                data.

        Raises:
            KeyError:
                If expected attributes (mask, label, score, zone_to_world) are missing in `data`.
            IndexError:
                If index access or dimension mismatch occurs during transformation.

        Note:
            - Each valid shape corresponds to a polygon with at least three coordinates. Invalid
              polygons (less than three coordinates) will be skipped, accompanied by a printed
              warning.
            - It is assumed that the coordinate transformation uses a 3x3 matrix and formats input
              points as homogenous coordinates.
        """

        outline_shapes = []
        for row in data.itertuples():
            mask = row.mask
            label = row.label
            score = row.score
            to_world = row.zone_to_world

            # Преобразование координат масок в координаты пикселей мира
            corners = []
            for coord in mask:
                x, y = coord  # ожидается, что coord - список или кортеж с двумя значениями
                transformed_coord = to_world @ np.array([x + 0.5, y + 0.5, 1]).reshape(3, 1)
                p = Metashape.Vector([transformed_coord[0, 0], transformed_coord[1, 0]])
                p = Metashape.CoordinateSystem.transform(p, self.chunk.orthomosaic.crs, self.chunk.shapes.crs)
                corners.append([p.x, p.y])

            if len(corners) >= 3:  # Полигон должен иметь по крайней мере три координаты
                shape = self.chunk.shapes.addShape()
                shape.group = shapes_group
                shape.geometry = Metashape.Geometry.Polygon(corners)
                shape.label = self.classes[int(label)]
                shape.attributes["Score"] = str(score)
                outline_shapes.append(shape)
            else:
                print(f"Invalid polygon with less than 3 coordinates: {corners}")
        return outline_shapes

    def apply_union_mask_shapes(self, chunk, new_shapes, detected_label):
        """
        Applies a union mask to the given shapes in a 3D chunk model, organizing the new
        shapes into a labeled group within the chunk. This method processes a collection
        of geometric shapes, creates new polygonal shapes from their corner coordinates,
        and assigns them to a labeled group with specific display settings.

        Parameters:
            chunk (Metashape.Chunk): The 3D chunk object where shapes will be modified and
                grouped.
            new_shapes (list): A list of shapes, where each shape is represented as a
                collection of corner coordinates (list of tuples).
            detected_label (str): Label to be assigned to the created shapes group for
                organizational purposes, helping to identify added shapes.
        """
        shapes_layer = self.chunk.shapes.addGroup()
        shapes_layer.label = detected_label
        shapes_layer.show_labels = False
        for shape in new_shapes:
            new_shape = chunk.shapes.addShape()
            new_shape.group = shapes_layer
            corners = shape
            new_shape.geometry = Metashape.Geometry.Polygon([Metashape.Vector([x, y]) for x, y in corners])

    def apply_union_boxes_shapes(self, chunk, new_shapes, detected_label):
        """
        Applies union boxes' shapes to a given chunk and assigns them to a new layer.

        This method works by creating a new shapes group to represent the bounding boxes.
        It assigns the provided label to the new shapes layer and populates it with
        polygonal shapes defined by the input list of coordinate tuples. Each shape is
        transformed from the orthomosaic coordinate reference system to the shapes
        coordinate reference system before being added to the group.

        Parameters:
        chunk (Metashape.Chunk): The target chunk to which the shapes are added.
        new_shapes (list): A list of bounding box coordinates. Each item is a tuple
            containing (xmin, ymin, xmax, ymax) that represents the corners of a
            rectangle.
        detected_label (str): The label to assign to the newly created shapes layer.
        """
        shapes_layer = self.chunk.shapes.addGroup()
        shapes_layer.label = detected_label
        shapes_layer.show_labels = False

        for shape in new_shapes:
            xmin, ymin, xmax, ymax = shape
            corners = []
            for x, y in [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]:
                p = Metashape.Vector([x, y])
                p = Metashape.CoordinateSystem.transform(p, self.chunk.orthomosaic.crs, self.chunk.shapes.crs)
                corners.append([p.x, p.y])

            new_shape = chunk.shapes.addShape()
            new_shape.group = shapes_layer
            new_shape.geometry = Metashape.Geometry.Polygon(corners)


    def detect(self):

        app = QtWidgets.QApplication.instance()

        print("Detection All Opp...")
        time_start = time.time()

        if not self.model:
            print("No init model!")
            return
        # Напечатать устройство для проверки
        device = get_device()
        print(f"Using device: {device}")

        nentity_detected = 0

        big_tiles_k = 8
        border = self.patch_inner_border
        area_overlap_threshold = 0.60

        detected_label = self.layer_name_detection_data + "box_detected ({:.2f} cm/pix, ".format(
            100.0 * self.orthomosaic_resolution)
        detected_label += f"size img: {self.max_image_size}, threshold: {self.detection_score_threshold})"
        detected_shapes_layer = self.chunk.shapes.addGroup()
        detected_shapes_layer.label = detected_label

        detected_mask_label = self.layer_name_detection_data + "outline_detected ({:.2f} cm/pix, ".format(
            100.0 * self.orthomosaic_resolution)
        detected_mask_label += f"size img: {self.max_image_size}, threshold: {self.detection_score_threshold})"
        detected_shapes_mask_layer = self.chunk.shapes.addGroup()
        detected_shapes_mask_layer.label = detected_mask_label

        big_tiles = set()
        for tile_x in range(self.tile_min_x, self.tile_max_x + 1):
            for tile_y in range(self.tile_min_y, self.tile_max_y + 1):
                if (tile_x, tile_y) not in self.tiles_paths:
                    continue
                big_tile_x, big_tile_y = tile_x // big_tiles_k, tile_y // big_tiles_k
                big_tiles.add((big_tile_x, big_tile_y))

        bigtiles_entity = {}
        bigtiles_to_world = {}
        bigtiles_idx_on_borders = {}

        for big_tile_index, (big_tile_x, big_tile_y) in enumerate(sorted(big_tiles)):
            big_tile = np.zeros(
                (border + big_tiles_k * self.patch_size + border, border + big_tiles_k * self.patch_size + border, 3),
                np.uint8)
            big_tile[:, :, :] = 255
            big_tile_to_world = None

            for xi in range(-1, big_tiles_k + 1):
                for yi in range(-1, big_tiles_k + 1):
                    tile_x, tile_y = big_tiles_k * big_tile_x + xi, big_tiles_k * big_tile_y + yi
                    if (tile_x, tile_y) not in self.tiles_paths:
                        continue
                    part = cv2.imread(self.tiles_paths[tile_x, tile_y])
                    part = cv2.copyMakeBorder(part, 0, self.patch_size - part.shape[0], 0,
                                              self.patch_size - part.shape[1], cv2.BORDER_CONSTANT,
                                              value=[255, 255, 255])
                    if xi in [-1, big_tiles_k] or yi in [-1, big_tiles_k]:
                        fromx, fromy = border + xi * self.patch_size, border + yi * self.patch_size
                        tox, toy = fromx + self.patch_size, fromy + self.patch_size
                        if xi == -1:
                            part = part[:, self.patch_size - border:, :]
                            fromx += self.patch_size - border
                        if xi == big_tiles_k:
                            part = part[:, :border, :]
                            tox = fromx + border
                        if yi == -1:
                            part = part[self.patch_size - border:, :, :]
                            fromy += self.patch_size - border
                        if yi == big_tiles_k:
                            part = part[:border, :, :]
                            toy = fromy + border
                        big_tile[fromy:toy, fromx:tox, :] = part
                    else:
                        big_tile[border + yi * self.patch_size:, border + xi * self.patch_size:, :][:self.patch_size,
                        :self.patch_size, :] = part
                        big_tile_to_world = self.add_pixel_shift(self.tiles_to_world[tile_x, tile_y],
                                                                 -(border + xi * self.patch_size),
                                                                 -(border + yi * self.patch_size))

            assert big_tile_to_world is not None

            subtiles_entity = {}
            tile_inner_size = self.patch_size - 2 * border
            inner_tiles_nx = (big_tile.shape[1] - 2 * border + tile_inner_size - 1) // tile_inner_size
            inner_tiles_ny = (big_tile.shape[0] - 2 * border + tile_inner_size - 1) // tile_inner_size
            for xi in range(inner_tiles_nx):
                for yi in range(inner_tiles_ny):
                    tox, toy = min(big_tile.shape[1], 2 * border + (xi + 1) * tile_inner_size), min(big_tile.shape[0],
                                                                                                    2 * border + (
                                                                                                            yi + 1) * tile_inner_size)
                    fromx, fromy = tox - self.patch_size, toy - self.patch_size
                    subtile = big_tile[fromy:toy, fromx:tox, :]

                    white_pixels_fraction = np.sum(np.all(subtile == 255, axis=-1)) / (
                            subtile.shape[0] * subtile.shape[1])

                    assert (subtile.shape == (self.patch_size, self.patch_size, 3))

                    save = False
                    save_conf = False
                    save_txt = False
                    save_crop = False

                    if self.isDebugMode:
                        save = True
                        save_conf = True
                        save_txt = True
                        save_crop = True

                    # Выполнение предсказания
                    with torch.no_grad():
                        subtile_prediction = self.model.predict(subtile, imgsz=self.max_image_size, device=device,
                                                           conf=self.detection_score_threshold, iou=0.45,
                                                           project=self.dir_subtiles_results, save=save,
                                                           save_conf=save_conf,
                                                           save_txt=save_txt, save_crop=save_crop, half=True, )

                    Metashape.app.update()
                    app.processEvents()
                    self.check_stopped()

                    subtile_prediction = subtile_prediction[0].cpu()

                    if subtile_prediction.boxes is not None:
                        # Получение размеров изображения (из results, оригинальные размеры фото)
                        original_shape = subtile_prediction.orig_shape  # Исходные размеры фото (высота, ширина)
                        orig_h, orig_w = original_shape[:2]

                        masks = None
                        # Получаем данные из предсказания
                        if subtile_prediction.masks is not None:
                            masks = subtile_prediction.masks.xyn

                        boxes = subtile_prediction.boxes.xyxyn
                        # labels = subtile_entity.boxes.cls.data.numpy()

                        subtile_inner_entity = pd.DataFrame(
                            columns=['image_path', 'xmin', 'ymin', 'xmax', 'ymax', 'label', 'score', 'mask'])


                        # print(f"Predict: {subtile_entity}")
                        for idx, bbox in enumerate(boxes):
                            box = subtile_prediction.boxes[idx]
                            score = box.conf.numpy()[0]
                            b_class = box.cls.numpy()
                            label = b_class[0]

                            xmin, ymin, xmax, ymax = bbox
                            xmin, ymin, xmax, ymax = int(xmin * orig_w), int(ymin * orig_h), int(xmax * orig_w), int(
                                ymax * orig_h)

                            row = {'image_path': f"{big_tile_index}_{xi}_{yi}_{idx}", 'xmin': xmin, 'ymin': ymin,
                                   'xmax': xmax, 'ymax': ymax, 'label': label, 'score': score, 'mask': []}

                            mask_contour = []

                            if masks is not None and len(masks) > 0:
                                # Добавление маски
                                mask = masks[idx]

                                # Преобразуем нормализованные координаты в пиксели
                                pixel_coords = np.array([[int(y * orig_h), int(x * orig_w)] for y, x in mask],
                                                        dtype=np.int32)

                                # Создаём пустую бинарную маску и рисуем заполненный полигон
                                binary_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
                                cv2.fillPoly(binary_mask, [pixel_coords], color=255)  # Заполняем маску белым цветом (255)

                                # Получение контуров используя cv2.findContours
                                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                                if contours:
                                    # Вычисление площади каждого контура
                                    areas = [cv2.contourArea(contour) for contour in contours]

                                    # Нахождение индекса контура с наибольшей площадью
                                    largest_contour_index = np.argmax(areas)

                                    # Выбор контура с наибольшей площадью
                                    largest_contour = contours[largest_contour_index]

                                    # Сдвигаем координаты точек контура на значения fromx и fromy
                                    shifted_contour = [[x + fromx, y + fromy] for [[x, y]] in largest_contour]

                                    # Добавляем смещенные точки в mask_contour
                                    mask_contour.extend(shifted_contour)


                                row['mask'] = mask_contour

                            if self.isDebugMode:

                                cv2.drawContours(subtile, contours, -1, (0, 255, 0),
                                                 2)  # Отрисовать контуры зелёным цветом
                                if mask_contour:
                                    cv_contour = np.array(mask_contour, dtype=np.int32).reshape((-1, 1, 2))
                                    cv2.drawContours(subtile, [cv_contour], -1, (0, 0, 255), 2)  # Красный цвет

                            if xmin >= self.patch_size - border or xmax <= border or ymin >= self.patch_size - border or ymax <= border:
                                continue

                            xmin, xmax = map(lambda x: fromx + x, [xmin, xmax])
                            ymin, ymax = map(lambda y: fromy + y, [ymin, ymax])


                            row['xmin'], row['ymin'], row['xmax'], row['ymax'] = xmin, ymin, xmax, ymax
                            subtile_inner_entity = pandas_append(subtile_inner_entity, pd.DataFrame([row]),
                                                                ignore_index=True)

                        if self.isDebugMode:

                            cv2.imwrite(
                                self.dir_subtiles_results + "{}-{}-{}-{}.jpg".format(big_tile_x, big_tile_y, xi, yi),
                                subtile)


                    else:
                        subtile_inner_entity = pd.DataFrame(
                            columns=['image_path', 'xmin', 'ymin', 'xmax', 'ymax', 'label', 'score', 'mask'])

                        if self.isDebugMode:
                            cv2.imwrite(self.dir_subtiles_results + "{}-{}-{}-{}_empty.jpg".format(big_tile_x, big_tile_y, xi, yi), subtile)

                    subtiles_entity[xi, yi] = subtile_inner_entity

            big_tile_entity = None
            for xi, yi in sorted(subtiles_entity.keys()):
                tox, toy = min(big_tile.shape[1], 2 * border + (xi + 1) * tile_inner_size), min(big_tile.shape[0],
                                                                                                2 * border + (
                                                                                                        yi + 1) * tile_inner_size)
                fromx, fromy = tox - self.patch_size, toy - self.patch_size

                a = subtiles_entity[xi, yi]

                a_idx_on_border = []
                for idx, rowA in a.iterrows():
                    axmin, aymin, axmax, aymax, ascore = int(rowA.xmin), int(rowA.ymin), int(rowA.xmax), int(
                        rowA.ymax), rowA.score
                    if axmin > fromx + border and axmax < tox - border and aymin > fromy + border and aymax < toy - border:
                        continue
                    a_idx_on_border.append(idx)

                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = xi + dx, yi + dy
                        if (nx, ny) not in subtiles_entity:
                            continue
                        b = subtiles_entity[nx, ny]

                        indices_to_check = a_idx_on_border

                        # because the last two columns/rows have much bigger overlap
                        if (xi == inner_tiles_nx - 2 and dx == 1) or (xi == inner_tiles_nx - 1 and dx == -1) \
                                or (yi == inner_tiles_ny - 2 and dy == 1) or (yi == inner_tiles_ny - 1 and dy == -1):
                            indices_to_check = a.index

                        for idx in indices_to_check:
                            rowA = a.loc[idx]
                            if rowA.label == "Suppressed":
                                continue
                            axmin, aymin, axmax, aymax, ascore = int(rowA.xmin), int(rowA.ymin), int(rowA.xmax), int(
                                rowA.ymax), rowA.score
                            areaA = (axmax - axmin) * (aymax - aymin)
                            for _, rowB in b.iterrows():
                                bxmin, bymin, bxmax, bymax, bscore = int(rowB.xmin), int(rowB.ymin), int(
                                    rowB.xmax), int(rowB.ymax), rowB.score
                                areaB = (bxmax - bxmin) * (bymax - bymin)

                                intersectionx = max(0, min(axmax, bxmax) - max(axmin, bxmin))
                                intersectiony = max(0, min(aymax, bymax) - max(aymin, bymin))
                                intersectionArea = intersectionx * intersectiony

                                if 'label' not in a.columns:
                                    a['label'] = None
                                else:
                                    a['label'] = a['label'].astype('object')

                                if intersectionArea > min(areaA, areaB) * area_overlap_threshold:

                                    if ascore + 0.2 < bscore:
                                        a.loc[idx, 'label'] = "Suppressed"
                                    elif not (bscore + 0.2 < ascore):
                                        if areaA < areaB:
                                            a.loc[idx, 'label'] = "Suppressed"
                                        elif not (areaB < areaA) and (xi, yi) < (nx, ny):
                                            assert not ((nx, ny) < (xi, yi))
                                            a.loc[idx, 'label'] = "Suppressed"

                if big_tile_entity is None:
                    big_tile_entity = pd.DataFrame(columns=a.columns)
                for idx, row in a.iterrows():
                    if row.label == "Suppressed":
                        continue
                    big_tile_entity = pandas_append(big_tile_entity, row, ignore_index=True)

            idx_on_borders = []
            for idx, rowA in big_tile_entity.iterrows():
                axmin, aymin, axmax, aymax, ascore = int(rowA.xmin), int(rowA.ymin), int(rowA.xmax), int(
                    rowA.ymax), rowA.score
                if axmin > 2 * border and axmax < big_tiles_k * self.patch_size and aymin > 2 * border and aymax < big_tiles_k * self.patch_size:
                    continue
                idx_on_borders.append(idx)

            bigtiles_entity[big_tile_x, big_tile_y] = big_tile_entity
            bigtiles_to_world[big_tile_x, big_tile_y] = big_tile_to_world
            bigtiles_idx_on_borders[big_tile_x, big_tile_y] = idx_on_borders


            self.detectionPBar.setValue((big_tile_index + 1) * 100 / len(big_tiles))
            Metashape.app.update()
            app.processEvents()
            self.check_stopped()

        for big_tile_x, big_tile_y in sorted(big_tiles):
            big_tile_entity = bigtiles_entity[big_tile_x, big_tile_y]
            if big_tile_entity is None:
                continue

            big_tile_to_world = bigtiles_to_world[big_tile_x, big_tile_y]

            a_idx_on_borders = bigtiles_idx_on_borders[big_tile_x, big_tile_y]

            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = big_tile_x + dx, big_tile_y + dy
                    if (nx, ny) not in bigtiles_entity:
                        continue
                    b = bigtiles_entity[nx, ny]
                    if b is None:
                        continue

                    b_idx_on_borders = bigtiles_idx_on_borders[nx, ny]

                    for idxA in a_idx_on_borders:
                        rowA = big_tile_entity.loc[idxA]
                        if rowA.label == "Suppressed":
                            continue
                        axmin, aymin, axmax, aymax, ascore = int(rowA.xmin), int(rowA.ymin), int(rowA.xmax), int(
                            rowA.ymax), rowA.score
                        areaA = (axmax - axmin) * (aymax - aymin)
                        for idxB in b_idx_on_borders:
                            rowB = b.loc[idxB]
                            bxmin, bymin, bxmax, bymax, bscore = int(rowB.xmin), int(rowB.ymin), int(rowB.xmax), int(
                                rowB.ymax), rowB.score
                            bxmin, bxmax = map(lambda x: x + dx * big_tiles_k * self.patch_size, [bxmin, bxmax])
                            bymin, bymax = map(lambda y: y + dy * big_tiles_k * self.patch_size, [bymin, bymax])
                            areaB = (bxmax - bxmin) * (bymax - bymin)

                            intersectionx = max(0, min(axmax, bxmax) - max(axmin, bxmin))
                            intersectiony = max(0, min(aymax, bymax) - max(aymin, bymin))
                            intersectionArea = intersectionx * intersectiony
                            if intersectionArea > min(areaA, areaB) * area_overlap_threshold:
                                if ascore + 0.2 < bscore:
                                    big_tile_entity.loc[idxA, 'label'] = "Suppressed"
                                elif not (bscore + 0.2 < ascore):
                                    if areaA < areaB:
                                        big_tile_entity.loc[idxA, 'label'] = "Suppressed"
                                    elif not (areaB < areaA) and (big_tile_x, big_tile_y) < (nx, ny):
                                        assert not ((nx, ny) < (big_tile_x, big_tile_y))
                                        big_tile_entity.loc[idxA, 'label'] = "Suppressed"

            big_tile_entity = big_tile_entity[big_tile_entity.label != "Suppressed"]

            nentity_detected += len(big_tile_entity)

            self.add_entity_boxes(big_tile_to_world, big_tile_entity, detected_shapes_layer)

            self.add_entity_masks(big_tile_to_world, big_tile_entity, detected_shapes_mask_layer)

        Metashape.app.update()

    def add_entity_boxes(self, to_world, tile_entity, shapes_group):

        for row in tile_entity.itertuples():
            xmin, ymin, xmax, ymax, label, score = int(row.xmin), int(row.ymin), int(row.xmax), int(
                row.ymax), row.label, row.score

            corners = []
            for x, y in [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]:
                x, y = to_world @ np.array([x + 0.5, y + 0.5, 1]).reshape(3, 1)
                p = Metashape.Vector([x, y])
                p = Metashape.CoordinateSystem.transform(p, self.chunk.orthomosaic.crs, self.chunk.shapes.crs)
                corners.append([p.x, p.y])

            shape = self.chunk.shapes.addShape()
            shape.group = shapes_group
            shape.geometry = Metashape.Geometry.Polygon(corners)

            shape.label = self.classes[int(label)]
            shape.attributes["Score"] = str(score)

    def add_entity_masks(self, to_world, tile_entity, shapes_group):

        for row in tile_entity.itertuples():
            label, score, mask = row.label, row.score, row.mask

            # Преобразование координат масок в координаты пикселей мира
            corners = []
            for coord in mask:
                x, y = coord  # ожидается, что coord - список или кортеж с двумя значениями
                transformed_coord = to_world @ np.array([x + 0.5, y + 0.5, 1]).reshape(3, 1)
                p = Metashape.Vector([transformed_coord[0, 0], transformed_coord[1, 0]])
                p = Metashape.CoordinateSystem.transform(p, self.chunk.orthomosaic.crs, self.chunk.shapes.crs)
                corners.append([p.x, p.y])
            if len(corners) == 0:
                continue

            if len(corners) >= 3:  # Полигон должен иметь по крайней мере три координаты
                shape = self.chunk.shapes.addShape()
                shape.group = shapes_group
                shape.geometry = Metashape.Geometry.Polygon(corners)
                shape.label = self.classes[int(label)]
                shape.attributes["Score"] = str(score)
            else:
                print(f"Invalid polygon with less than 3 coordinates: {corners}")


    def show_results_dialog(self):
        message = "Finished in {:.2f} sec:\n".format(self.results_time_total)

        print(message)
        Metashape.app.messageBox(message)

    def create_gui(self):
        self.labelDetectZonesLayer = QtWidgets.QLabel("Select zones:")
        self.detectZonesLayer = QtWidgets.QComboBox()
        self.noDataChoice = (None, "No additional (use as is)", True)
        self.layers = [self.noDataChoice]

        slow_shape_layers_enumerating_but_with_number_of_shapes = False
        if slow_shape_layers_enumerating_but_with_number_of_shapes:

            print("Enumerating all shape layers...")

            shapes_enumerating_start = time.time()
            self.layersDict = {}
            self.layersSize = {}
            shapes = self.chunk.shapes

            for shape in shapes:
                layer = shape.group
                if layer.key not in self.layersDict:
                    self.layersDict[layer.key] = (layer.key, layer.label, layer.enabled)
                    self.layersSize[layer.key] = 1
                else:
                    self.layersSize[layer.key] += 1

            print("Found {} shapes layers in {:.2f} sec:".format(len(self.layersDict),
                                                                 time.time() - shapes_enumerating_start))
            for key in sorted(self.layersDict.keys()):
                key, label, enabled = self.layersDict[key]
                size = self.layersSize[key]

                print("Shape layer: {} shapes, key={}, label={}".format(size, key, label))

                if label == '':
                    label = 'Layer'

                label = label + " ({} shapes)".format(size)
                self.layers.append((key, label, enabled))

            self.layersDict = None
            self.layersSize = None

        else:
            if self.chunk.shapes is None:
                print("No shapes")
            else:
                for layer in self.chunk.shapes.groups:
                    key, label, enabled = layer.key, layer.label, layer.enabled

                    print("Shape layer: key={}, label={}, enabled={}".format(key, label, enabled))

                    if label == '':
                        label = 'Layer'

                    self.layers.append((key, label, layer.enabled))

        for key, label, enabled in self.layers:
            self.detectZonesLayer.addItem(label)

        self.detectZonesLayer.setCurrentIndex(0)

        for i, (key, label, enabled) in enumerate(self.layers):
            if not enabled:
                continue
            if label.lower().startswith(self.expected_layer_name_train_zones.lower()):
                self.detectZonesLayer.setCurrentIndex(i)

        self.chkUse5mmResolution = QtWidgets.QCheckBox("Process with 0.50 cm/pix resolution")
        self.chkUse5mmResolution.setToolTip(
            "Process with downsampling to 0.50 cm/pix instad of original orthomosaic resolution.")
        self.chkUse5mmResolution.setChecked(not self.prefer_original_resolution)

        self.groupBoxGeneral = QtWidgets.QGroupBox("General")
        generalLayout = QtWidgets.QGridLayout()

        self.labelWorkingDir = QtWidgets.QLabel()
        self.labelWorkingDir.setText("Working dir:")
        self.workingDirLineEdit = QtWidgets.QLineEdit()
        self.workingDirLineEdit.setText(self.working_dir)
        self.workingDirLineEdit.setPlaceholderText("Path to dir for intermediate data")
        self.workingDirLineEdit.setToolTip("Path to dir for intermediate data")
        self.btnWorkingDir = QtWidgets.QPushButton("...")
        self.btnWorkingDir.setFixedSize(25, 25)
        QtCore.QObject.connect(self.btnWorkingDir, QtCore.SIGNAL("clicked()"), lambda: self.choose_working_dir())
        generalLayout.addWidget(self.labelWorkingDir, 0, 0)
        generalLayout.addWidget(self.workingDirLineEdit, 0, 1)
        generalLayout.addWidget(self.btnWorkingDir, 0, 2)

        generalLayout.addWidget(self.chkUse5mmResolution, 1, 1)

        self.debugModeCbox = QtWidgets.QCheckBox("Debug mode")
        generalLayout.addWidget(self.debugModeCbox, 4, 1, 1, 2)

        self.maxSizeImageSpinBox = QtWidgets.QSpinBox(self)
        self.maxSizeImageSpinBox.setMaximumWidth(150)
        self.maxSizeImageSpinBox.setMinimum(256)
        self.maxSizeImageSpinBox.setMaximum(2048)
        self.maxSizeImageSpinBox.setSingleStep(256)
        self.maxSizeImageSpinBox.setValue(1024)
        self.maxSizeImageLabel = QtWidgets.QLabel("Max size image:")
        generalLayout.addWidget(self.maxSizeImageLabel, 5, 0)
        generalLayout.addWidget(self.maxSizeImageSpinBox, 5, 1, 1, 2)

        self.loadTilesLabel = QtWidgets.QLabel("Load tiles from working dir:")
        generalLayout.addWidget(self.loadTilesLabel, 6, 0)
        self.loadTilesCbox = QtWidgets.QCheckBox('Off')
        generalLayout.addWidget(self.loadTilesCbox, 6, 1)
        generalLayout.addWidget(self.labelDetectZonesLayer, 7, 0)
        generalLayout.addWidget(self.detectZonesLayer, 7, 1, 1, 2)

        self.groupBoxGeneral.setLayout(generalLayout)
        # Создаем таб-панель
        self.tabWidget = QtWidgets.QTabWidget()

        self.modelLoadPathLabel = QtWidgets.QLabel()
        self.modelLoadPathLabel.setText("Load model from:")
        self.modelLoadPathLineEdit = QtWidgets.QLineEdit()
        self.modelLoadPathLineEdit.setText(self.load_model_path)
        self.modelLoadPathLineEdit.setPlaceholderText(
            "File with previously saved neural network model (resolution must be the same)")
        self.modelLoadPathLineEdit.setToolTip(
            "File with previously saved neural network model (resolution must be the same)")
        self.btnModelLoadPath = QtWidgets.QPushButton("...")
        self.btnModelLoadPath.setFixedSize(25, 25)
        QtCore.QObject.connect(self.btnModelLoadPath, QtCore.SIGNAL("clicked()"), lambda: self.choose_model_load_path())
        generalLayout.addWidget(self.modelLoadPathLabel, 8, 0)
        generalLayout.addWidget(self.modelLoadPathLineEdit, 8, 1)
        generalLayout.addWidget(self.btnModelLoadPath, 8, 2)

        self.tabModelPrediction = QtWidgets.QWidget()
        self.tabWidget.addTab(self.tabModelPrediction, "Model detection")
        detectionLayout = QtWidgets.QGridLayout()

        # detection_score_threshold
        self.scoreThresholdLabel = QtWidgets.QLabel("Threshold:")
        self.scoreThresholdLabel.setFixedWidth(130)
        self.scoreThresholdLabel.setAlignment(QtCore.Qt.AlignLeft)
        self.scoreThresholdSpinBox = CustomDoubleSpinBox()
        self.scoreThresholdSpinBox.setMaximumWidth(150)
        self.scoreThresholdSpinBox.setRange(0, 1)
        self.scoreThresholdSpinBox.setSingleStep(0.0001)
        self.scoreThresholdSpinBox.setDecimals(5)
        self.scoreThresholdSpinBox.setValue(self.detection_score_threshold)

        detectionLayout.addWidget(self.scoreThresholdLabel, 0, 0)
        detectionLayout.addWidget(self.scoreThresholdSpinBox, 0, 1, 1, 2)

        self.btnDetect = QtWidgets.QPushButton("Detection")
        self.btnDetect.setMaximumWidth(100)
        self.btnStop = QtWidgets.QPushButton("Stop")
        self.btnStop.setEnabled(False)
        self.btnStop.setMaximumWidth(100)

        layout = QtWidgets.QGridLayout()
        row = 0

        layout.addWidget(self.groupBoxGeneral, row, 0, 1, 3)
        row += 1

        self.tabModelPrediction.setLayout(detectionLayout)
        layout.addWidget(self.tabWidget, row, 0, 1, 3)
        row += 1

        self.txtInfoPBar = QtWidgets.QLabel()
        self.txtInfoPBar.setText("")
        layout.addWidget(self.txtInfoPBar, row, 1, 1, 3)
        row += 1

        self.txtDetectionPBar = QtWidgets.QLabel()
        self.txtDetectionPBar.setText("Progress:")
        self.detectionPBar = QtWidgets.QProgressBar()
        self.detectionPBar.setTextVisible(True)
        layout.addWidget(self.txtDetectionPBar, row, 0)
        layout.addWidget(self.detectionPBar, row, 1, 1, 2)
        row += 1

        layout.addWidget(self.btnDetect, row, 1)
        layout.addWidget(self.btnStop, row, 3)
        row += 1

        self.setLayout(layout)

        QtCore.QObject.connect(self.btnDetect, QtCore.SIGNAL("clicked()"), lambda: self.run_detect())
        QtCore.QObject.connect(self.btnStop, QtCore.SIGNAL("clicked()"), lambda: self.stop())

        self.debugModeCbox.stateChanged.connect(self.change_debug_mode)
        self.loadTilesCbox.stateChanged.connect(self.change_load_tiles)

    def change_debug_mode(self, value):
        self.isDebugMode = value
        print(f"Debug mode: {'On' if value else 'Off'}")

    def change_load_tiles(self, value):
        self.isLoadTiles = value
        self.loadTilesCbox.setText('On' if value else 'Off')

    def choose_working_dir(self):
        working_dir = Metashape.app.getExistingDirectory()
        self.workingDirLineEdit.setText(working_dir)

    def choose_model_save_path(self):
        models_dir = ""
        load_path = Metashape.app.settings.value("scripts/yolo/model_load_path")
        if load_path is not None:
            models_dir = str(pathlib.Path(load_path).parent)

        save_path = Metashape.app.getSaveFileName("Trained model save path", models_dir,
                                                  "Model Files (*.model *.pth *.pt);;All Files (*)")
        if len(save_path) <= 0:
            return

        self.modelSavePathLineEdit.setText(save_path)

    def choose_model_load_path(self):
        load_path = Metashape.app.getOpenFileName("Trained model load path", "",
                                                  "Model Files (*.model *.pth *.pt);;All Files (*)")
        self.modelLoadPathLineEdit.setText(load_path)

    def load_params(self):

        app = QtWidgets.QApplication.instance()
        self.prefer_original_resolution = not self.chkUse5mmResolution.isChecked()

        self.max_image_size = self.maxSizeImageSpinBox.value()
        self.preferred_patch_size = self.max_image_size

        Metashape.app.settings.setValue("scripts/yolo/max_image_size", str(self.max_image_size))

        if not self.prefer_original_resolution:
            self.orthomosaic_resolution = self.preferred_resolution
            self.patch_size = self.preferred_patch_size
        else:
            self.orthomosaic_resolution = self.chunk.orthomosaic.resolution

            if self.orthomosaic_resolution > 0.105:
                raise Exception("Orthomosaic should have resolution <= 10 cm/pix.")
            if self.force_small_patch_size:
                patch_size_multiplier = 1
            else:
                patch_size_multiplier = max(1, min(4, self.preferred_resolution / self.orthomosaic_resolution))

            self.patch_size = round(self.preferred_patch_size * patch_size_multiplier)

        self.patch_inner_border = self.patch_size // 8

        print("Using resolution {} m/pix with patch {}x{}".format(self.orthomosaic_resolution, self.patch_size,
                                                                  self.patch_size))
        self.working_dir = self.workingDirLineEdit.text()

        self.load_model_path = self.modelLoadPathLineEdit.text()

        self.detection_score_threshold = self.scoreThresholdSpinBox.value()

        trainZonesLayer = self.layers[self.detectZonesLayer.currentIndex()]

        if trainZonesLayer == self.noDataChoice:
            self.train_on_user_data_enabled = False
            print("Additional neural network disabled")
        else:
            self.train_on_user_data_enabled = True
            print("Additional neural network detecting expected on key={} layer zones".format(trainZonesLayer[0]))

        print("Loading shapes...")

        loading_train_shapes_start = time.time()
        shapes = self.chunk.shapes
        self.detected_zones = []

        print(f"All shapes chunk: {len(shapes)}")

        # Получаем ключи слоев для тренировки
        train_zones_key = trainZonesLayer[0]

        # Группируем shapes по ключу слоя
        #grouped_shapes = itertools.groupby(sorted(shapes, key=lambda x: x.group.key), key=lambda x: x.group.key)

        # Создаем два отдельных списка для train_zones и train_data
        for i, shape in enumerate(shapes):
            if shape.group.key == train_zones_key:
                self.detected_zones.append(shape)

            i += 1
            self.detectionPBar.setValue(int((100 * i + 1) / len(shapes)))
            app.processEvents()
            self.check_stopped()

        print("{} zones  loaded in {:.2f} sec".format(len(self.detected_zones),
                                                      time.time() - loading_train_shapes_start))

class CustomDoubleSpinBox(QtWidgets.QDoubleSpinBox):
    """
    Custom implementation of QDoubleSpinBox to modify text representation.

    This class provides a customized text representation for the numerical
    values of a QDoubleSpinBox. It removes trailing zeroes and unnecessary
    decimal points from the displayed value for cleaner visualization.

    Attributes
    ----------
    Inherited from QDoubleSpinBox.
    """
    def textFromValue(self, value):
        import re
        text = super(CustomDoubleSpinBox, self).textFromValue(value)
        return re.sub(r'0*$', '', re.sub(r'\.0*$', '', text))

class WindowConvertShapes(QtWidgets.QDialog):

    def __init__(self, parent):
        QtWidgets.QDialog.__init__(self, parent)
        self.setWindowTitle("Convert shapes on orthomosaic")
        self.chunk = Metashape.app.document.chunk

        self.data_zones = []
        self.data_shapes = []
        self.results_time_total = None
        self.stopped = False
        self.selected_mode = None
        self.sizeBox = 0.4

        self.isLabelFromShape = True

        # Список для хранения всех QComboBox (dataLayer)
        self.dataLayerList:list[QtWidgets.QComboBox()] = []

        # Инициализация элементов
        self.labelZonesLayer = QtWidgets.QLabel("Layer zones:")
        self.zonesLayer = QtWidgets.QComboBox()
        self.labelDataLayer = QtWidgets.QLabel("Layer data:")
        self.dataLayer = QtWidgets.QComboBox()
        self.dataLayerList.append(self.dataLayer)  # Добавляем первый QComboBox в список

        # self.btnPointsToBoxes = QtWidgets.QPushButton("Points to boxes")
        # self.btnPointsToBoxes.setMaximumWidth(100)

        self.btnAddDataLayer = QtWidgets.QPushButton("Add Data Layer")  # Кнопка добавления нового dataLayer

        self.btnRun = QtWidgets.QPushButton("Run")
        self.btnRun.setMaximumWidth(100)
        self.btnRun.setEnabled(False)

        self.btnStop = QtWidgets.QPushButton("Stop")
        self.btnStop.setEnabled(False)
        self.btnStop.setMaximumWidth(100)

        # Создаем чекбоксы
        self.cbxPointsToBoxes = QtWidgets.QCheckBox("Points to boxes")
        self.cbxPointsToBoxes.setChecked(False)
        self.cbxBoxesToPoints = QtWidgets.QCheckBox("Boxes to points")
        self.cbxPointsToBoxes.setDisabled(False)

        # Объединяем в группу
        self.buttonGroup = QtWidgets.QButtonGroup(self)
        self.buttonGroup.setExclusive(True)  # Устанавливаем, что выбор только один
        self.buttonGroup.addButton(self.cbxPointsToBoxes, 0)
        self.buttonGroup.addButton(self.cbxBoxesToPoints, 1)

        self.labelSizeBox = QtWidgets.QLabel("Size box (metres):")
        self.sizeSpinBox = CustomDoubleSpinBox()
        self.sizeSpinBox.setMaximumWidth(150)
        self.sizeSpinBox.setRange(0.1, 10)
        self.sizeSpinBox.setSingleStep(0.1)
        self.sizeSpinBox.setDecimals(5)
        self.sizeSpinBox.setValue(0.4)
        self.labelSizeBox.setVisible(False)
        self.sizeSpinBox.setVisible(False)

        self.txtPBar = QtWidgets.QLabel()
        self.txtPBar.setText("Progress:")
        self.mainPBar = QtWidgets.QProgressBar()
        self.mainPBar.setTextVisible(True)

        # Создаем чекбоксы
        self.labelFromLayer = QtWidgets.QCheckBox("Label From Layer")
        self.labelFromLayer.setChecked(False)
        self.labelFromShape = QtWidgets.QCheckBox("Label From Shape")
        self.labelFromShape.setChecked(True)

        # Объединяем в группу
        self.buttonGroupLabel = QtWidgets.QButtonGroup(self)
        self.buttonGroupLabel.setExclusive(True)  # Устанавливаем, что выбор только один
        self.buttonGroupLabel.addButton(self.labelFromLayer, 0)
        self.buttonGroupLabel.addButton(self.labelFromShape, 1)

        # Построение макета
        self.layout = QtWidgets.QGridLayout()
        self.layoutData = QtWidgets.QGridLayout()

        row = 0
        self.layout.addWidget(self.labelZonesLayer, row, 0)
        self.layout.addWidget(self.zonesLayer, row, 1, 1, 2)

        row += 1
        self.layout.addWidget(self.labelDataLayer, row, 0)
        # self.layout.addWidget(self.dataLayer, row, 1, 1, 2)

        self.layoutData.addWidget(self.dataLayer, 0, 1, 1, 2)

        self.layout.addLayout(self.layoutData, row,1,1, 2)

        self.row_pointer = 1  # Указатель на текущий ряд в макете (для добавления новых dataLayer)

        row += 1
        self.layout.addWidget(self.btnAddDataLayer, row, 0, 1, 3)  # Кнопка добавления dataLayer

        row += 1
        self.layout.addWidget(self.labelFromLayer, row, 0)

        row += 1
        self.layout.addWidget(self.labelFromShape, row, 0)

        row += 1
        self.layout.addWidget(self.cbxPointsToBoxes, row, 0)

        row += 1
        self.layout.addWidget(self.cbxBoxesToPoints, row, 0)

        row += 1
        self.layout.addWidget(self.labelSizeBox, row, 0)
        self.layout.addWidget(self.sizeSpinBox, row, 1, 1, 2)

        row += 1
        self.layout.addWidget(self.txtPBar, row, 0)
        self.layout.addWidget(self.mainPBar, row, 1, 1, 2)

        row += 1
        self.layout.addWidget(self.btnRun, row, 1)
        self.layout.addWidget(self.btnStop, row, 2)

        self.setLayout(self.layout)


        self.noDataChoice = (None, "No additional (use as is)", True)
        self.layers = [self.noDataChoice]

        if self.chunk.shapes is None:
            print("No shapes")
        else:
            for layer in self.chunk.shapes.groups:
                key, label, enabled = layer.key, layer.label, layer.enabled

                print("Shape layer: key={}, label={}, enabled={}".format(key, label, enabled))

                if label == '':
                    label = 'Layer'

                self.layers.append((key, label, layer.enabled))

        for key, label, enabled in self.layers:
            self.dataLayer.addItem(label)
            self.zonesLayer.addItem(label)

        self.dataLayer.setCurrentIndex(0)
        self.zonesLayer.setCurrentIndex(0)

        for i, (key, label, enabled) in enumerate(self.layers):
            if not enabled:
                continue
            if label.lower().startswith("zone"):
                self.zonesLayer.setCurrentIndex(i)
            elif label.lower().startswith("data"):
                self.dataLayer.setCurrentIndex(i)

        # Подключение сигналов
        QtCore.QObject.connect(self.btnStop, QtCore.SIGNAL("clicked()"), lambda: self.stop())
        self.buttonGroup.buttonClicked[int].connect(self.check_selection_mode)
        self.btnRun.clicked.connect(self.run_convert_shapes)
        self.dataLayer.currentIndexChanged.connect(self.check_selected_params)
        self.btnAddDataLayer.clicked.connect(self.add_data_layer)  # Подключаем к кнопке функцию

        self.buttonGroupLabel.buttonClicked[int].connect(self.change_selected_params_label)

        self.exec()

    def change_selected_params_label(self, button_id):
        if button_id == 0:
            self.isLabelFromShape = False
        elif button_id == 1:
            self.isLabelFromShape = True

    def add_data_layer(self):
        """Добавление нового QComboBox (dataLayer) под основным и перед кнопкой Add Data Layer"""
        new_data_layer = QtWidgets.QComboBox()  # Создаем новый QComboBox
        # Заполняем его элементами так же, как первый
        for key, label, enabled in self.layers:
            new_data_layer.addItem(label)

        # Добавляем в список созданных dataLayer
        self.dataLayerList.append(new_data_layer)

        # Добавляем новый QComboBox под последним существующим dataLayer
        self.row_pointer += 1
        self.layoutData.addWidget(new_data_layer, self.row_pointer, 1, 1, 2)

    def show_results_dialog(self, txt=""):
        message = "Finished in {:.2f} sec:\n {}".format(self.results_time_total, txt)

        print(message)
        Metashape.app.messageBox(message)

    def stop(self):
        self.stopped = True

    def check_stopped(self):
        if self.stopped:
            raise InterruptedError("Stop was pressed")

    def check_selection_mode(self, button_id):
        if button_id == 0:
            self.selected_mode = "points_to_boxes"
            self.sizeSpinBox.setVisible(True)
            self.labelSizeBox.setVisible(True)
        elif button_id == 1:
            self.selected_mode = "boxes_to_points"
            self.sizeSpinBox.setVisible(False)
            self.labelSizeBox.setVisible(False)
        self.check_selected_params()

    def check_selected_params(self):
        self.btnRun.setEnabled(False)
        if self.selected_mode is None:
            return
        elif self.dataLayer.currentIndex() <= 0:
            return

        self.btnRun.setEnabled(True)

    @staticmethod
    def getShapeVertices(shape):
        chunk = Metashape.app.document.chunk
        if chunk is None:
            raise Exception("Null chunk")

        T = chunk.transform.matrix
        result = []

        if shape.is_attached:
            assert (len(shape.geometry.coordinates) == 1)
            for key in shape.geometry.coordinates[0]:
                for marker in chunk.markers:
                    if marker.key == key:
                        if not marker.position:
                            raise Exception("Invalid shape vertex")

                        point = T.mulp(marker.position)
                        point = Metashape.CoordinateSystem.transform(point, chunk.world_crs, chunk.shapes.crs)
                        result.append(point)
        else:
            assert len(shape.geometry.coordinates) == 1
            for coord in shape.geometry.coordinates[0]:
                result.append(coord)

        return result

    @staticmethod
    def is_point_in_polygon(point, polygon) -> bool:
        """
        Проверяет, находится ли точка внутри многоугольника.

        :param point: Точка для проверки.
        :param polygon: Список точек (вершин), описывающих многоугольник.
        :return: True, если точка внутри, False в противном случае.
        """
        x, y = point.x, point.y
        n = len(polygon)
        inside = False

        p1x, p1y = polygon[0].x, polygon[0].y
        for i in range(n + 1):
            p2x, p2y = polygon[i % n].x, polygon[i % n].y
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    @staticmethod
    def create_boxes(points, size: float):
        """
        Создает боксы размером `size` вокруг каждой точки.

        :param points: Список точек (Metashape.Vector), для которых нужно создать боксы.
        :param size: Размер каждой стороны бокса в метрах.
        :return: Список боксов, где каждый бокс представлен списком из 4 вершин.
        """
        boxes = []
        half_size = size / 2  # Размер половины стороны бокса

        for data_point in points:
            point = data_point["point"]
            label = data_point["label"]

            x, y = point.x, point.y  # Центр точки

            # Корректируем размер долготы в градусах
            lat_in_meters = 111320  # Постоянное значение для широты: 1° ≈ 111.32 км
            lon_in_meters = 111320 * math.cos(math.radians(y))  # Долгота уменьшается на cos(широты)

            # Перевод размеров в градусы
            half_size_lat = half_size / lat_in_meters
            half_size_lon = half_size / lon_in_meters

            # Верхний левый, верхний правый, нижний правый, нижний левый
            box = [
                Metashape.Vector([x - half_size_lon, y - half_size_lat]),  # Верхний левый угол
                Metashape.Vector([x - half_size_lon, y + half_size_lat]),  # Верхний правый угол
                Metashape.Vector([x + half_size_lon, y + half_size_lat]),  # Нижний правый угол
                Metashape.Vector([x + half_size_lon, y - half_size_lat]),  # Нижний левый угол
            ]

            boxes.append({"box":box, "label":label})

        return boxes

    def run_convert_shapes(self):
        try:
            time_start = time.time()
            self.load_params()

            print("Script started...")

            if self.chunk.shapes is None:
                self.chunk.shapes = Metashape.Shapes()
                self.chunk.shapes.crs = self.chunk.crs

            self.btnRun.setEnabled(False)

            if self.selected_mode and self.selected_mode == "points_to_boxes":
                self.create_boxes_from_points()
            elif self.selected_mode and self.selected_mode == "boxes_to_points":
                self.create_points_from_boxes()


            self.results_time_total = time.time() - time_start
            self.show_results_dialog()
        except Exception as ex:
            if self.stopped:
                Metashape.app.messageBox("Processing was stopped.")
            else:
                Metashape.app.messageBox("Something gone wrong.\n"
                                         "Please check the console.")

                raise
        finally:
            self.btnRun.setEnabled(False)
            self.btnStop.setEnabled(False)
            self.stopped = False
            self.reject()

        print("Script finished.")
        return True

    def create_boxes_from_points(self):

        app = QtWidgets.QApplication.instance()
        print("Create boxes from points...")

        if not self.data_shapes:
            raise "No shapes in the data"

        all_points = []

        self.mainPBar.setValue(0)
        for i, data_shape in enumerate(self.data_shapes):
            if self.isLabelFromShape:
                label = data_shape.label
            else:
                label = data_shape.group.label

            p = self.getShapeVertices(data_shape)
            p = Metashape.CoordinateSystem.transform(p, self.chunk.shapes.crs, self.chunk.orthomosaic.crs)

            if self.data_zones:
                for zone_i, shape in enumerate(self.data_zones):
                    zone_shape = self.getShapeVertices(shape)
                    zone_points = []
                    for zp in zone_shape:
                        zp = Metashape.CoordinateSystem.transform(zp, self.chunk.shapes.crs, self.chunk.orthomosaic.crs)
                        zone_points.append(zp)

                    if self.is_point_in_polygon(p, zone_points):
                        all_points.append({"point":p, "label":label})
            else:
                all_points.append({"point":p, "label":label})

            self.mainPBar.setValue(int((i + 1) * 100 / len(self.data_shapes)))

        print("Found {} points".format(len(all_points)))

        boxes = self.create_boxes(all_points, self.sizeBox)
        print(f"Add {len(boxes)} boxes to shapes layer...")
        self.mainPBar.setValue(0)

        boxes_shapes_layer = self.chunk.shapes.addGroup()
        boxes_shapes_layer.label = f"data boxes (size: {self.sizeBox}/m)"
        boxes_shapes_layer.show_labels = False

        for ib, data_box in enumerate(boxes):
            box = data_box["box"]
            label = data_box["label"]
            corners = []
            for bp in box:
                p = Metashape.CoordinateSystem.transform(bp, self.chunk.orthomosaic.crs, self.chunk.shapes.crs)
                corners.append([p.x, p.y])

            shape = self.chunk.shapes.addShape()
            shape.group = boxes_shapes_layer
            shape.geometry = Metashape.Geometry.Polygon(corners)
            shape.label = label

            self.mainPBar.setValue(int((ib + 1) * 100 / len(boxes)))

        Metashape.app.update()
        app.processEvents()

    def create_points_from_boxes(self):
        app = QtWidgets.QApplication.instance()
        print("Create points from boxes...")

        if not self.data_shapes:
            raise ValueError("No shapes in the data")

        all_points = []
        self.mainPBar.setValue(0)

        # Обрабатываем каждую коробку
        for i, data_shape in enumerate(self.data_shapes):
            # Получение углов коробки
            box_vertices = self.getShapeVertices(data_shape)

            if self.isLabelFromShape:
                label = data_shape.label
            else:
                label = data_shape.group.label

            box_vertices = box_vertices[:-1]

            # Расчет центра коробки
            center_x = sum(vertex.x for vertex in box_vertices) / len(box_vertices)
            center_y = sum(vertex.y for vertex in box_vertices) / len(box_vertices)
            center = Metashape.Vector([center_x, center_y])

            # Трансформация центра
            #center = Metashape.CoordinateSystem.transform(center, self.chunk.shapes.crs, self.chunk.orthomosaic.crs)

            # Проверяем, принадлежит ли центр зоне
            if self.data_zones:
                for zone_i, shape in enumerate(self.data_zones):
                    zone_shape = self.getShapeVertices(shape)
                    zone_points = []
                    for zp in zone_shape:
                        zp = Metashape.CoordinateSystem.transform(zp, self.chunk.shapes.crs, self.chunk.orthomosaic.crs)
                        zone_points.append(zp)

                    if self.is_point_in_polygon(center, zone_points):
                        all_points.append({"point":center, "label":label})
                        break  # Если точка принадлежит зоне, больше не проверяем для других зон
            else:
                all_points.append({"point":center, "label":label})

            # Обновляем ProgressBar
            self.mainPBar.setValue(int((i + 1) * 100 / len(self.data_shapes)))

        print("Found {} points".format(len(all_points)))

        self.mainPBar.setValue(0)

        # Добавляем точки в слой форм
        points_shapes_layer = self.chunk.shapes.addGroup()
        points_shapes_layer.label = f"data points"
        points_shapes_layer.show_labels = True

        # Создаем точки
        for ip, data_point in enumerate(all_points):
            point = data_point["point"]
            label = data_point["label"]
            # Конвертация обратно (при необходимости)
            # transformed_point = Metashape.CoordinateSystem.transform(point, self.chunk.orthomosaic.crs,
            #                                                          self.chunk.shapes.crs)
            shape = self.chunk.shapes.addShape()
            shape.group = points_shapes_layer
            shape.geometry = Metashape.Geometry.Point([point.x, point.y])
            shape.label = label

            self.mainPBar.setValue(int((ip + 1) * 100 / len(all_points)))

        Metashape.app.update()
        app.processEvents()

    def load_params(self):

        app = QtWidgets.QApplication.instance()
        zonesLayer = self.layers[self.zonesLayer.currentIndex()]

        print("Loading shapes...")
        loading_train_shapes_start = time.time()

        Metashape.app.update()
        app.processEvents()

        shapes = self.chunk.shapes
        self.data_zones = []
        self.data_shapes = []

        self.sizeBox = self.sizeSpinBox.value()

        print(f"All shapes in chunk: {len(shapes)}")

        # Получаем ключи слоев для тренировки
        train_zones_key = zonesLayer[0]

        train_data_keys = [self.layers[item.currentIndex()][0] for item in self.dataLayerList]

        for i, shape in enumerate(shapes):
            if shape.group.key == train_zones_key:
                self.train_zones.append(shape)
            else:
                for train_data_key in train_data_keys:
                    if shape.group.key == train_data_key:
                        self.data_shapes.append(shape)
                        break

            self.mainPBar.setValue(int((100 * i + 1) / len(shapes)))
            i += 1
            app.processEvents()
            self.check_stopped()


        print(f"{len(self.data_zones)} zones and {len(self.data_shapes)} data loaded in {time.time() - loading_train_shapes_start :.2f} sec")

class NoAliasDumper(yaml.Dumper):
    """
    A YAML dumper class to disable the generation of YAML aliases.

    This class extends `yaml.Dumper` to override its behavior and
    ensure that no aliases are generated while dumping YAML data.
    This is especially useful when dealing with large data
    structures where aliases can compromise clarity and
    maintainability.

    Attributes
    ----------
    None
    """
    def ignore_aliases(self, data):
        return True

class YOLODatasetConverter:
    def __init__(self, all_annotations, input_dir_images=None, output_dir=None, split_ratios=None, empty_ratio=None,
                 allow_empty_annotations=True, callback=None, mode_converter=None):
        """
        Универсальный класс для преобразования COCO-аннотаций в формат YOLO с разделением на train и val.

        :param all_annotations: словарь аннотаций COCO-формата со структурой {images, annotations, categories}.
        :param input_dir_images: входная директория, где содержатся изображения.
        :param output_dir: каталог для сохранения преобразованного датасета.
        :param split_ratios: словарь с долями разделения на train и val, например {"train": 0.8, "val": 0.2}.
        :param empty_ratio: пропорции разбиения пустых изображений между train и val, например {"train": 0.8, "val": 0.2}.
        :param allow_empty_annotations: включать ли изображения без аннотаций.
        """
        self.all_annotations = all_annotations
        self.input_dir_images = input_dir_images
        self.output_dir = output_dir
        self.split_ratios = split_ratios if split_ratios else {"train": 0.8, "val": 0.2}
        self.empty_ratio = empty_ratio if empty_ratio else {"train": 0.8, "val": 0.2}
        self.allow_empty_annotations = allow_empty_annotations

        # Каталоги для train и val
        self.train_dir = os.path.join(self.output_dir, "train")
        self.val_dir = os.path.join(self.output_dir, "valid")

        self.callback = callback
        self.mode_converter = mode_converter

    def _log(self, message=None):
        """Логирование через callback или print."""
        if self.callback:
            self.callback(message)
        else:
            print(message)

    def prepare_directories(self):
        """Создает необходимые папки для сохранения."""
        os.makedirs(os.path.join(self.train_dir, "labels"), exist_ok=True)
        os.makedirs(os.path.join(self.val_dir, "labels"), exist_ok=True)
        os.makedirs(os.path.join(self.train_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.val_dir, "images"), exist_ok=True)

    def visualize_annotations_for_image(self, image, annotations, target_dir):
        """
        Отображает все аннотации для одного конкретного изображения и сохраняет его с аннотациями.

        :param image: Словарь с информацией об изображении (из COCO-формата).
        :param annotations: Список всех аннотаций (bbox, category_id и т.д.).
        :param target_dir: Папка, куда будет сохранено изображение с аннотациями.
        """
        # Создаем папку, если ее нет
        os.makedirs(target_dir, exist_ok=True)

        # Пути к оригинальному изображению и сохранению результатов
        img_path = os.path.join(self.input_dir_images, image['file_name'])
        output_path = os.path.join(target_dir, image['file_name'])

        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}. Skipping.")
            return

        # Загружаем изображение
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not load image: {img_path}. Skipping.")
            return

        # Фильтруем аннотации для данного изображения
        image_id = image['id']
        image_annotations = [ann for ann in annotations if ann['image_id'] == image_id]

        # Отрисовываем каждую аннотацию
        for annotation in image_annotations:
            bbox = annotation['bbox']  # [xmin, ymin, width, height]
            category_id = annotation['category_id']
            segmentation = annotation['segmentation']

            # Преобразуем bbox из COCO в формат OpenCV
            if bbox:
                x_min, y_min, width, height = bbox
                x_max, y_max = int(x_min + width), int(y_min + height)
                x_min, y_min = int(x_min), int(y_min)

                # Рисуем прямоугольник вокруг объекта
                color = (0, 255, 0)  # Зеленый цвет для рамок
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)

                # Добавляем текст с категорией
                cv2.putText(
                    img,
                    f"Class: {category_id}",
                    (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1
                )

            # Если доступна сегментация, рисуем контур
            if self.mode_converter == "segmentation" and segmentation:
                # Загрузка координат сегментации
                segmentation_points = segmentation  # Здесь уже предполагается список [x1, y1, x2, y2, ...]
                try:
                    # Проверяем, что segmentation_points - это список и корректен
                    if isinstance(segmentation_points, list) and len(segmentation_points) % 2 == 0 and len(
                            segmentation_points) > 2:

                            # Преобразуем список в массив numpy и формируем точки
                            points = np.array(segmentation_points, dtype=np.int32).reshape((-1, 2))

                            # Цвет для контура сегментации
                            contour_color = (255, 0, 0)  # Синий цвет для контура

                            # Рисуем замкнутый контур
                            cv2.polylines(img, [points], isClosed=True, color=contour_color, thickness=2)

                    else:
                        print(f"Некорректные данные сегментации: {segmentation_points}")
                except Exception as e:
                    print(f"Ошибка при обработке сегментации: {e} >>>> {segmentation}")

        # Сохраняем изображение с аннотациями
        cv2.imwrite(output_path, img)
        # print(f"Saved annotated image: {output_path}")

    @staticmethod
    def write_yolo_annotation(output_path, yolo_lines):
        """Записывает YOLO-аннотацию в файл."""
        with open(output_path, "a") as f:
            for line in yolo_lines:
                f.write(line + "\n")

    def split_dataset(self):
        """Разделяет датасет на train и val."""
        # Словарь для быстрого доступа к изображениям
        image_dict = {image['id']: image for image in self.all_annotations['images']}

        # Разделяем изображения с аннотациями и без
        annotated_images = set(annotation['image_id'] for annotation in self.all_annotations['annotations'])
        images_with_annotations = [image for image in self.all_annotations['images'] if image['id'] in annotated_images]
        images_without_annotations = [image for image in self.all_annotations['images'] if
                                      image['id'] not in annotated_images]

        print(f"Images with annotations: {len(images_with_annotations)}")
        print(f"Images without annotations: {len(images_without_annotations)}")
        self._log()

        # Разделяем изображения с аннотациями
        train_with_annotations, val_with_annotations = train_test_split(
            images_with_annotations,
            test_size=self.split_ratios["val"],
            random_state=42
        )

        # Если разрешены пустые аннотации, проверяем их и добавляем ограниченный процент
        if self.allow_empty_annotations and images_without_annotations:
            # Рассчитываем допустимое количество пустых изображений для train и val
            max_train_empty = int(len(train_with_annotations) * self.empty_ratio["train"])
            max_val_empty = int(len(val_with_annotations) * self.empty_ratio["train"])

            # # Ограничиваем количество пустых изображений
            # train_empty_candidates, val_empty_candidates = train_test_split(
            #     images_without_annotations,
            #     test_size=self.empty_ratio["val"],
            #     random_state=42
            # )
            random.shuffle(images_without_annotations)
            # random.shuffle(val_empty_candidates)

            # train_empty = train_empty_candidates[:max_train_empty]  # Ограничиваем train пустые
            # val_empty = val_empty_candidates[:max_val_empty]  # Ограничиваем val пустые
            # Проверка для train_empty
            train_empty = images_without_annotations[:min(max_train_empty, len(images_without_annotations))]

            # Проверка для val_empty
            # val_empty_start = min(max_train_empty, len(images_without_annotations))
            # val_empty_end = min(max_train_empty + max_val_empty, len(images_without_annotations))
            # val_empty = images_without_annotations[val_empty_start:val_empty_end]

        else:
            train_empty, val_empty = [], []

        # Итоговые наборы train и val
        self.train_images = train_with_annotations + train_empty
        self.val_images = val_with_annotations # + val_empty

        # Аннотации для train/val
        self.train_annotations = []
        self.val_annotations = []

        for annotation in self.all_annotations['annotations']:
            image_id = annotation['image_id']
            image_info = image_dict[image_id]

            if image_info in self.train_images:
                self.train_annotations.append(annotation)
            elif image_info in self.val_images:
                self.val_annotations.append(annotation)

    def move_images(self, images, target_dir):
        """Копирует изображения из input_dir_images в указанный каталог."""
        for image in images:
            source_path = os.path.join(self.input_dir_images, image['file_name'])
            target_path = os.path.join(target_dir, "images", image['file_name'])
            if os.path.exists(source_path):
                shutil.copy(source_path, target_path)
            else:
                print(f"Файл не найден. Пропускаю: {source_path}")
            self._log()

    def convert_to_yolo_format(self, annotations, target_dir, images):
        """Конвертирует аннотации в формат YOLO."""
        image_dict = {image['id']: image for image in images}

        for annotation in annotations:
            # Получаем информацию об изображении
            image_id = annotation['image_id']
            category_id = annotation['category_id']
            image_info = image_dict[image_id]
            img_width = image_info['width']
            img_height = image_info['height']

            yolo_line = ""
            # Преобразуем bbox в формат YOLO

            if self.mode_converter == "detect":
                if annotation['bbox']:
                    xmin, ymin, width, height = annotation['bbox']
                    x_center = (xmin + width / 2) / img_width
                    y_center = (ymin + height / 2) / img_height
                    width /= img_width
                    height /= img_height

                    # Формируем строку в формате YOLO
                    yolo_line = f"{category_id} {x_center:.10f} {y_center:.10f} {width:.10f} {height:.10f}"
            elif self.mode_converter == "segmentation":
                if annotation['segmentation']:
                    segmentation_points = annotation['segmentation']  # Предполагается формат [x1, y1, x2, y2, ...]

                    if len(segmentation_points) % 2 != 0 or len(segmentation_points) == 0:
                        print(f"Ошибка сегментации в аннотации: {annotation}")
                        continue  # Пропускаем эту аннотацию

                    # Выполняем нормализацию и запись
                    normalized_points = [
                        f"{x / img_width:.10f} {y / img_height:.10f}" for x, y in
                        zip(segmentation_points[::2], segmentation_points[1::2])
                    ]
                    points_str = " ".join(normalized_points)
                    yolo_line = f"{category_id} {points_str}"

            # Сохраняем аннотацию в файл
            output_file = os.path.join(target_dir, "labels", os.path.splitext(image_info['file_name'])[0] + ".txt")
            if os.path.isfile(output_file) and yolo_line == "":
                continue
            self.write_yolo_annotation(output_file, [yolo_line])

            self._log()

        # Для изображений без аннотаций создаем пустые файлы
        annotation_image_id = [annotation['image_id'] for annotation in annotations]
        for image in images:
            if image['id'] not in annotation_image_id:
                empty_file = os.path.join(target_dir, "labels", os.path.splitext(image['file_name'])[0] + ".txt")
                open(empty_file, "w").close()

    def create_data_yaml(self):
        """
        Создаёт файл data.yaml для YOLOv5/YOLOv8 с упорядочением ключей.
        """
        # Получаем количество классов и их имена
        categories = self.all_annotations['categories']
        names = [name for name, index in sorted(categories.items(), key=lambda item: int(item[1]))]
        nc = len(categories)

        # Формируем данные для YAML с установленным порядком ключей
        data = OrderedDict([
            ("train", "../train/images"),
            ("val", "../valid/images"),
            ("nc", nc),
            ("names", names)
        ])

        # Путь для сохранения YAML
        yaml_path = os.path.join(self.output_dir, "data.yaml")

        # Сохраняем YAML в правильном формате с исправлением !!python/object/apply
        with open(yaml_path, "w") as yaml_file:
            yaml.dump(dict(data), yaml_file, Dumper=NoAliasDumper, default_flow_style=None, sort_keys=False)
            print(f"The data file.yaml has been created successfully: {yaml_path}")

    @staticmethod
    def analyze_txt_files(directory):
        # Счётчики
        total_files = 0
        empty_files = 0
        non_empty_files = 0
        total_sum = 0

        # Перебираем файлы в указанной директории
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.txt'):  # Фильтруем только .txt файлы
                    total_files += 1
                    file_path = os.path.join(root, file)

                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            lines = f.readlines()

                            if not lines:  # Проверяем, пустой ли файл (по строкам)
                                empty_files += 1
                            else:
                                non_empty_files += 1
                                # Пытаемся извлечь числовые значения и суммировать

                                try:
                                    total_sum += len(lines)
                                except ValueError:
                                    continue
                    except Exception as e:
                        print(f"Ошибка при обработке файла {file_path}: {e}")

        # Вычисляем проценты
        empty_percentage = (empty_files / total_files * 100) if total_files > 0 else 0
        non_empty_percentage = (non_empty_files / total_files * 100) if total_files > 0 else 0

        # Выводим результаты
        print(f"Total labels: {total_files}")
        print(f"Empty labels: {empty_files} ({empty_percentage:.2f}%)")
        print(f"Total labels with data: {non_empty_files} ({non_empty_percentage:.2f}%)")
        if non_empty_files > 0:
            print(f"Sum all rows labels: {total_sum}")

    def convert(self):
        self._log("Mode converter: {}".format(self.mode_converter))
        """Основной метод для преобразования датасета."""
        self._log("Preparing directories...")
        self.prepare_directories()

        self._log("Splitting dataset...")
        self.split_dataset()

        self._log("Moving training images...")
        self.move_images(self.train_images, self.train_dir)

        self._log("Moving validation images...")
        self.move_images(self.val_images, self.val_dir)

        self._log("Converting training dataset to YOLO format...")
        self.convert_to_yolo_format(self.train_annotations, self.train_dir, self.train_images)

        self._log("Converting validation dataset to YOLO format...")
        self.convert_to_yolo_format(self.val_annotations, self.val_dir, self.val_images)

        self._log("Creating data.yaml file...")
        self.create_data_yaml()

        self._log(f"The dataset has been successfully split and converted. The result is saved in: {self.output_dir}")
        self._log(f"Train images: {len(self.train_images)} | Val images: {len(self.val_images)}")

class WindowCreateYoloDataset(QtWidgets.QDialog):

    def __init__(self, parent):

        QtWidgets.QDialog.__init__(self, parent)
        self.shapes = []
        self.setWindowTitle("Create Yolo dataset on orthomosaic")

        self.patch_size = None
        self.orthomosaic_resolution = None
        self.patch_inner_border = None
        self.train_on_user_data_enabled = None
        self.train_data = None
        self.train_zones = None
        self.force_small_patch_size = True
        self.max_image_size = None
        self.train_percentage = 0.8
        self.percent_empty_limit = 0.4
        self.isLoadTiles = False
        self.augment_colors = False
        self.cleanup_working_dir = False
        self.isDebugMode = False
        self.selected_mode = "detect"

        self.preferred_patch_size = 640
        self.preferred_resolution = 0.005
        self.isAugmentData = False

        self.prefer_original_resolution = False

        self.expected_layer_name_train_zones = "Zone"
        self.expected_layer_name_train_data = "Data"

        if len(Metashape.app.document.path) > 0:
            self.working_dir = str(pathlib.Path(Metashape.app.document.path).parent / "dataset_yolo")
        else:
            self.working_dir = ""

        self.chunk = Metashape.app.document.chunk
        self.create_gui()

        if os.path.exists(os.path.join(self.working_dir,"tiles")):
            self.isLoadTiles = True
            self.loadTilesCbox.setChecked(True)

        max_image_size = Metashape.app.settings.value("scripts/create_yolo_dataset/max_image_size")
        self.max_image_size = int(max_image_size) if max_image_size else 640
        self.maxSizeImageSpinBox.setValue(self.max_image_size)

        percent_empty_limit = Metashape.app.settings.value("scripts/create_yolo_dataset/percent_empty_limit")
        self.percent_empty_limit = float(percent_empty_limit) if percent_empty_limit else 0.4
        self.proportionBackgroundSpinBox.setValue(self.percent_empty_limit)

        train_percentage = Metashape.app.settings.value("scripts/create_yolo_dataset/train_percentage")
        self.train_percentage = float(train_percentage) if train_percentage else 0.8
        self.separationDataSpinBox.setValue(self.train_percentage)

        self.exec()

    def stop(self):
        self.stopped = True

    def check_stopped(self):
        if self.stopped:
            raise InterruptedError("Stop was pressed")

    def run_process(self):
        try:
            self.stopped = False
            self.btnRun.setEnabled(False)
            self.btnStop.setEnabled(True)

            time_start = time.time()

            self.load_params()

            self.prepare()

            print("Script started...")

            self.export_orthomosaic()

            if self.chunk.shapes is None:
                self.chunk.shapes = Metashape.Shapes()
                self.chunk.shapes.crs = self.chunk.crs

            if self.train_on_user_data_enabled:
                self.create_on_user_data()

            self.results_time_total = time.time() - time_start
            self.show_results_dialog()
        except:
            if self.stopped:
                Metashape.app.messageBox("Processing was stopped.")
            else:
                Metashape.app.messageBox("Something gone wrong.\n"
                                         "Please check the console.")
                raise
        finally:
            remove_directory(self.dir_inner_data)
            self.btnStop.setEnabled(False)
            self.btnRun.setEnabled(True)
            self.reject()

        print("Script finished.")
        return True

    def prepare(self):
        if self.working_dir == "":
            raise Exception("You should specify working directory (or save .psx project)")

        os.makedirs(self.working_dir, exist_ok=True)
        print("Working dir: {}".format(self.working_dir))

        self.cleanup_working_dir = False

        self.dir_tiles = self.working_dir + "/tiles/"
        self.dir_data = ensure_unique_directory(self.working_dir + f"/datasets_{self.selected_mode}")
        self.dir_inner_data = os.path.join(self.dir_data, "inner")
        self.dir_inner_images = os.path.join(self.dir_inner_data, "images")

        self.dir_train_subtiles_debug_dataset = os.path.join(self.dir_data, "debug_dataset")

        create_dirs = [self.dir_tiles, self.dir_data, self.dir_inner_images,]
        if self.isDebugMode:
            create_dirs.append(self.dir_train_subtiles_debug_dataset)

        for subdir in create_dirs:
            os.makedirs(subdir, exist_ok=True)

    def export_orthomosaic(self):
        import numpy as np

        print("Preparing orthomosaic...")

        kwargs = {}
        if not self.prefer_original_resolution:# and (self.chunk.orthomosaic.resolution < self.preferred_resolution * 0.90):
            kwargs["resolution"] = self.preferred_resolution
        else:
            print("no resolution downscaling required")

        if not self.isLoadTiles:
            self.chunk.exportRaster(path=self.dir_tiles + "tile.jpg",
                                    source_data=Metashape.OrthomosaicData,
                                    image_format=Metashape.ImageFormat.ImageFormatJPEG,
                                    save_alpha=False,
                                    white_background=True,
                                    save_world=True,
                                    split_in_blocks=True,
                                    block_width=self.patch_size,
                                    block_height=self.patch_size,
                                    **kwargs)

        tiles = os.listdir(self.dir_tiles)
        if not tiles:
            raise Exception("No tiles found in the directory.")

        self.tiles_paths = {}
        self.tiles_to_world = {}

        for tile in sorted(tiles):
            assert tile.startswith("tile-")

            _, tile_x, tile_y = tile.split(".")[0].split("-")
            tile_x, tile_y = map(int, [tile_x, tile_y])
            if tile.endswith(".jgw") or tile.endswith(".pgw"):  # https://en.wikipedia.org/wiki/World_file
                with open(self.dir_tiles + tile, "r") as file:
                    matrix2x3 = list(map(float, file.readlines()))
                matrix2x3 = np.array(matrix2x3).reshape(3, 2).T
                self.tiles_to_world[tile_x, tile_y] = matrix2x3
            elif tile.endswith(".jpg"):
                self.tiles_paths[tile_x, tile_y] = self.dir_tiles + tile

        assert (len(self.tiles_paths) == len(self.tiles_to_world))
        assert (self.tiles_paths.keys() == self.tiles_to_world.keys())

        self.tile_min_x = min([key[0] for key in self.tiles_paths.keys()])
        self.tile_max_x = max([key[0] for key in self.tiles_paths.keys()])
        self.tile_min_y = min([key[1] for key in self.tiles_paths.keys()])
        self.tile_max_y = max([key[1] for key in self.tiles_paths.keys()])
        print("{} tiles, tile_x in [{}; {}], tile_y in [{}; {}]".format(len(self.tiles_paths), self.tile_min_x,
                                                                        self.tile_max_x, self.tile_min_y,
                                                                        self.tile_max_y))

    def collate_fn(self, batch):
        return tuple(zip(*batch))

    def resize_tile(self, image, max_size=1024):
        import numpy as np
        import cv2

        # Измерение исходных размеров изображения
        orig_height, orig_width = image.shape[:2]

        # Определение коэффициента масштабирования
        if max(orig_height, orig_width) != max_size:
            scale = max_size / max(orig_height, orig_width)
        else:
            scale = 1.0

        new_height = int(orig_height * scale)
        new_width = int(orig_width * scale)

        # Изменение размера изображения
        # image = Image.fromarray(image.astype(np.uint8))  # Преобразование из numpy массива в PIL Image
        # image = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
        # image = np.array(image)  # Преобразование обратно в numpy массив
        # Изменение размера изображения
        image = cv2.resize(np.array(image), (new_width, new_height), interpolation=cv2.INTER_AREA)

        return image, scale

    def add_boxes_zone_tiles(self, tiles_data, shapes_group):
        import numpy as np

        for row in tiles_data:
            xmin, ymin, xmax, ymax, label, zone_to_world = (int(row["xmin"]),
                                                            int(row["ymin"]),
                                                            int(row["xmax"]),
                                                            int(row["ymax"]),
                                                            row["label"],
                                                            row["zone_to_world"])

            corners = []
            for x, y in [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]:
                x, y = zone_to_world @ np.array([x + 0.5, y + 0.5, 1]).reshape(3, 1)
                p = Metashape.Vector([x, y])
                p = Metashape.CoordinateSystem.transform(p, self.chunk.orthomosaic.crs, self.chunk.shapes.crs)
                corners.append([p.x, p.y])

            shape = self.chunk.shapes.addShape()
            shape.group = shapes_group
            shape.geometry = Metashape.Geometry.Polygon(corners)
            shape.label = label

    def custom_callback(self, message=None):
        app = QtWidgets.QApplication.instance()
        if message:
            print(message)
        Metashape.app.update()
        app.processEvents()
        self.check_stopped()

    def create_on_user_data(self):
        import sys
        import cv2
        import numpy as np

        import os
        from concurrent.futures import ThreadPoolExecutor



        random.seed(2391231231324531)

        app = QtWidgets.QApplication.instance()

        training_start = time.time()

        num_cores = os.cpu_count()
        executor = ThreadPoolExecutor(max_workers=num_cores)

        # Напечатать устройство для проверки

        print(f'Max size image: {self.max_image_size}')

        tales_boxes_data = []

        nannotations = 1

        self.train_zones_on_ortho = []

        n_train_zone_shapes_out_of_orthomosaic = 0
        for zone_i, shape in enumerate(self.train_zones):
            shape_vertices = getShapeVertices(shape)
            zone_from_world = None
            zone_from_world_best = None
            zone_to_world = None
            for tile_x in range(self.tile_min_x, self.tile_max_x + 1):
                for tile_y in range(self.tile_min_y, self.tile_max_y + 1):
                    if (tile_x, tile_y) not in self.tiles_paths:
                        continue
                    to_world = self.tiles_to_world[tile_x, tile_y]
                    from_world = self.invert_matrix_2x3(to_world)
                    for p in shape_vertices:
                        p = Metashape.CoordinateSystem.transform(p, self.chunk.shapes.crs,
                                                                 self.chunk.orthomosaic.crs)
                        p_in_tile = from_world @ [p.x, p.y, 1]
                        distance2_to_tile_center = np.linalg.norm(
                            p_in_tile - [self.patch_size / 2, self.patch_size / 2])
                        if zone_from_world_best is None or distance2_to_tile_center < zone_from_world_best:
                            zone_from_world_best = distance2_to_tile_center
                            zone_from_world = self.invert_matrix_2x3(
                                self.add_pixel_shift(to_world, -tile_x * self.patch_size,
                                                     -tile_y * self.patch_size))
                            zone_to_world = self.add_pixel_shift(to_world, -tile_x * self.patch_size,
                                                                 -tile_y * self.patch_size)
            if zone_from_world_best > 1.1 * (self.patch_size / 2) ** 2:
                n_train_zone_shapes_out_of_orthomosaic += 1

            zone_from = None
            zone_to = None
            for p in shape_vertices:
                p = Metashape.CoordinateSystem.transform(p, self.chunk.shapes.crs, self.chunk.orthomosaic.crs)
                p_in_ortho = np.int32(np.round(zone_from_world @ [p.x, p.y, 1]))
                if zone_from is None:
                    zone_from = p_in_ortho
                if zone_to is None:
                    zone_to = p_in_ortho
                zone_from = np.minimum(zone_from, p_in_ortho)
                zone_to = np.maximum(zone_to, p_in_ortho)
            train_size = zone_to - zone_from
            train_size_m = np.int32(np.round(train_size * self.orthomosaic_resolution))
            if np.any(train_size < self.patch_size):
                print("Zone #{} {}x{} pixels ({}x{} meters) is too small - each side should be at least {} meters"
                      .format(zone_i + 1, train_size[0], train_size[1], train_size_m[0], train_size_m[1],
                              self.patch_size * self.orthomosaic_resolution), file=sys.stderr)
                self.train_zones_on_ortho.append(None)
            else:
                print("Zone #{}: {}x{} orthomosaic pixels, {}x{} meters".format(zone_i + 1, train_size[0],
                                                                                train_size[1], train_size_m[0],
                                                                                train_size_m[1]))
                self.train_zones_on_ortho.append((zone_from, zone_to, zone_from_world, zone_to_world))
        assert len(self.train_zones_on_ortho) == len(self.train_zones)

        if n_train_zone_shapes_out_of_orthomosaic > 0:
            print(f"Warning, {n_train_zone_shapes_out_of_orthomosaic} train zones shapes are out of orthomosaic")

        area_threshold = 0.3

        all_annotations = {"images": [],
                           "annotations": [],
                           "categories": None
                           }

        image_id = 1

        self.train_nannotations_in_zones = 0
        id_label = 0
        classes = {}

        for zone_i, shape in enumerate(self.train_zones):
            if self.train_zones_on_ortho[zone_i] is None:
                continue

            self.txtPBar.setText(f"Create dataset (zones {zone_i + 1} of {len(self.train_zones)}):")
            Metashape.app.update()
            app.processEvents()
            self.check_stopped()

            zone_from, zone_to, zone_from_world, zone_to_world = self.train_zones_on_ortho[zone_i]
            annotations = []
            annotations_boxes = []

            for annotation in self.train_data:
                Metashape.app.update()
                app.processEvents()
                self.check_stopped()

                annotation_vertices = getShapeVertices(annotation)
                annotation_from = None
                annotation_to = None
                poly = []
                for p in annotation_vertices:
                    p = Metashape.CoordinateSystem.transform(p, self.chunk.shapes.crs, self.chunk.orthomosaic.crs)
                    p_in_ortho = np.int32(np.round(zone_from_world @ [p.x, p.y, 1]))

                    if annotation_from is None:
                        annotation_from = p_in_ortho
                    if annotation_to is None:
                        annotation_to = p_in_ortho

                    poly.append(p_in_ortho)
                    annotation_from = np.minimum(annotation_from, p_in_ortho)
                    annotation_to = np.maximum(annotation_to, p_in_ortho)

                bbox_from, bbox_to = self.intersect(zone_from, zone_to, annotation_from, annotation_to)
                if self.area(bbox_from, bbox_to) > self.area(annotation_from, annotation_to) * area_threshold:
                    _label = annotation.label
                    if _label not in classes:
                        classes[_label] = id_label
                        id_label += 1

                    try:
                        annotations_boxes.append({"box": (annotation_from, annotation_to), "category": classes[_label]})
                        annotations.append(poly)
                    except ValueError:
                        raise ValueError(f"The label annotation of {_label} was not found in the classes list")

            all_annotations['categories'] = classes
            self.train_nannotations_in_zones += len(annotations)
            print(f"Create dataset zone #{zone_i + 1}: {len(annotations)} annotations inside")

            border = self.patch_inner_border
            inner_path_size = self.patch_size - 2 * border

            zone_size = zone_to - zone_from
            assert np.all(zone_size >= self.patch_size)
            nx_tiles, ny_tiles = np.int32((zone_size - 2 * border + inner_path_size - 1) // inner_path_size)
            assert nx_tiles >= 1 and ny_tiles >= 1
            xy_step = np.int32(np.round((zone_size + [nx_tiles, ny_tiles] - 1) // [nx_tiles, ny_tiles]))

            out_of_orthomosaic_train_tile = 0
            total_steps = nx_tiles * ny_tiles

            for x_tile in range(0, nx_tiles):
                for y_tile in range(0, ny_tiles):
                    tile_to = zone_from + self.patch_size + xy_step * [x_tile, y_tile]
                    if x_tile == nx_tiles - 1 and y_tile == ny_tiles - 1:
                        assert np.all(tile_to >= zone_to)
                    tile_to = np.minimum(tile_to, zone_to)
                    tile_from = tile_to - self.patch_size
                    if x_tile == 0 and y_tile == 0:
                        assert np.all(tile_from == zone_from)
                    assert np.all(tile_from >= zone_from)

                    tile = self.read_part(tile_from, tile_to)
                    label_tile = f"{(zone_i + 1)}-{x_tile}-{y_tile}"
                    tales_boxes_data.append(
                        {"xmin": tile_from[0], "ymin": tile_from[1], "xmax": tile_to[0], "ymax": tile_to[1],
                         "label": label_tile, "zone_to_world": zone_to_world})
                    assert tile.shape == (self.patch_size, self.patch_size, 3)

                    if np.all(tile == 255):
                        out_of_orthomosaic_train_tile += 1
                        continue

                    tile_annotations_boxes = []
                    tile_annotations = []

                    for idx, ann in enumerate(annotations_boxes):
                        annotation_from, annotation_to = ann["box"]
                        category = ann["category"]
                        bbox_from, bbox_to = self.intersect(tile_from, tile_to, annotation_from, annotation_to)
                        if self.area(bbox_from, bbox_to) > self.area(annotation_from,
                                                                     annotation_to) * area_threshold:
                            tile_annotations_boxes.append(
                                {"box": (bbox_from - tile_from, bbox_to - tile_from), "category": category})
                            tile_annotations.append(np.array(annotations[idx]) - np.array(tile_from))


                    transformations = [
                        (False, 0),  # Оригинальное изображение
                        (False, 1),  # Повернуть на 90 градусов
                        (False, 2),  # Повернуть на 180 градусов
                        (False, 3),  # Повернуть на 270 градусов
                        (True, 0),  # Зеркально отразить
                        (True, 1),  # Зеркально отразить и повернуть на 90
                        (True, 2),  # Зеркально отразить и повернуть на 180
                        (True, 3)  # Зеркально отразить и повернуть на 270
                    ]
                    if self.isAugmentData:
                        transformations = transformations[0:8]
                    else:
                        transformations = transformations[0:1]

                    version_i = 0
                    for is_mirrored, n90rotation in transformations:
                        tile_version = tile.copy()
                        tile_annotations_version = [item["box"] for item in tile_annotations_boxes]
                        tile_annotations_mask_version = tile_annotations.copy()

                        if is_mirrored:
                            if tile_annotations:
                                tile_annotations_version, tile_annotations_mask_version = self.flip_annotations(
                                    tile_annotations_version, tile_annotations_mask_version, tile_version)
                            tile_version = cv2.flip(tile_version, 1)

                        for _ in range(n90rotation):
                            if tile_annotations:
                                tile_annotations_version, tile_annotations_mask_version = self.rotate90clockwise_annotations(
                                    tile_annotations_version, tile_annotations_mask_version, tile_version
                                )
                            tile_version = cv2.rotate(tile_version, cv2.ROTATE_90_CLOCKWISE)

                        tile_version = self.random_augmentation(tile_version)

                        h, w, cn = tile_version.shape

                        tile_name = f"{(zone_i + 1)}-{x_tile}-{y_tile}-{version_i}_{image_id}.jpg"

                        row_img = {"id": image_id, "file_name": tile_name,
                                   "height": h, "width": w}
                        all_annotations["images"].append(row_img)

                        boxes = []
                        masks = []


                        if tile_annotations_version:
                            for idx_an, (_min, _max) in enumerate(tile_annotations_version):

                                contour = [[point[0], point[1]] for point in tile_annotations_mask_version[idx_an]]
                                contour = self.correct_contour(contour, w, h)
                                xmin, ymin, xmax, ymax = self.get_bounding_box(contour)

                                coords_array = np.array(contour)
                                flat_coords = coords_array.flatten().tolist()

                                boxes.append([xmin, ymin, xmax, ymax])
                                masks.append(flat_coords)

                                row_annatation = {"image_id": image_id,
                                                  "id": nannotations,
                                                  "category_id": tile_annotations_boxes[idx_an]["category"],
                                                  "bbox": [xmin, ymin, xmax-xmin, ymax-ymin],
                                                  "segmentation": flat_coords if flat_coords else None,
                                                  "area": (xmax - xmin) * (ymax - ymin),
                                                  "iscrowd": 0}
                                all_annotations["annotations"].append(row_annatation)
                                nannotations += 1


                        else:
                            # row_annatation = {"image_id": image_id,
                            #                   "id": nannotations,
                            #                   "category_id": 0,
                            #                   "bbox": [],
                            #                   "segmentation": [],
                            #                   "area": 0,
                            #                   "iscrowd": 0}
                            # all_annotations["annotations"].append(row_annatation)
                            # nannotations += 1
                            pass

                        image_id += 1
                        executor.submit(cv2.imwrite, os.path.join(self.dir_inner_images, tile_name), tile_version)

                        self.check_stopped()
                        if self.isDebugMode:
                            tile_with_entity = self.debug_draw_objects(tile_version,
                                                                    boxes,
                                                                    masks)
                            os.makedirs(os.path.join(self.dir_train_subtiles_debug_dataset,"inner"), exist_ok=True)
                            executor.submit(cv2.imwrite, os.path.join(self.dir_train_subtiles_debug_dataset,"inner",tile_name),
                                                 tile_with_entity)

                    current_step = x_tile * ny_tiles + y_tile + 1
                    progress = (current_step * 100) / total_steps
                    self.progressBar.setValue(progress)

                    app.processEvents()
                    self.check_stopped()

            if out_of_orthomosaic_train_tile == nx_tiles * ny_tiles:
                raise RuntimeError(
                    f"It seems that zone #{zone_i + 1} has no orthomosaic data, please check zones, orthomosaic and its Outer Boundary.")
            else:
                if out_of_orthomosaic_train_tile > 0:
                    print(
                        f"{out_of_orthomosaic_train_tile}/{nx_tiles * ny_tiles} of tiles in zone #{zone_i + 1} has no orthomosaic data")

        executor.shutdown(wait=True)

        random.shuffle(all_annotations["images"])
        random.shuffle(all_annotations["annotations"])


        print(f"Total images: {len(all_annotations['images'])}")
        print(f"Total annotations: {len(all_annotations['annotations'])}")
        print(f"Classes: {all_annotations['categories']}")

        if self.isDebugMode:
            detected_shapes_layer = self.chunk.shapes.addGroup()
            detected_shapes_layer.label = "Tiles Boxes"
            detected_shapes_layer.show_labels = False
            self.add_boxes_zone_tiles(tales_boxes_data, detected_shapes_layer)

        print(">>> Create dataset...")

        # Создаем экземпляр конвертера
        val_pr = 1-self.train_percentage
        converter = YOLODatasetConverter(
            all_annotations,
            input_dir_images=self.dir_inner_images,
            output_dir=self.dir_data,
            split_ratios={"train": self.train_percentage, "val": val_pr if val_pr > 0 else None},
            empty_ratio={"train": self.percent_empty_limit, "val": 1-self.percent_empty_limit},
            allow_empty_annotations=True,
            callback=self.custom_callback,
            mode_converter=self.selected_mode,

        )

        print("Please wait...")
        Metashape.app.update()
        app.processEvents()
        self.check_stopped()

        # Конвертируем датасет
        converter.convert()

        if self.isDebugMode:
            for i, image in enumerate(converter.train_images):
                converter.visualize_annotations_for_image(
                    image=image,
                    annotations=converter.train_annotations,
                    target_dir=os.path.join(self.dir_train_subtiles_debug_dataset, "train_annotated_images")

                )

            for i, image in enumerate(converter.val_images):
                converter.visualize_annotations_for_image(
                    image=image,
                    annotations=converter.val_annotations,
                    target_dir=os.path.join(self.dir_train_subtiles_debug_dataset, "val_annotated_images")
                )

            # debug only boxes
            # for root, dirs, files in os.walk(os.path.join(converter.train_dir, "labels")):
            #     for file in files:
            #         if file.endswith('.txt'):
            #             # Visualize
            #             visualize_image_annotations(
            #                 image_path=os.path.join(pathlib.Path(root).parent, "images", pathlib.Path(file).stem + ".jpg"),
            #                 txt_path=os.path.join(root, file),
            #                 label_map={0: "nfs-pups"},
            #             )
        print("Dataset prepared:")
        print("Train:")
        converter.analyze_txt_files(os.path.join(converter.train_dir, "labels"))
        print("Val:")
        converter.analyze_txt_files(os.path.join(converter.val_dir, "labels"))

        self.results_time_training = time.time() - training_start

    @staticmethod
    def flip_annotations(bboxes, contours, img):
        h, w, _ = img.shape
        flipped_entity = []
        flipped_contours = []

        # Переворот прямоугольников
        for bbox_from, bbox_to in bboxes:
            (xmin, ymin), (xmax, ymax) = bbox_from, bbox_to
            # Зеркальное отражение по горизонтали (x меняется относительно центра изображения)
            flipped_entity.append(((w - xmax, ymin), (w - xmin, ymax)))

        # Переворот контуров
        for contour in contours:
            flipped_contour = []
            for x, y in contour:
                flipped_contour.append((w - x, y))  # Зеркальное отражение по оси x
            flipped_contours.append(flipped_contour)

        return flipped_entity, flipped_contours

    def rotate90clockwise_annotations(self, entity, contours, img):
        h, w, _ = img.shape
        rotated_entity = []
        rotated_contours = []

        for bbox_from, bbox_to in entity:
            (xmin, ymin), (xmax, ymax) = bbox_from, bbox_to
            xmin2, ymin2 = self.rotate90clockwise_point(xmin, ymin, w, h)
            xmax2, ymax2 = self.rotate90clockwise_point(xmax, ymax, w, h)
            rotated_entity.append(((xmin2, ymin2), (xmax2, ymax2)))

        for contour in contours:
            rotated_contour = []
            for x, y in contour:
                x_new, y_new = self.rotate90clockwise_point(x, y, w, h)
                rotated_contour.append((x_new, y_new))
            rotated_contours.append(rotated_contour)

        return rotated_entity, rotated_contours

    def rotate90clockwise_point(self, x, y, w, h):
        # Вращение точки на 90 градусов по часовой стрелке
        return h - y, x

    @staticmethod
    def get_bounding_box(points):
        # Проверяем наличие точек
        if not points:
            return []

        # Инициируем минимальные и максимальные координаты на основе первой точки
        min_x = min_y = float('inf')
        max_x = max_y = float('-inf')

        # Проходим по всем точкам, чтобы определить мин/макс
        for (x, y) in points:
            if x < min_x:
                min_x = x
            if y < min_y:
                min_y = y
            if x > max_x:
                max_x = x
            if y > max_y:
                max_y = y

        # Возвращаем координаты ограничивающего прямоугольника
        return [min_x, min_y, max_x, max_y]

    def correct_contour(self, contour, width, height):
        contour = self.interpolate_contour(contour)
        corrected_contour = []

        def is_inside(x, y):
            return 0 <= x < width and 0 <= y < height

        def correct_point(x, y, w, h):
            corrected_x = min(max(x, 0), w)
            corrected_y = min(max(y, 0), h)
            return corrected_x, corrected_y

        inside_state = False

        for (x, y) in contour:
            if is_inside(x, y):
                corrected_contour.append((x, y))
                inside_state = True
            else:
                if inside_state:
                    corrected_contour.append(correct_point(x, y, width, height))
                    inside_state = False

                # Проверка для угловых случаев
                if x > width and y > height:
                    if (width, height) not in corrected_contour:
                        corrected_contour.append((width, height))
                elif x > width and y < 0:
                    if (width, 0) not in corrected_contour:
                        corrected_contour.append((width, 0))
                elif x < 0 and y > height:
                    if (0, height) not in corrected_contour:
                        corrected_contour.append((0, height))
                elif x < 0 and y < 0:
                    if (0, 0) not in corrected_contour:
                        corrected_contour.append((0, 0))

        if corrected_contour and corrected_contour[0] != corrected_contour[-1]:
            corrected_contour.append(corrected_contour[0])

        return corrected_contour

    @staticmethod
    def interpolate_contour(contour, step=0.5):
        import numpy as np
        interpolated_contour = []

        for i in range(len(contour)):
            x1, y1 = contour[i]
            x2, y2 = contour[(i + 1) % len(contour)]

            interpolated_contour.append((x1, y1))

            dx = x2 - x1
            dy = y2 - y1
            distance = np.hypot(dx, dy)

            if distance > step:
                num_points = int(np.floor(distance / step))
                new_x = np.linspace(x1, x2, num=num_points, endpoint=False)
                new_y = np.linspace(y1, y2, num=num_points, endpoint=False)

                for nx, ny in zip(new_x[1:], new_y[1:]):
                    interpolated_contour.append((int(round(nx)), int(round(ny))))

        return interpolated_contour

    def random_augmentation(self, img):
        import albumentations as A

        stages = []
        if self.augment_colors:
            stages.append(
                A.HueSaturationValue(hue_shift_limit=360, sat_shift_limit=30, val_shift_limit=20, always_apply=True))
            stages.append(A.ISONoise(p=0.5))
            stages.append(A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0, p=0.5))

            random.shuffle(stages)

            transform = A.Compose(stages)

            img = transform(image=img)["image"]
        return img

    def read_part(self, res_from, res_to):
        import cv2
        import numpy as np

        res_size = res_to - res_from
        assert np.all(res_size >= [self.patch_size, self.patch_size])
        res = np.zeros((res_size[1], res_size[0], 3), np.uint8)
        res[:, :, :] = 255

        tile_xy_from = np.int32(res_from // self.patch_size)
        tile_xy_upto = np.int32((res_to - 1) // self.patch_size)
        assert np.all(tile_xy_from <= tile_xy_upto)
        for tile_x in range(tile_xy_from[0], tile_xy_upto[0] + 1):
            for tile_y in range(tile_xy_from[1], tile_xy_upto[1] + 1):
                if (tile_x, tile_y) not in self.tiles_paths:
                    continue
                part = cv2.imread(self.tiles_paths[tile_x, tile_y])
                part = cv2.copyMakeBorder(part, 0, self.patch_size - part.shape[0], 0, self.patch_size - part.shape[1],
                                          cv2.BORDER_CONSTANT, value=[255, 255, 255])
                part_from = np.int32([tile_x, tile_y]) * self.patch_size - res_from
                part_to = part_from + self.patch_size

                res_inner_from = np.int32([max(0, part_from[0]), max(0, part_from[1])])
                res_inner_to = np.int32([min(part_to[0], res_size[0]), min(part_to[1], res_size[1])])

                part_inner_from = res_inner_from - part_from
                part_inner_to = part_inner_from + res_inner_to - res_inner_from

                res[res_inner_from[1]:res_inner_to[1], res_inner_from[0]:res_inner_to[0], :] = part[part_inner_from[1]:
                                                                                                    part_inner_to[1],
                                                                                               part_inner_from[0]:
                                                                                               part_inner_to[0], :]

        return res

    def intersect(self, a_from, a_to, b_from, b_to):
        import numpy as np
        c_from = np.maximum(a_from, b_from)
        c_to = np.minimum(a_to, b_to)
        if np.any(c_from >= c_to):
            return c_from, c_from
        else:
            return c_from, c_to

    def area(self, a_from, a_to):
        a_size = a_to - a_from
        return a_size[0] * a_size[1]

    def add_pixel_shift(self, to_world, dx, dy):
        to_world = to_world.copy()
        to_world[0, 2] = to_world[0, :] @ [dx, dy, 1]
        to_world[1, 2] = to_world[1, :] @ [dx, dy, 1]
        return to_world

    def invert_matrix_2x3(self, to_world):
        import numpy as np

        to_world33 = np.vstack([to_world, [0, 0, 1]])
        from_world = np.linalg.inv(to_world33)

        assert (from_world[2, 0] == from_world[2, 1] == 0)
        assert (from_world[2, 2] == 1)
        from_world = from_world[:2, :]

        return from_world

    def show_results_dialog(self):
        message = "Finished in {:.2f} sec:\n".format(self.results_time_total)

        print(message)
        Metashape.app.messageBox(message)

    def create_gui(self):
        locale = QtCore.QLocale(QtCore.QLocale.C)
        self.labelTrainZonesLayer = QtWidgets.QLabel("Select zones:")
        self.trainZonesLayer = QtWidgets.QComboBox()
        self.labelTrainDataLayer = QtWidgets.QLabel("Train data:")
        self.trainDataLayer = QtWidgets.QComboBox()

        self.noTrainDataChoice = (None, "Please select...", True)
        self.layers = [self.noTrainDataChoice]

        if self.chunk.shapes is None:
            print("No shapes")
        else:
            for layer in self.chunk.shapes.groups:
                key, label, enabled = layer.key, layer.label, layer.enabled
                print("Shape layer: key={}, label={}, enabled={}".format(key, label, enabled))
                if label == '':
                    label = 'Layer'
                self.layers.append((key, label, layer.enabled))

        for key, label, enabled in self.layers:
            self.trainZonesLayer.addItem(label)
            self.trainDataLayer.addItem(label)
        self.trainZonesLayer.setCurrentIndex(0)
        self.trainDataLayer.setCurrentIndex(0)
        for i, (key, label, enabled) in enumerate(self.layers):
            if not enabled:
                continue

            if  self.expected_layer_name_train_zones.lower() in label.lower():
                self.trainZonesLayer.setCurrentIndex(i)
            if self.expected_layer_name_train_data.lower() in label.lower():
                self.trainDataLayer.setCurrentIndex(i)

        self.chkUse5mmResolution = QtWidgets.QCheckBox("Process with 0.50 cm/pix resolution")
        self.chkUse5mmResolution.setToolTip(
            "Process with downsampling to 0.50 cm/pix instad of original orthomosaic resolution.")
        self.chkUse5mmResolution.setChecked(not self.prefer_original_resolution)

        self.groupBoxGeneral = QtWidgets.QGroupBox("General")
        generalLayout = QtWidgets.QGridLayout()

        self.labelWorkingDir = QtWidgets.QLabel()
        self.labelWorkingDir.setText("Working dir:")
        self.workingDirLineEdit = QtWidgets.QLineEdit()
        self.workingDirLineEdit.setText(self.working_dir)
        self.workingDirLineEdit.setPlaceholderText("Path to dir for intermediate data")
        self.workingDirLineEdit.setToolTip("Path to dir for intermediate data")
        self.btnWorkingDir = QtWidgets.QPushButton("...")
        self.btnWorkingDir.setFixedSize(25, 25)

        QtCore.QObject.connect(self.btnWorkingDir, QtCore.SIGNAL("clicked()"), lambda: self.choose_working_dir())
        generalLayout.addWidget(self.labelWorkingDir, 0, 0)
        generalLayout.addWidget(self.workingDirLineEdit, 0, 1)
        generalLayout.addWidget(self.btnWorkingDir, 0, 2)
        generalLayout.addWidget(self.chkUse5mmResolution, 1, 1)

        self.debugModeCbox = QtWidgets.QCheckBox("Debug mode")
        generalLayout.addWidget(self.debugModeCbox, 4, 1, 1, 2)

        self.maxSizeImageSpinBox = QtWidgets.QSpinBox(self)
        self.maxSizeImageSpinBox.setMaximumWidth(150)
        self.maxSizeImageSpinBox.setMinimum(256)
        self.maxSizeImageSpinBox.setMaximum(2048)
        self.maxSizeImageSpinBox.setSingleStep(256)
        self.maxSizeImageSpinBox.setValue(1024)
        self.maxSizeImageLabel = QtWidgets.QLabel("Max size image:")
        generalLayout.addWidget(self.maxSizeImageLabel, 5, 0)
        generalLayout.addWidget(self.maxSizeImageSpinBox, 5, 1, 1, 2)

        self.loadTilesLabel = QtWidgets.QLabel("Load tiles from working dir:")
        generalLayout.addWidget(self.loadTilesLabel, 6, 0)
        self.loadTilesCbox = QtWidgets.QCheckBox('Off')
        generalLayout.addWidget(self.loadTilesCbox, 6, 1)

        self.groupBoxGeneral.setLayout(generalLayout)

        # Создаем таб-панель
        self.tabWidget = QtWidgets.QTabWidget()

        self.tabDataSetTraining = QtWidgets.QWidget()
        self.tabWidget.addTab(self.tabDataSetTraining, "Dataset Training")
        # Additional dataset training
        self.groupBoxDataSetTraining = QtWidgets.QGroupBox("Additional dataset training")
        trainingDataSetLayout = QtWidgets.QGridLayout()

        # Separation of annotations
        trainingDataSetLayout.addWidget(self.labelTrainZonesLayer, 0, 0)
        trainingDataSetLayout.addWidget(self.trainZonesLayer, 0, 1,1,2)
        trainingDataSetLayout.addWidget(self.labelTrainDataLayer, 1, 0)
        trainingDataSetLayout.addWidget(self.trainDataLayer, 1, 1, 1, 2)

        self.separationDataLable = QtWidgets.QLabel("Separation of annotations (train):")

        self.separationDataSpinBox = CustomDoubleSpinBox()
        self.separationDataSpinBox.setMaximumWidth(150)
        self.separationDataSpinBox.setRange(0.1, 1)
        self.separationDataSpinBox.setSingleStep(0.1)
        self.separationDataSpinBox.setDecimals(1)
        self.separationDataSpinBox.setValue(self.train_percentage)
        self.separationDataSpinBox.setLocale(locale)

        trainingDataSetLayout.addWidget(self.separationDataLable, 2, 0)
        trainingDataSetLayout.addWidget(self.separationDataSpinBox, 2, 1)

        self.labelProportionBackground = QtWidgets.QLabel("Proportion background:")

        self.proportionBackgroundSpinBox = CustomDoubleSpinBox()
        self.proportionBackgroundSpinBox.setMaximumWidth(150)
        self.proportionBackgroundSpinBox.setRange(0.1, 1)
        self.proportionBackgroundSpinBox.setSingleStep(0.1)
        self.proportionBackgroundSpinBox.setDecimals(1)
        self.proportionBackgroundSpinBox.setValue(self.percent_empty_limit)
        self.proportionBackgroundSpinBox.setLocale(locale)
        trainingDataSetLayout.addWidget(self.labelProportionBackground, 3, 0)
        trainingDataSetLayout.addWidget(self.proportionBackgroundSpinBox, 3, 1)


        self.labelAugmentedAnnotations = QtWidgets.QLabel("Augmented data (x8):")
        trainingDataSetLayout.addWidget(self.labelAugmentedAnnotations, 4, 0)

        self.augmentCbox = QtWidgets.QCheckBox("Off")
        trainingDataSetLayout.addWidget(self.augmentCbox, 4, 1)

        self.labelAugmentColorsCbox = QtWidgets.QLabel("Augment colors:")
        trainingDataSetLayout.addWidget(self.labelAugmentColorsCbox, 5, 0)

        self.augmentColorsCbox = QtWidgets.QCheckBox("Off")
        trainingDataSetLayout.addWidget(self.augmentColorsCbox, 5, 1)

        # Создаем чекбоксы
        self.labelMode = QtWidgets.QLabel("Mode:")
        self.cbxDetect = QtWidgets.QCheckBox("Detect (Boxes)")
        self.cbxDetect.setChecked(True)
        self.cbxSegmentation = QtWidgets.QCheckBox("Segmentation (Contours)")
        self.cbxSegmentation.setDisabled(False)

        # Объединяем в группу
        self.buttonGroup = QtWidgets.QButtonGroup(self)
        self.buttonGroup.setExclusive(True)  # Устанавливаем, что выбор только один
        self.buttonGroup.addButton(self.cbxDetect, 0)
        self.buttonGroup.addButton(self.cbxSegmentation, 1)

        trainingDataSetLayout.addWidget(self.labelMode, 6, 0)
        trainingDataSetLayout.addWidget(self.cbxDetect, 6, 1)
        trainingDataSetLayout.addWidget(self.cbxSegmentation, 7, 1)

        self.btnRun = QtWidgets.QPushButton("Run")
        self.btnRun.setMaximumWidth(100)
        self.btnRun.setEnabled(True)

        self.btnStop = QtWidgets.QPushButton("Stop")
        self.btnStop.setEnabled(False)
        self.btnStop.setMaximumWidth(100)

        layout = QtWidgets.QGridLayout()
        row = 0

        layout.addWidget(self.groupBoxGeneral, row, 0, 1, 3)
        row += 1

        self.tabDataSetTraining.setLayout(trainingDataSetLayout)
        layout.addWidget(self.tabWidget, row, 0, 1, 3)
        row += 1

        self.txtInfoPBar = QtWidgets.QLabel()
        self.txtInfoPBar.setText("")
        layout.addWidget(self.txtInfoPBar, row, 1, 1, 3)
        row += 1

        self.txtPBar = QtWidgets.QLabel()
        self.txtPBar.setText("Progress:")
        self.progressBar = QtWidgets.QProgressBar()
        self.progressBar.setTextVisible(True)
        layout.addWidget(self.txtPBar, row, 0)
        layout.addWidget(self.progressBar, row, 1, 1, 2)
        row += 1

        layout.addWidget(self.btnRun, row, 0)
        layout.addWidget(self.btnStop, row, 3)
        row += 1

        self.setLayout(layout)

        QtCore.QObject.connect(self.btnRun, QtCore.SIGNAL("clicked()"), lambda: self.run_process())
        QtCore.QObject.connect(self.btnStop, QtCore.SIGNAL("clicked()"), lambda: self.stop())

        self.buttonGroup.buttonClicked[int].connect(self.check_selection_mode)

        self.debugModeCbox.stateChanged.connect(self.change_debug_mode)
        self.augmentColorsCbox.stateChanged.connect(self.change_augment_colors)
        self.augmentCbox.stateChanged.connect(self.change_augment_data)
        self.loadTilesCbox.stateChanged.connect(self.change_load_tiles)

    def check_selection_mode(self, button_id):
        if button_id == 0:
            self.selected_mode = "detect"
        elif button_id == 1:
            self.selected_mode = "segmentation"

    def change_debug_mode(self, value):
        self.isDebugMode = value
        print(f"Debug mode: {'On' if value else 'Off'}")

    def change_augment_colors(self, value):
        self.augment_colors = value
        self.augmentColorsCbox.setText('On' if value else 'Off')

    def change_augment_data(self, value):
        self.isAugmentData = value
        self.augmentCbox.setText('On' if value else 'Off')

    def change_load_tiles(self, value):
        self.isLoadTiles = value
        self.loadTilesCbox.setText('On' if value else 'Off')

    def choose_working_dir(self):
        working_dir = Metashape.app.getExistingDirectory()
        self.workingDirLineEdit.setText(working_dir)

    def load_params(self):

        app = QtWidgets.QApplication.instance()

        self.prefer_original_resolution = not self.chkUse5mmResolution.isChecked()

        self.percent_empty_limit = self.proportionBackgroundSpinBox.value()
        self.max_image_size = self.maxSizeImageSpinBox.value()
        self.preferred_patch_size = self.max_image_size

        self.working_dir = self.workingDirLineEdit.text()
        self.train_percentage = self.separationDataSpinBox.value()

        Metashape.app.settings.setValue("scripts/create_yolo_dataset/percent_empty_limit",
                                        str(self.percent_empty_limit))
        Metashape.app.settings.setValue("scripts/create_yolo_dataset/max_image_size", str(self.max_image_size))
        Metashape.app.settings.setValue("scripts/create_yolo_dataset/train_percentage", str(self.train_percentage))

        if not self.prefer_original_resolution:
            self.orthomosaic_resolution = self.preferred_resolution
            self.patch_size = self.preferred_patch_size
        else:
            self.orthomosaic_resolution = self.chunk.orthomosaic.resolution
            if self.orthomosaic_resolution > 0.105:
                raise Exception("Orthomosaic should have resolution <= 10 cm/pix.")
            if self.force_small_patch_size:
                patch_size_multiplier = 1
            else:
                patch_size_multiplier = max(1, min(4, self.preferred_resolution / self.orthomosaic_resolution))
            self.patch_size = round(self.preferred_patch_size * patch_size_multiplier)

        self.patch_inner_border = self.patch_size // 8
        print("Using resolution {} m/pix with patch {}x{}".format(self.orthomosaic_resolution, self.patch_size,
                                                                  self.patch_size))


        trainZonesLayer = self.layers[self.trainZonesLayer.currentIndex()]
        trainDataLayer = self.layers[self.trainDataLayer.currentIndex()]

        if trainZonesLayer == self.noTrainDataChoice or trainDataLayer == self.noTrainDataChoice:
            self.train_on_user_data_enabled = False
            print("Additional dataset disabled")
        else:
            self.train_on_user_data_enabled = True
            print("Additional dataset expected on key={} layer data w.r.t. key={} layer zones".format(
                trainDataLayer[0], trainZonesLayer[0]))

        loading_train_shapes_start = time.time()
        self.shapes = self.chunk.shapes
        self.train_zones = []
        self.train_data = []

        print(f"All shapes chunk: {len(self.shapes)}")

        # Получаем ключи слоев для тренировки
        train_zones_key = trainZonesLayer[0]
        train_data_key = trainDataLayer[0]

        print("Grouping shapes by key: {} and {}".format(train_zones_key, train_data_key))
        for i, shape in enumerate(self.shapes):
            if shape.group.key == train_zones_key:
                self.train_zones.append(shape)
            elif shape.group.key == train_data_key:
                self.train_data.append(shape)

            i+=1
            self.progressBar.setValue(int((100 * i + 1) / len(self.shapes)))
            app.processEvents()
            self.check_stopped()



        print("{} zones and {} data loaded in {:.2f} sec".format(len(self.train_zones),
                                                                             len(self.train_data),
                                                                             time.time() - loading_train_shapes_start))

    def debug_draw_objects(self, img, bboxes, contours):
        import cv2
        import numpy as np

        img = img.copy()
        h, w, cn = img.shape

        for tree in bboxes:
            # Преобразование к целым числам
            xmin, ymin, xmax, ymax = map(int, map(round, tree))

            # Проверка значений bbox
            assert np.all(np.array([xmin, ymin]) >= np.int32([0, 0])), \
                f"Bounding box values out of bounds: {xmin}, {ymin}"
            assert np.all(np.array([xmax, ymax]) <= np.int32([w, h])), \
                f"Bounding box values out of bounds: {xmax}, {ymax} (image size: {w}, {h})"
            # if np.all(np.array([xmin, ymin]) >= np.int32([0, 0])):
            #     xmin = max(xmin, 0)
            #     ymin = max(ymin, 0)
            # if np.all(np.array([xmax, ymax]) <= np.int32([w, h])):
            #     xmax = min(xmax, w)
            #     ymax = min(ymax, h)

            # Преобразование в массив numpy
            (xmin, ymin), (xmax, ymax) = np.array([xmin, ymin]), np.array([xmax, ymax])

            # Отрисовка прямоугольника
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

        # Отрисовка контуров
        for contour in contours:
            # corrected_contour = []
            # for (x, y) in contour:
            #     x = min(max(x, 0), w)
            #     y = min(max(y, 0), h)
            #     corrected_contour.append([x, y])
            contour = np.array(contour, dtype=np.int32).reshape(-1, 2)
            cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)

        return img


def detect_objects():
    chunk = Metashape.app.document.chunk

    if chunk is None or chunk.orthomosaic is None:
        raise Exception("No active orthomosaic.")

    if chunk.orthomosaic.resolution > 0.105:
        raise Exception("Orthomosaic should have resolution <= 10 cm/pix.")

    app = QtWidgets.QApplication.instance()
    parent = app.activeWindow()
    dlg = MainWindowDetect(parent)

def convert_shapes():
    chunk = Metashape.app.document.chunk

    if chunk is None or chunk.orthomosaic is None:
        raise Exception("No active orthomosaic.")

    if chunk.orthomosaic.resolution > 0.105:
        raise Exception("Orthomosaic should have resolution <= 10 cm/pix.")

    app = QtWidgets.QApplication.instance()
    parent = app.activeWindow()
    dlg = WindowConvertShapes(parent)

def create_yolo_dataset():
    chunk = Metashape.app.document.chunk

    if chunk is None or chunk.orthomosaic is None:
        raise Exception("No active orthomosaic.")

    if chunk.orthomosaic.resolution > 0.105:
        raise Exception("Orthomosaic should have resolution <= 10 cm/pix.")

    app = QtWidgets.QApplication.instance()
    parent = app.activeWindow()
    dlg = WindowCreateYoloDataset(parent)


Metashape.app.addMenuItem("Scripts/YOLO", detect_objects)
print("To execute this script press {}".format("Scripts/YOLO"))

Metashape.app.addMenuItem("Scripts/Convert shapes", convert_shapes)
print("To execute this script press {}".format("Scripts/Convert shapes"))

Metashape.app.addMenuItem("Scripts/Create yolo dataset", create_yolo_dataset)
print("To execute this script press {}".format("Scripts/Create yolo dataset"))
