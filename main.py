import sys
import cv2
import numpy as np
import os
import logging
from termcolor import colored
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, \
                            QPushButton, QLabel, QFileDialog, QSizePolicy, QSlider
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt

# logger
logger = logging.getLogger()

# Stream handler for logging output
console_handler = logging.StreamHandler()

# Formatter for log messages
formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s'
)

console_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.setLevel(logging.DEBUG)

# main class
class ObjectRecognitionApp(QWidget):
    def __init__(self):
        super().__init__()

        # app start size
        self.app_width = 700
        self.app_height = 650

        # titlu
        self.setWindowTitle("Recunoastere de obiecte folosind key points")

        # default size
        self.resize(self.app_width, self.app_height)

        # # disable resize - set default size
        # self.setFixedSize(self.app_width, self.app_height)

        # main box
        self.layout = QVBoxLayout()

        # top box
        self.top_layout = QHBoxLayout()

        # buttons - image
        self.button_layout = QVBoxLayout()

        self.load_button = QPushButton("Load Image", self)
        self.load_button.clicked.connect(self.load_image)
        self.button_layout.addWidget(self.load_button)

        # toggle button
        self.toggle_view_button = QPushButton("Show Key Points", self)
        self.toggle_view_button.clicked.connect(self.toggle_view)
        self.button_layout.addWidget(self.toggle_view_button)

        self.save_image_button = QPushButton("Save Image", self)
        self.save_image_button.clicked.connect(self.save_image)
        self.button_layout.addWidget(self.save_image_button)


        self.top_layout.addLayout(self.button_layout)

        # execution time label
        self.execution_time_label = QLabel("Execution Time: Not yet calculated", self)
        self.layout.addWidget(self.execution_time_label)

        # keypoints count
        self.keypoints_label = QLabel("Keypoints: 100", self)
        self.layout.addWidget(self.keypoints_label)

        # keypoints slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(10, 1000)
        self.slider.setValue(100)
        self.slider.valueChanged.connect(self.update_keypoints)
        self.layout.addWidget(self.slider)


        # imaginea prelucrata
        self.image_label = QLabel(self)
        self.image_label.setText("No Image Loaded")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet(
            "border: 3px dashed black;"
            "background-color: lightgray;"
        )
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.layout.addWidget(self.image_label)

        # imaginea originala
        self.image_box_label = QLabel(self)
        self.image_box_label.setText("No Image Loaded")
        self.image_box_label.setAlignment(Qt.AlignCenter)
        self.image_box_label.setStyleSheet(
            "border: 3px solid black;"
            "background-color: lightgray;"
        )
        self.image_box_label.setFixedSize(300, 200)


        self.top_layout.addWidget(self.image_box_label)

        self.layout.addLayout(self.top_layout)

        # main layout
        self.setLayout(self.layout)

        # initializam variabilele
        self.test_image = None
        self.original_image = None
        self.test_keypoints = None
        self.test_descriptors = None
        self.orb = cv2.ORB_create()
        self.show_keypoints = False


    def load_image(self):
        try:
            file_path, _ = QFileDialog.getOpenFileName(self, 'Open Image', '', 'Images (*.png *.jpg *.bmp *.tiff)')

            if not file_path:
                self.log_with_color("No file selected for loading.", "yellow")
                raise ValueError("No file selected")

            # load test image
            self.original_image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

            if self.original_image is None:

                self.log_with_color(f"Failed to load image from {file_path}", "red")
                raise ValueError(f"Error: Could not load test image from {file_path}")

            self.log_with_color(f"Test image loaded from {file_path}", "green")

            # convert image to default look
            rgb_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)

            # store test image for processing
            self.test_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)

            # luam keypoints si descriptors din imagine
            start_time = cv2.getTickCount()
            self.test_keypoints, self.test_descriptors = self.orb.detectAndCompute(self.test_image, None)

            # luam numarul de keypoints din slider
            max_keypoints = self.slider.value()
            self.test_keypoints = self.test_keypoints[:max_keypoints]

            # draw keypoints image
            if self.show_keypoints:

                keypoints_image = cv2.drawKeypoints(self.test_image, self.test_keypoints, None, color=(0, 255, 0), flags=0)
                polygon_image = keypoints_image
            else:
                # draw polygon image
                polygon_image = self.process_bounding_box()

            # calculam timpul de executie
            execution_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()

            # processed image
            self.display_image(polygon_image, execution_time)

            # original image (neschimbata)
            self.display_original_image(rgb_image)


        except Exception as e:
            self.log_with_color(f"Error loading test image: {e}", "red")


    # toggle between box - keypoints
    def toggle_view(self):

        # swap bool value
        self.show_keypoints = not self.show_keypoints

        if self.show_keypoints:

            self.toggle_view_button.setText("Show Bounding Box")
            self.log_with_color("Switched to Key Points view", "green")
        else:

            self.toggle_view_button.setText("Show Key Points")
            self.log_with_color("Switched to Bounding Box view", "green")


    def save_image(self):
        try:

            if self.test_image is None:
                self.log_with_color("No test image loaded for saving.", "yellow")
                raise ValueError("Test image is not loaded.")

            default_filename = "test_image_with_keypoints_or_bounding_box.png"


            file_path, _ = QFileDialog.getSaveFileName(self, 'Save Image', default_filename, 'Images (*.png *.jpg *.bmp *.tiff)')

            if not file_path:
                self.log_with_color("No file selected for saving.", "yellow")
                return

            # check if the file exists
            base_name, ext = os.path.splitext(file_path)
            counter = 1

            while os.path.exists(file_path):
                file_path = f"{base_name}({counter}){ext}"
                counter += 1

            result_image = self.process_image()

            # save image
            cv2.imwrite(file_path, result_image)
            self.log_with_color(f"Image saved to {file_path}", "green")

        except Exception as e:
            self.log_with_color(f"Error saving image: {e}", "red")


    # function for the save file functionality
    def process_image(self):
        # process chosen image
        if self.show_keypoints:

            return cv2.drawKeypoints(self.test_image, self.test_keypoints, None, color=(0, 255, 0), flags=0)
        else:
            return self.process_bounding_box()


    # creates the bounding box displayed around the object
    def process_bounding_box(self):

        # check if there are enough keypoints
        if len(self.test_keypoints) > 4:

            points = np.float32([kp.pt for kp in self.test_keypoints]).reshape(-1, 1, 2)
            hull = cv2.convexHull(points)
            result_image = cv2.cvtColor(self.test_image, cv2.COLOR_GRAY2BGR)
            cv2.polylines(result_image, [np.int32(hull)], True, (0, 255, 0), 2)

        else:
            result_image = cv2.cvtColor(self.test_image, cv2.COLOR_GRAY2BGR)

        return result_image


    # display the processed image on the interface
    def display_image(self, image, exec_time):

        try:
            height, width, channel = image.shape
            # Since the image is RGB (3 channels), the number of bytes per line is calculated as 3 * width
            bytes_per_line = 3 * width
            qimg = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)

            pixmap = QPixmap.fromImage(qimg)
            pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

            self.image_label.setPixmap(pixmap) # processed image
            self.image_label.setScaledContents(True)

            self.execution_time_label.setText(f"Execution Time: {exec_time:.4f} seconds")

        except Exception as e:
            self.log_with_color(f"Error displaying image: {e}", "red")


    # display the original image on the interface
    def display_original_image(self, image):

        try:
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            qimg = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)

            pixmap = QPixmap.fromImage(qimg)
            pixmap = pixmap.scaled(self.image_box_label.size(), Qt.KeepAspectRatio)

            self.image_box_label.setPixmap(pixmap)
            self.image_box_label.setScaledContents(True)

        except Exception as e:
            self.log_with_color(f"Error displaying original image: {e}", "red")


    # update keypoints number text on interface
    def update_keypoints(self):
        self.keypoints_label.setText(f"Keypoints: {self.slider.value()}")

    # log cu o anumita culoare
    def log_with_color(self, message, color):

        colored_message = colored(message, color)
        print(colored_message)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ObjectRecognitionApp()
    window.show()
    sys.exit(app.exec_())
