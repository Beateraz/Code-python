import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QSlider, QPushButton,
    QWidget, QGridLayout, QFileDialog, QLineEdit
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class ImageProcessorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Điều chỉnh ảnh")
        self.setGeometry(100, 100, 1200, 800)

        # Biến lưu ảnh gốc và ảnh đã chỉnh sửa
        self.original_image = None
        self.processed_image = None
        self.edge_image = None

        # Giao diện chính
        self.init_ui()

    def init_ui(self):
        main_layout = QGridLayout()

        # Thiết lập các thành phần giao diện
        self.original_image_label = QLabel("Ảnh gốc")
        self.original_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.original_image_label.setFixedSize(400, 300)

        self.processed_image_label = QLabel("Ảnh đã chỉnh sửa")
        self.processed_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.processed_image_label.setFixedSize(400, 300)

        self.edge_image_label = QLabel("Ảnh phát hiện biên")
        self.edge_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.edge_image_label.setFixedSize(400, 300)

        # Slider điều chỉnh độ tương phản
        self.contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.contrast_slider.setRange(0, 50)
        self.contrast_slider.setValue(20)  # Giá trị mặc định
        self.contrast_slider.valueChanged.connect(self.adjust_contrast)

        # Nút chọn ảnh
        self.load_button = QPushButton("Chọn ảnh")
        self.load_button.clicked.connect(self.load_image)

        # Nút lưu ảnh
        self.save_button = QPushButton("Lưu ảnh")
        self.save_button.clicked.connect(self.save_image)
        self.save_button.setEnabled(False)

        # Slider điều chỉnh ngưỡng dưới và ngưỡng trên cho Canny
        self.lower_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.lower_threshold_slider.setRange(0, 255)
        self.lower_threshold_slider.setValue(50)
        self.lower_threshold_slider.valueChanged.connect(self.detect_edges)

        self.upper_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.upper_threshold_slider.setRange(0, 255)
        self.upper_threshold_slider.setValue(150)
        self.upper_threshold_slider.valueChanged.connect(self.detect_edges)

        # Nhập góc để xoay ảnh
        self.rotate_angle_input = QLineEdit()
        self.rotate_angle_input.setPlaceholderText("Nhập góc xoay")

        # Nút xoay ảnh
        self.rotate_button = QPushButton("Xoay ảnh")
        self.rotate_button.clicked.connect(self.rotate_image)

        # Biểu đồ histogram
        self.histogram_canvas = FigureCanvas(plt.figure(figsize=(4, 2)))

        # Slider điều chỉnh độ nhiễu
        self.noise_slider = QSlider(Qt.Orientation.Horizontal)
        self.noise_slider.setRange(0, 100)
        self.noise_slider.setValue(0)  # Mặc định không có nhiễu
        self.noise_slider.valueChanged.connect(self.add_noise)

        # Slider điều chỉnh độ sắc nét
        self.sharpness_slider = QSlider(Qt.Orientation.Horizontal)
        self.sharpness_slider.setRange(0, 10)
        self.sharpness_slider.setValue(1)  # Mặc định độ sắc nét là 1 (không thay đổi)
        self.sharpness_slider.valueChanged.connect(self.sharpen_image)

        # Slider điều chỉnh độ sáng
        self.brightness_slider = QSlider(Qt.Orientation.Horizontal)
        self.brightness_slider.setRange(-100, 100)  # Giá trị từ -100 đến 100
        self.brightness_slider.setValue(0)  # Giá trị mặc định
        self.brightness_slider.valueChanged.connect(self.adjust_brightness)

        # Bố cục giao diện
        main_layout.addWidget(self.original_image_label, 0, 0, 1, 1)
        main_layout.addWidget(self.processed_image_label, 0, 1, 1, 1)
        main_layout.addWidget(self.edge_image_label, 0, 2, 1, 1)

        main_layout.addWidget(self.load_button, 1, 0, 1, 1)
        main_layout.addWidget(self.save_button, 1, 1, 1, 1)

        main_layout.addWidget(QLabel("Điều chỉnh độ tương phản"), 2, 0, 1, 1)
        main_layout.addWidget(self.contrast_slider, 2, 1, 1, 2)

        # Các slider điều chỉnh ngưỡng Canny
        main_layout.addWidget(QLabel("Ngưỡng dưới Biên"), 3, 0)
        main_layout.addWidget(self.lower_threshold_slider, 3, 1, 1, 2)

        main_layout.addWidget(QLabel("Ngưỡng trên Biên"), 4, 0)
        main_layout.addWidget(self.upper_threshold_slider, 4, 1, 1, 2)

        # Nhập thông số cho xoay ảnh
        main_layout.addWidget(QLabel("Xoay ảnh (Góc xoay)"), 5, 0, 1, 1)
        main_layout.addWidget(self.rotate_angle_input, 5, 1, 1, 2)

        main_layout.addWidget(self.rotate_button, 6, 0, 1, 3)

        # Các slider điều chỉnh độ nhiễu và độ sắc nét
        main_layout.addWidget(QLabel("Điều chỉnh độ nhiễu"), 7, 0)
        main_layout.addWidget(self.noise_slider, 7, 1, 1, 2)

        main_layout.addWidget(QLabel("Điều chỉnh độ sắc nét"), 8, 0)
        main_layout.addWidget(self.sharpness_slider, 8, 1, 1, 2)

        # Slider điều chỉnh độ sáng
        main_layout.addWidget(QLabel("Điều chỉnh độ sáng"), 9, 0)
        main_layout.addWidget(self.brightness_slider, 9, 1, 1, 2)

        # Biểu đồ histogram
        main_layout.addWidget(self.histogram_canvas, 10, 0, 1, 3)

        # Chèn layout vào cửa sổ chính
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Chọn ảnh", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            self.original_image = cv2.imread(file_path)
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            self.processed_image = self.original_image.copy()

            self.display_image(self.original_image, self.original_image_label)
            self.display_image(self.processed_image, self.processed_image_label)

            self.update_histogram(self.original_image, self.processed_image)
            self.save_button.setEnabled(True)

    def save_image(self):
        if self.processed_image is not None:
            file_path, _ = QFileDialog.getSaveFileName(self, "Lưu ảnh", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
            if file_path:
                save_image = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(file_path, save_image)

    def display_image(self, image, label):
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        label.setPixmap(pixmap.scaled(label.width(), label.height(), Qt.AspectRatioMode.KeepAspectRatio))

    def adjust_contrast(self):
        if self.original_image is not None:
            contrast_value = self.contrast_slider.value()
            alpha = 1 + contrast_value / 10.0
            beta = 0
            contrast_adjusted = cv2.convertScaleAbs(self.original_image, alpha=alpha, beta=beta)
            self.processed_image = contrast_adjusted
            self.display_image(self.processed_image, self.processed_image_label)
            self.update_histogram(self.original_image, self.processed_image)

    def adjust_brightness(self):
        if self.original_image is not None:
            brightness_value = self.brightness_slider.value()
            beta = brightness_value
            brightness_adjusted = cv2.convertScaleAbs(self.original_image, alpha=1, beta=beta)
            self.processed_image = brightness_adjusted
            self.display_image(self.processed_image, self.processed_image_label)
            self.update_histogram(self.original_image, self.processed_image)

    def detect_edges(self):
        if self.original_image is not None:
            lower_threshold = self.lower_threshold_slider.value()
            upper_threshold = self.upper_threshold_slider.value()
            gray_image = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray_image, lower_threshold, upper_threshold)
            self.edge_image = edges
            edge_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            self.display_image(edge_colored, self.edge_image_label)

    def rotate_image(self):
        if self.processed_image is not None:
            angle = float(self.rotate_angle_input.text())
            (h, w) = self.processed_image.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(self.processed_image, rotation_matrix, (w, h))
            self.processed_image = rotated
            self.display_image(self.processed_image, self.processed_image_label)

    def add_noise(self):
        if self.original_image is not None:
            noise_level = self.noise_slider.value()
            noise = np.random.randint(0, noise_level, self.original_image.shape, dtype='uint8')
            noisy_image = cv2.add(self.original_image, noise)
            self.processed_image = noisy_image
            self.display_image(self.processed_image, self.processed_image_label)

    def sharpen_image(self):
        if self.original_image is not None:
            sharpness_value = self.sharpness_slider.value()
            kernel = np.array([[0, -1, 0], [-1, 5 + sharpness_value, -1], [0, -1, 0]])
            sharpened = cv2.filter2D(self.original_image, -1, kernel)
            self.processed_image = sharpened
            self.display_image(self.processed_image, self.processed_image_label)

    def update_histogram(self, original_image, processed_image):
        self.histogram_canvas.figure.clf()
        ax = self.histogram_canvas.figure.add_subplot(111)
        original_hist = cv2.calcHist([original_image], [0], None, [256], [0, 256])
        processed_hist = cv2.calcHist([processed_image], [0], None, [256], [0, 256])
        ax.plot(original_hist, color='blue', label='Ảnh gốc')
        ax.plot(processed_hist, color='red', label='Ảnh đã chỉnh sửa')
        ax.set_xlim(0, 255)
        ax.set_title("Histogram")
        ax.legend()
        self.histogram_canvas.draw()



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageProcessorApp()
    window.show()
    sys.exit(app.exec())
