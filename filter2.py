import sys
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QGridLayout,
    QMessageBox,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import qdarkstyle
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from skimage import io, util


class AppDemo(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spatial Filter Simulation")
        self.setFixedSize(600, 800)  # 윈도우 크기 설정
        self.image = None

        self.canvas_before = FigureCanvas(plt.Figure(figsize=(3, 3)))  # 크기 조절
        self.canvas_after = FigureCanvas(plt.Figure(figsize=(3, 3)))  # 크기 조절

        self.initUI()

    def initUI(self):
        # 레이아웃 및 스타일 개선
        self.setStyleSheet("font: 11pt Sans-serif; color: #ffffff;")
        grid_layout = QGridLayout()
        grid_layout.setSpacing(2)  # 그리드의 간격을 설정합니다.

        self.canvas_before = FigureCanvas(plt.Figure(figsize=(5, 5)))
        self.canvas_after = FigureCanvas(plt.Figure(figsize=(5, 5)))

        button_width = 120
        button_height = 40
        laplacian_button_width = 2 * button_width + 10

        # 업로드 이미지 버튼 추가 및 중앙 정렬
        self.load_image_btn = QPushButton("Upload Image")
        self.load_image_btn.clicked.connect(self.load_image)
        self.load_image_btn.setFixedSize(200, 50)
        grid_layout.addWidget(
            self.load_image_btn, 0, 0, 1, 4, Qt.AlignCenter
        )  # 가운데 정렬 옵션 추가

        # 간단한 프로그램 설명 레이블 추가
        description_label = QLabel("Select an image file to apply spatial filters.")
        description_label.setAlignment(Qt.AlignCenter)
        description_label.setFont(QFont("Arial", 10))
        description_label.setStyleSheet(
            "QLabel { color: #FFFFFF; }"
        )  # 글자 색상을 밝은 색으로 설정
        grid_layout.addWidget(
            description_label, 1, 0, 1, 4
        )  # 레이블을 그리드 레이아웃의 첫 번째 행에 추가

        # Mean Filter Buttons
        mean_filter_label = QLabel("")
        mean_filter_label.setStyleSheet(
            "QLabel { color: #FFFFFF; }"
        )  # 글자 색상을 밝은 색으로 설정
        grid_layout.addWidget(
            mean_filter_label, 3, 0, 1, 2
        )  # 레이블을 그리드 레이아웃의 세 번째 행에 추가
        # Mean Filter 버튼들 초기화 및 추가, 중앙 정렬
        self.mean_filter_buttons = [QPushButton(f"Mean {i}x{i}") for i in [3, 5, 7]]
        for i, btn in enumerate(self.mean_filter_buttons):
            btn.clicked.connect(
                lambda chk, size=3 + i * 2: self.apply_mean_filter(size)
            )
            btn.setFixedSize(120, 40)
            grid_layout.addWidget(btn, 2 + i, 0, 1, 2, Qt.AlignCenter)  # 가운데 정렬 옵션 추가

        # Median Filter Buttons 시작 행을 조정
        median_filter_label = QLabel("")
        median_filter_label.setStyleSheet(
            "QLabel { color: #FFFFFF; }"
        )  # 글자 색상을 밝은 색으로 설정
        grid_layout.addWidget(
            median_filter_label, 3, 2, 1, 2
        )  # 레이블을 그리드 레이아웃의 세 번째 행에 추가
        self.median_filter_buttons = [QPushButton(f"Median {i}x{i}") for i in [3, 5, 7]]
        for i, btn in enumerate(self.median_filter_buttons):
            btn.clicked.connect(
                lambda chk, size=3 + i * 2: self.apply_median_filter(size)
            )
            btn.setFixedSize(120, 40)
            grid_layout.addWidget(btn, 2 + i, 2, 1, 2, Qt.AlignCenter)  # 가운데 정렬 옵션 추가

        # Laplacian Filter Button
        self.laplacian_filter_btn = QPushButton("Laplacian Filter")
        self.laplacian_filter_btn.clicked.connect(self.apply_laplacian_filter)
        self.laplacian_filter_btn.setFixedSize(laplacian_button_width, button_height)
        self.laplacian_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])

        button_style = """
        QPushButton {
            color: white;
            background-color: #4a4a4a;
            border-radius: 5px;
            padding: 10px;
        }
        QPushButton:hover {
            background-color: #5a5a5a;
        }
        """
        for btn in (
            self.mean_filter_buttons
            + self.median_filter_buttons
            + [self.laplacian_filter_btn]
        ):
            btn.setStyleSheet(button_style)

        self.laplacian_filter_btn.setStyleSheet(
            """
            QPushButton {
                color: white;
                background-color: #5e5e5e;
                border-radius: 3px;
                padding: 5px 0;
                margin: 5px 0;
            }
            QPushButton:hover {
                background-color: #707070;
            }
            """
        )
        grid_layout.addWidget(self.laplacian_filter_btn, 5, 0, 1, 4)

        # Labels for before and after images
        self.label_before = QLabel("Before")
        self.label_after = QLabel("After")
        self.label_before.setAlignment(Qt.AlignCenter)
        self.label_after.setAlignment(Qt.AlignCenter)
        self.label_before.setFont(QFont("Arial", 12))
        self.label_after.setFont(QFont("Arial", 12))

        # Image display layout
        image_layout = QVBoxLayout()
        image_layout.addWidget(self.canvas_before)
        image_layout.addWidget(self.label_before)
        image_layout.addWidget(self.canvas_after)
        image_layout.addWidget(self.label_after)

        # Set the layout
        main_layout = QHBoxLayout()
        main_layout.addLayout(image_layout, 1)
        main_layout.addSpacing(20)
        main_layout.addLayout(grid_layout, 1)
        self.setLayout(main_layout)

    def apply_laplacian_filter(self):
        if self.image is not None:
            # 적절한 데이터 타입으로 이미지 변환
            img_to_filter = util.img_as_float(self.image)
            # 라플라시안 커널 적용
            filtered_image = cv2.filter2D(img_to_filter, -1, self.laplacian_kernel)
            # 결과 이미지의 범위를 [0, 1]로 조정
            filtered_image = np.clip(filtered_image, 0, 1)
            # float 이미지를 다시 8비트 이미지로 변환
            filtered_image = util.img_as_ubyte(filtered_image)
            self.display_image(filtered_image, self.canvas_after)
        else:
            self.show_error_message("Please load an image first.")

    def apply_filter(self, kernel):
        if self.image is not None:  # Check if the image is loaded
            filtered_image = cv2.filter2D(util.img_as_ubyte(self.image), -1, kernel)
            self.display_image(filtered_image, self.canvas_after)
        else:
            self.show_error_message("Please load an image first.")

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_name:
            self.image = io.imread(file_name)
            self.display_image(self.image, self.canvas_before)

    def apply_mean_filter(self, size):
        if self.image is not None:  # Check if the image is loaded
            kernel = np.ones((size, size), np.float32) / (
                size**2
            )  # Correct the power operator for size
            self.apply_filter(kernel)  # Call the correct method
        else:
            self.show_error_message("Please load an image first.")

    def apply_median_filter(self, size):
        if self.image is not None:  # Check if the image is loaded
            filtered_image = cv2.medianBlur(util.img_as_ubyte(self.image), size)
            self.display_image(filtered_image, self.canvas_after)
        else:
            self.show_error_message("Please load an image first.")

    def display_image(self, image, canvas):
        canvas.figure.clf()
        ax = canvas.figure.subplots()
        ax.imshow(image, cmap="gray")
        ax.axis("off")
        canvas.draw()

    def show_error_message(self, message):
        QMessageBox.critical(self, "Error", message)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    demo = AppDemo()
    demo.show()
    sys.exit(app.exec_())
