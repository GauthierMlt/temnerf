import sys
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QSlider, QPushButton, QFileDialog
from PySide6.QtCore import Qt, QRect, QSize
from PySide6.QtGui import QPixmap, QImage

class SliceViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Volume Viewer")
        self.volume = None
        self.current_axis = 'x'
        self.initUI()
        self.centerWindow()

    def initUI(self):
        main_layout = QVBoxLayout()

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(800, 800)
        main_layout.addWidget(self.image_label)

        self.sliders = {}
        self.value_labels = {}

        for axis in ['x', 'y', 'z']:
            slider = QSlider(Qt.Horizontal, self)
            slider.setMinimum(0)
            slider.valueChanged.connect(lambda _, axis=axis: self.update_image(axis))
            self.sliders[axis] = slider

            value_label = QLabel('0', self)
            self.value_labels[axis] = value_label

            layout = QHBoxLayout()
            layout.addWidget(QLabel(f'{axis.upper()}'))
            layout.addWidget(slider)
            layout.addWidget(value_label)
            main_layout.addLayout(layout)

        load_button = QPushButton('Load .npy file', self)
        load_button.clicked.connect(self.load_npy)
        main_layout.addWidget(load_button)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def centerWindow(self):
        screen_geometry = QApplication.primaryScreen().availableGeometry()
        window_geometry = self.frameGeometry()
        window_geometry.moveCenter(screen_geometry.center())
        self.move(window_geometry.topLeft())

    def load_npy(self):
        file_dialog = QFileDialog(self)
        file_path, _ = file_dialog.getOpenFileName(self, "Open .npy file", "", "NumPy Files (*.npy)")
        if file_path:
            self.volume = np.load(file_path)
            self.update_sliders()

    def update_sliders(self):
        if self.volume is not None:
            self.sliders['x'].setMaximum(self.volume.shape[0] - 1)
            self.sliders['y'].setMaximum(self.volume.shape[1] - 1)
            self.sliders['z'].setMaximum(self.volume.shape[2] - 1)
            self.update_image('x')

    def update_image(self, axis):
        if self.volume is not None:
            self.current_axis = axis
            x_index = self.sliders['x'].value()
            y_index = self.sliders['y'].value()
            z_index = self.sliders['z'].value()

            self.value_labels['x'].setText(str(x_index))
            self.value_labels['y'].setText(str(y_index))
            self.value_labels['z'].setText(str(z_index))

            if axis == 'x':
                slice_ = self.volume[x_index, :, :]
            elif axis == 'y':
                slice_ = self.volume[:, y_index, :]
            elif axis == 'z':
                slice_ = self.volume[:, :, z_index]

            self.display_image(slice_)

    def display_image(self, slice_):
        slice_[slice_ < 0] = 0
        slice_normalized = (255 *(slice_ - np.min(slice_)) / np.ptp(slice_)).astype(np.uint8)
        height, width = slice_normalized.shape
        q_image = QImage(slice_normalized.data, width, height, slice_normalized.strides[0], QImage.Format_Grayscale8)

        scaled_q_image = q_image.scaled(800, 800, Qt.KeepAspectRatio)

        self.image_label.setPixmap(QPixmap.fromImage(scaled_q_image))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = SliceViewer()
    viewer.show()
    sys.exit(app.exec())
