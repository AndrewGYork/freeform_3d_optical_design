from qtpy.QtWidgets import QWidget, QLabel, QSizePolicy, QVBoxLayout
from qtpy.QtCore import Qt
from qtpy.QtGui import QPixmap
from gui.visualization.visualization import Visualization
from gui.visdata import VisData
from typing import Optional

class PngVisualization(Visualization):
    def __init__(self, key: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._image_label = QLabel(f"No {key} yet.")
        self._image_label.setAlignment(Qt.AlignCenter)
        self._image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._image_label.setScaledContents(True)
        self._key = key

        layout = QVBoxLayout()
        layout.addWidget(self._image_label)
        self.setLayout(layout)

    def update(self, data: VisData):
        img = getattr(data, self._key)
        if img is None:
            self._image_label.setText(f"{self._key} not found.")
            return
        # Convert QImage -> QPixmap and display
        pix = QPixmap.fromImage(img)
        self._image_label.setPixmap(pix)
        self._image_label.setMinimumSize(320, 240)
