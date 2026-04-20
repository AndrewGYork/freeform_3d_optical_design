from PyQt5.QtWidgets import QLabel, QSizePolicy
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

class ScaledLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._orig = None
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(100, 100)  # allow shrinking gracefully
        # self.setScaledContents(True)

    def setPixmap(self, pm: QPixmap):
        """Store original pixmap and show a scaled copy."""
        self._orig = pm
        super().setPixmap(pm)
        self._rescale()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self):
        if not self._orig or self.width() == 0 or self.height() == 0:
            return
        # Use the label's content rect so scrollbars/margins don't skew size
        target_size = self.contentsRect().size()
        print(target_size)
        scaled = self._orig.scaled(
            target_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        super().setPixmap(scaled)
