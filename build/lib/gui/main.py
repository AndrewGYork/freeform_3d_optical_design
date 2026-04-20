import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from gui.visdata import VisData
from gui.watcher import DataWatcher
from gui.visualization import Visualization, PngVisualization, VolumeOrthoSlicesVisualization

OUTPUT_DIR = Path(__file__).parent.parent / "output"
POLL_INTERVAL_MS = 1000

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

class VisualizationWindow(QMainWindow):
    def __init__(self, widget: Visualization, on_close_callback, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle(widget.name)
        self.setCentralWidget(widget)
        self.viz = widget
        self._on_close_callback = on_close_callback

    def closeEvent(self, event):
        try:
            self._on_close_callback(self.viz)
        finally:
            return super().closeEvent(event)


# ------------------------------
# Main Window
# ------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GRIN Optimizer Visualization")
        self.setMinimumSize(480, 640)

        # Data watcher & live updates
        self.watcher = DataWatcher(OUTPUT_DIR, POLL_INTERVAL_MS, self)

        # Track open visualization windows by name
        self._open_windows: Dict[str, VisualizationWindow] = {}

        # Root content: a scrollable vertical stack
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        vbox = QVBoxLayout(container)

        # Title label
        title = QLabel("<h2>GRIN Optimizer Visualization</h2>")
        title.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        vbox.addWidget(title)

        # Live updates checkbox
        live_box = QHBoxLayout()
        self.live_checkbox = QCheckBox("Live updates")
        self.live_checkbox.setChecked(True)
        self.live_checkbox.toggled.connect(self._on_live_toggled)
        live_box.addWidget(self.live_checkbox)
        live_box.addStretch(1)
        vbox.addLayout(live_box)

        # Divider line
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        vbox.addWidget(line)

        visualizations = [
            PngVisualization("loss_history_image").rename("Loss History"),
            VolumeOrthoSlicesVisualization("concentration").rename("Material Orthoview")
        ]

        if isinstance(os.environ.get("DEBUG", None), str):
            visualizations += [
                VolumeOrthoSlicesVisualization("concentration").rename("DEBUG: Composition Orthoview")
            ]

        # Visualization buttons
        for viz in visualizations:
            btn = QPushButton(viz.name)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            btn.clicked.connect(lambda checked=False, c=viz: self._open_or_focus_visualization(c))
            vbox.addWidget(btn)

        vbox.addStretch(1)

        scroll.setWidget(container)
        self.setCentralWidget(scroll)

        # Connect watcher to propagate updates to all open visualization widgets
        self.watcher.data_updated.connect(self._on_data_updated)

        # Kick things off
        self._start_watching_initial()

    # ----- Event handlers -----
    def _start_watching_initial(self):
        try:
            self.watcher.start()
        except Exception as e:
            QMessageBox.warning(self, "Initialization Error", f"Failed to start watcher: {e}")

    def _on_live_toggled(self, enabled: bool):
        self.watcher.set_live_updates(enabled)
        if enabled:
            # Emit immediately so open windows refresh now
            self.watcher.force_emit()

    def _on_data_updated(self, data: VisData):
        # Push updates to all open visualization widgets
        for viz in self._open_windows.keys():
            viz.update(data)

    def _open_or_focus_visualization(self, viz):
        if viz in self._open_windows and self._open_windows[viz] is not None:
            win = self._open_windows[viz]
            win.show()
            win.raise_()
            win.activateWindow()
            return

        # Create a new instance and wire it up
        win = VisualizationWindow(viz, self._on_viz_window_closed, self)
        self._open_windows[viz] = win

        # Feed it the latest data immediately
        try:
            viz.update(self.watcher.vis_data)
        except Exception as e:
            logger.exception(f"Initial update failed for '{viz.name}': {e}")

        win.show()

    def _on_viz_window_closed(self, viz):
        self._open_windows.pop(viz, None)


# ------------------------------
# Entrypoint
# ------------------------------

def main():
    app = QApplication(sys.argv)

    # Create and show the main window.
    main_window = MainWindow()

    main_window.show()

    result = app.exec_()
    sys.exit(result)


if __name__ == "__main__":
    main()
