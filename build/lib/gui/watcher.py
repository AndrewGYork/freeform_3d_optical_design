from qtpy.QtCore import QObject, QTimer, Signal
from gui.visdata import VisData
from typing import Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DataWatcher(QObject):
    data_updated = Signal(object)  # emits the VisData instance

    def __init__(self, base_dir: Path, interval: float, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.vis_data = VisData(base_dir)
        self.timer = QTimer(self)
        self.timer.setInterval(interval)
        self.timer.timeout.connect(self._poll)

    def start(self):
        if not self.vis_data.base_dir.exists():
            logger.warning(f"Output directory does not exist yet: {self.vis_data.base_dir}")
        self.vis_data.initial_load()
        self.timer.start()
        self.data_updated.emit(self.vis_data)

    def stop(self):
        self.timer.stop()

    def set_live_updates(self, enabled: bool):
        if enabled:
            if not self.timer.isActive():
                self.timer.start()
        else:
            if self.timer.isActive():
                self.timer.stop()

    def force_emit(self):
        self.data_updated.emit(self.vis_data)

    def _poll(self):
        try:
            if self.vis_data.reload_if_changed():
                self.data_updated.emit(self.vis_data)
        except Exception as e:
            logger.exception(f"Error while polling data: {e}")
