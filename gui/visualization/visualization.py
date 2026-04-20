from typing import Optional
from qtpy.QtWidgets import QWidget
from gui.visdata import VisData


class Visualization(QWidget):
    """Abstract Visualization corresponding to a Qt widget.

    Subclasses must define:
    - name: a human-readable name (string)
    - update(self, data: VisData): called whenever new data is available
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

    def update(self, data: VisData):  # noqa: D401 - method intentionally named per spec
        """Receive updates to the underlying VisData. Must be implemented by subclasses."""
        raise NotImplementedError()

    @property
    def name(self) -> str:
        """Return a human readable name for this visualization"""
        return getattr(self, "_human_readable_name", "Unnamed Visualization")
    
    def rename(self, name: str) -> 'Visualization':
        """
        Changes this visualization's human readable displayname, and then returns self
        """
        self._human_readable_name = name
        return self
