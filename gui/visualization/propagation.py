from typing import Optional, Tuple

import numpy as np

from qtpy.QtCore import Qt, QSignalBlocker
from qtpy.QtWidgets import (
    QWidget,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QHBoxLayout,
    QSlider,
    QGroupBox,
    QCheckBox,
)
from qtpy.QtGui import QPixmap, QImage

from gui.visdata import VisData
from gui.visualization.visualization import Visualization
from gui.visualization.scaled_label import ScaledLabel


class PropagationViewer(Visualization):
    # ... docstring unchanged ...

    def __init__(
        self,
        vol_key: str = "calculated_field_complex",
        desired_key: str = "desired_output_field_complex",
        percentile_clip: Tuple[float, float] = (0, 100),
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._vol_key = vol_key
        self._desired_key = desired_key
        self._percentiles = percentile_clip

        # Cached arrays
        self._vol: Optional[np.ndarray] = None
        self._desired: Optional[np.ndarray] = None
        self._shape3: Optional[tuple[int, int, int]] = None
        self._shape2: Optional[tuple[int, int]] = None

        self._iz: int = 0
        self._show_red: bool = True

        # --- UI ---
        root = QVBoxLayout(self)

        img_group = QGroupBox("Light Propagation (Red: desired, Green: computed slice)")
        img_vbox = QVBoxLayout()
        img_group.setLayout(img_vbox)

        self._img = ScaledLabel("No data yet.")
        img_vbox.addWidget(self._img, 1)
        root.addWidget(img_group, 1)

        # Slider row
        slider_group = QGroupBox("Depth (z)")
        slider_hbox = QHBoxLayout()
        slider_group.setLayout(slider_hbox)

        self._lab_z = QLabel("z: 0")
        self._sld_z = QSlider(Qt.Horizontal)
        self._sld_z.setRange(0, 0)
        self._sld_z.valueChanged.connect(self._on_slider_z)

        self._chk_red = QCheckBox("Show desired (red)")  # NEW
        self._chk_red.setChecked(True)
        self._chk_red.toggled.connect(self._on_toggle_red)

        slider_hbox.addWidget(self._lab_z)
        slider_hbox.addWidget(self._sld_z, 1)
        slider_hbox.addWidget(self._chk_red)

        root.addWidget(slider_group)

        if hasattr(self, "rename"):
            self.rename("Light Propagation")

    # ----------------------- Public API -----------------------
    def update(self, data: VisData):
        vol = getattr(data, self._vol_key, None)
        desired = getattr(data, self._desired_key, None)

        if vol is None or desired is None:
            self._show_missing(vol is None, desired is None)
            # Optional: disable checkbox if desired missing
            if desired is None:
                self._chk_red.setEnabled(False)
            else:
                self._chk_red.setEnabled(True)
            return

        # Convert complex -> magnitude
        if np.iscomplexobj(vol):
            vol = np.abs(vol)
        if np.iscomplexobj(desired):
            desired = np.abs(desired)

        if vol.ndim != 3 or desired.ndim != 2:
            self._img.setText(
                f"Shapes must be 3D and 2D, got vol.ndim={vol.ndim}, desired.ndim={desired.ndim}"
            )
            return

        vol = np.asarray(vol, dtype=np.float64)
        desired = np.asarray(desired, dtype=np.float64)

        shape3 = tuple(int(s) for s in vol.shape)  # (Z, Y, X)
        shape2 = tuple(int(s) for s in desired.shape)  # (Y, X)

        if shape3[1:] != shape2:
            self._img.setText(
                f"XY size mismatch: volume XY={shape3[1:]} vs desired={shape2}."
            )
            self._maybe_reset_slider(shape3[0])
            return

        self._vol = vol
        self._desired = desired
        self._shape3 = shape3
        self._shape2 = shape2

        self._chk_red.setEnabled(True)

        self._maybe_reset_slider(shape3[0])
        self._render()

    # ----------------------- Internals -----------------------
    def _show_missing(self, vol_missing: bool, desired_missing: bool):
        if vol_missing and desired_missing:
            self._img.setText("Computed volume and desired output not found.")
        elif vol_missing:
            self._img.setText("Computed volume not found.")
        else:
            self._img.setText("Desired output not found.")

    def _maybe_reset_slider(self, zdim: int):
        new_max = max(0, zdim - 1)
        with QSignalBlocker(self._sld_z):
            self._sld_z.setRange(0, new_max)
            self._iz = min(self._iz, new_max) if self._iz <= new_max else new_max
            if zdim > 0 and self._iz == 0:
                self._iz = zdim // 2
            self._sld_z.setValue(self._iz)
        self._lab_z.setText(f"z: {self._iz}")

    def _on_slider_z(self, value: int):
        self._iz = int(value)
        self._lab_z.setText(f"z: {self._iz}")
        self._render()

    def _on_toggle_red(self, checked: bool):           # NEW
        self._show_red = checked
        self._render()

    def _render(self):
        if self._vol is None or self._desired is None:
            return
        z, y, x = self._shape3
        iz = int(np.clip(self._iz, 0, z - 1))

        # Green: volume slice at z
        xy = self._vol[iz, :, :]
        g = self._normalize_to_u8(xy)

        # Red: desired output (constant w.r.t z), toggled by checkbox
        if self._show_red:                              # NEW
            r = self._normalize_to_u8(self._desired)    # NEW
        else:                                           # NEW
            r = np.zeros_like(self._desired, dtype=np.uint8)  # NEW

        # Compose RGB (R, G, B). Blue stays 0.
        h, w = r.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[..., 0] = r  # Red
        rgb[..., 1] = g  # Green

        qimg = QImage(np.ascontiguousarray(rgb).data, w, h, 3 * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        self._img.setPixmap(pix)

    def _normalize_to_u8(self, arr: np.ndarray) -> np.ndarray:
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return np.zeros_like(arr, dtype=np.uint8)
        lo, hi = self._percentiles
        try:
            vmin = float(np.percentile(finite, lo))
            vmax = float(np.percentile(finite, hi))
        except Exception:
            vmin = float(np.min(finite))
            vmax = float(np.max(finite))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = 0.0, 1.0
        x = arr.astype(np.float64)
        x = np.clip((x - vmin) / (vmax - vmin), 0.0, 1.0)
        return (x * 255.0).round().astype(np.uint8)
