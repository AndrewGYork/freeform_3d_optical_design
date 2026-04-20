from typing import Optional

import numpy as np

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QWidget,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QSizePolicy,
    QSlider,
    QGroupBox,
)
from qtpy.QtGui import QPixmap, QImage

from gui.visdata import VisData
from gui.visualization.visualization import Visualization
from gui.visualization.scaled_label import ScaledLabel


class VolumeOrthoSlicesVisualization(Visualization):
    """Viewer for orthogonal planes (XY, XZ, YZ) intersecting a point (x, y, z)."""

    def __init__(
        self,
        key: str,
        percentile_clip: tuple[float, float] = (2.0, 98.0),
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._key = key
        self._percentiles = percentile_clip

        # Cached volume and shape
        self._vol: Optional[np.ndarray] = None
        self._shape: Optional[tuple[int, int, int]] = None

        # Normalization bounds
        self._vmin: Optional[float] = None
        self._vmax: Optional[float] = None

        # Current indices
        self._ix: int = 0
        self._iy: int = 0
        self._iz: int = 0

        # --- UI ---
        root = QVBoxLayout(self)

        # Images group (side by side)
        imgs_group = QGroupBox("Orthogonal Slices")
        imgs_layout = QHBoxLayout()
        imgs_group.setLayout(imgs_layout)

        self._lbl_xy_title = QLabel("XY (z=0)")
        self._img_xy = ScaledLabel("No data yet.")
        # self._setup_img_label(self._img_xy)

        self._lbl_xz_title = QLabel("XZ (y=0)")
        self._img_xz = ScaledLabel("No data yet.")
        # self._setup_img_label(self._img_xz)

        self._lbl_yz_title = QLabel("YZ (x=0)")
        self._img_yz = ScaledLabel("No data yet.")
        # self._setup_img_label(self._img_yz)

        imgs_layout.addLayout(self._column(self._lbl_xy_title, self._img_xy))
        imgs_layout.addLayout(self._column(self._lbl_xz_title, self._img_xz))
        imgs_layout.addLayout(self._column(self._lbl_yz_title, self._img_yz))

        root.addWidget(imgs_group)

        # Sliders group (vertical stack instead of grid)
        sliders_group = QGroupBox("Position (x, y, z)")
        sliders_layout = QVBoxLayout()
        sliders_group.setLayout(sliders_layout)

        self._lab_x = QLabel("x: 0")
        self._sld_x = QSlider(Qt.Horizontal)
        self._sld_x.setRange(0, 0)
        self._sld_x.valueChanged.connect(self._on_slider_x)
        sliders_layout.addWidget(self._lab_x)
        sliders_layout.addWidget(self._sld_x)

        self._lab_y = QLabel("y: 0")
        self._sld_y = QSlider(Qt.Horizontal)
        self._sld_y.setRange(0, 0)
        self._sld_y.valueChanged.connect(self._on_slider_y)
        sliders_layout.addWidget(self._lab_y)
        sliders_layout.addWidget(self._sld_y)

        self._lab_z = QLabel("z: 0")
        self._sld_z = QSlider(Qt.Horizontal)
        self._sld_z.setRange(0, 0)
        self._sld_z.valueChanged.connect(self._on_slider_z)
        sliders_layout.addWidget(self._lab_z)
        sliders_layout.addWidget(self._sld_z)

        root.addWidget(sliders_group)

    def _column(self, title_label: QLabel, image_label: QLabel) -> QVBoxLayout:
        col = QVBoxLayout()
        col.addWidget(title_label, 0, Qt.AlignHCenter)  # only center the title
        col.addWidget(image_label)                      # <-- no alignment, allow stretch
        col.setStretch(0, 0)                            # title doesn't stretch
        col.setStretch(1, 1)                            # image takes all extra vertical space
        return col

    def update(self, data: VisData):
        vol = getattr(data, self._key, None)
        if vol is None:
            self._show_missing()
            return
        if np.iscomplexobj(vol):
            vol = np.abs(vol)
        if vol.ndim != 3:
            self._show_invalid_dim(vol.ndim)
            return
        vol = np.asarray(vol, dtype=np.float64)
        shape = tuple(int(s) for s in vol.shape)
        if self._shape != shape or self._vol is None:
            self._vol = vol
            self._shape = shape
            self._compute_normalization(vol)
            self._reset_sliders(shape)
        else:
            self._vol = vol
            self._compute_normalization(vol)
            z, y, x = shape
            self._ix = min(self._ix, x - 1)
            self._iy = min(self._iy, y - 1)
            self._iz = min(self._iz, z - 1)
            self._apply_indices_to_sliders()
        self._render_all()

    def _show_missing(self):
        msg = f"{self._key} not found."
        self._img_xy.setText(msg)
        self._img_xz.setText(msg)
        self._img_yz.setText(msg)

    def _show_invalid_dim(self, ndim: int):
        msg = f"{self._key} must be 3D, got ndim={ndim}."
        self._img_xy.setText(msg)
        self._img_xz.setText(msg)
        self._img_yz.setText(msg)

    def _compute_normalization(self, vol: np.ndarray):
        finite = vol[np.isfinite(vol)]
        if finite.size == 0:
            self._vmin, self._vmax = 0.0, 1.0
            return
        lo, hi = self._percentiles
        try:
            vmin = float(np.percentile(finite, lo))
            vmax = float(np.percentile(finite, hi))
        except Exception:
            vmin = float(np.min(finite))
            vmax = float(np.max(finite))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = 0.0, 1.0
        self._vmin, self._vmax = vmin, vmax

    def _reset_sliders(self, shape: tuple[int, int, int]):
        z, y, x = shape
        self._sld_x.blockSignals(True)
        self._sld_y.blockSignals(True)
        self._sld_z.blockSignals(True)
        self._sld_x.setRange(0, max(0, x - 1))
        self._sld_y.setRange(0, max(0, y - 1))
        self._sld_z.setRange(0, max(0, z - 1))
        self._ix = x // 2
        self._iy = y // 2
        self._iz = z // 2
        self._apply_indices_to_sliders()
        self._sld_x.blockSignals(False)
        self._sld_y.blockSignals(False)
        self._sld_z.blockSignals(False)

    def _apply_indices_to_sliders(self):
        self._sld_x.blockSignals(True)
        self._sld_y.blockSignals(True)
        self._sld_z.blockSignals(True)
        self._sld_x.setValue(self._ix)
        self._sld_y.setValue(self._iy)
        self._sld_z.setValue(self._iz)
        self._sld_x.blockSignals(False)
        self._sld_y.blockSignals(False)
        self._sld_z.blockSignals(False)
        self._lab_x.setText(f"x: {self._ix}")
        self._lab_y.setText(f"y: {self._iy}")
        self._lab_z.setText(f"z: {self._iz}")

    def _render_all(self):
        if self._vol is None or self._shape is None:
            return
        z, y, x = self._shape
        self._lbl_xy_title.setText(f"XY (z={self._iz} / {z-1})")
        self._lbl_xz_title.setText(f"XZ (y={self._iy} / {y-1})")
        self._lbl_yz_title.setText(f"YZ (x={self._ix} / {x-1})")
        self._set_image(self._img_xy, self._vol[self._iz, :, :])
        self._set_image(self._img_xz, self._vol[:, self._iy, :])
        self._set_image(self._img_yz, self._vol[:, :, self._ix])

    def _on_slider_x(self, value: int):
        self._ix = int(value)
        self._lab_x.setText(f"x: {self._ix}")
        if self._vol is None:
            return
        self._set_image(self._img_yz, self._vol[:, :, self._ix])

    def _on_slider_y(self, value: int):
        self._iy = int(value)
        self._lab_y.setText(f"y: {self._iy}")
        if self._vol is None:
            return
        self._set_image(self._img_xz, self._vol[:, self._iy, :])

    def _on_slider_z(self, value: int):
        self._iz = int(value)
        self._lab_z.setText(f"z: {self._iz}")
        if self._vol is None:
            return
        self._set_image(self._img_xy, self._vol[self._iz, :, :])

    def _normalize_to_u8(self, img2d: np.ndarray) -> np.ndarray:
        vmin = self._vmin if self._vmin is not None else float(np.nanmin(img2d))
        vmax = self._vmax if self._vmax is not None else float(np.nanmax(img2d))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = 0.0, 1.0
        x = img2d.astype(np.float64)
        x = np.clip((x - vmin) / (vmax - vmin), 0.0, 1.0)
        x = (x * 255.0).round().astype(np.uint8)
        return x

    def _qimage_from_u8(self, u8: np.ndarray) -> QImage:
        h, w = u8.shape
        buf = np.ascontiguousarray(u8)
        return QImage(buf.data, w, h, w, QImage.Format_Grayscale8)

    def _set_image(self, label: QLabel, img2d: np.ndarray):
        u8 = self._normalize_to_u8(img2d)
        qimg = self._qimage_from_u8(u8)
        pix = QPixmap.fromImage(qimg)
        label.setPixmap(pix)
