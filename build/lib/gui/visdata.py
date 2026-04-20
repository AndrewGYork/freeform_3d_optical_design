import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from qtpy.QtGui import QImage

logger = logging.getLogger(__name__)

# ------------------------------
# Utilities for I/O
# ------------------------------

def _read_tiff(path: Path) -> Optional[np.ndarray]:
    """Read a TIFF file into a numpy array. Tries tifffile, falls back to imageio.
    Returns None if read fails or file missing.
    """
    if not path.exists():
        return None
    try:
        import tifffile as tiff

        arr = tiff.imread(str(path))
        return np.asarray(arr)
    except Exception as e1:
        logger.debug(f"tifffile failed for {path}: {e1}")
        try:
            import imageio.v2 as iio

            arr = iio.imread(str(path))
            return np.asarray(arr)
        except Exception as e2:
            logger.warning(f"Failed to read TIFF {path}: {e2}")
            return None


def _read_png_qimage(path: Path) -> Optional[QImage]:
    """Load a PNG as a QImage for display. Returns None if missing or unreadable."""
    if not path.exists():
        return None
    img = QImage(str(path))
    if img.isNull():
        logger.warning(f"Failed to read PNG {path}")
        return None
    return img


def _amp_phase_to_complex(amp: Optional[np.ndarray], phase: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if amp is None or phase is None:
        return None
    try:
        return amp.astype(np.complex128) * np.exp(1j * phase.astype(np.float64))
    except Exception as e:
        logger.warning(f"Failed to convert amplitude/phase to complex: {e}")
        return None

@dataclass
class FileStamp:
    mtime_ns: int
    size: int

class VisData:
    """Loads and caches data from the hardcoded 'output' directory.

    - All TIFFs are loaded to numpy arrays when present.
    - Amplitude/phase pairs are converted to single complex-valued numpy arrays.
    - loss_history.png is exposed as a QImage for convenience.
    - Uses file modification stamps to detect changes and only reloads when needed.
    """

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        logger.info(f"Searching for output files in {base_dir.resolve()}")
        # Known file names
        self.paths: dict[str, Path] = {
            # Scalars / volumes
            "composition": base_dir / "composition.tif",
            "concentration": base_dir / "01_concentration.tif",
            # Input field (2D)
            "input_field_amplitude": base_dir / "input_field_amplitude.tif",
            "input_field_phase": base_dir / "input_field_phase.tif",
            # Desired output field (2D)
            "desired_output_field_amplitude": base_dir / "desired_output_field_amplitude.tif",
            "desired_output_field_phase": base_dir / "desired_output_field_phase.tif",
            # Calculated field (3D)
            "calculated_field_amplitude": base_dir / "calculated_field_amplitude.tif",
            "calculated_field_phase": base_dir / "calculated_field_phase.tif",
            # Errors (3D)
            "error_3d_amplitude": base_dir / "error_3d_amplitude.tif",
            "error_3d_phase": base_dir / "error_3d_phase.tif",
            # Gradient (3D)
            "gradient_amplitude": base_dir / "gradient_amplitude.tif",
            "gradient_phase": base_dir / "gradient_phase.tif",
            # PNG
            "loss_history": base_dir / "10_loss_history.png",
        }

        # Data containers
        self.tensors: dict[str, Optional[np.ndarray]] = {k: None for k in self.paths if k.endswith(".tif") is False}
        # Note: We'll store TIFF arrays under explicit keys below instead of relying on the dict above
        self.composition: Optional[np.ndarray] = None
        self.concentration: Optional[np.ndarray] = None
        self.input_field_complex: Optional[np.ndarray] = None  # 2D complex
        self.desired_output_field_complex: Optional[np.ndarray] = None  # 2D complex
        self.calculated_field_complex: Optional[np.ndarray] = None  # 3D complex
        self.error_3d_complex: Optional[np.ndarray] = None  # 3D complex
        self.gradient_complex: Optional[np.ndarray] = None  # 3D complex

        self.loss_history_image: Optional[QImage] = None

        # Internal bookkeeping for change detection
        self._stamps: dict[str, FileStamp] = {}
        self.version: int = 0  # increments whenever something reloads

    @staticmethod
    def _stat(path: Path) -> Optional[FileStamp]:
        try:
            st = path.stat()
            return FileStamp(mtime_ns=st.st_mtime_ns, size=st.st_size)
        except FileNotFoundError:
            return None

    def _changed(self, key: str) -> bool:
        path = self.paths[key]
        new = self._stat(path)
        old = self._stamps.get(key)
        if new is None and old is None:
            return False  # still missing
        if new is None and old is not None:
            self._stamps[key] = None  # type: ignore
            return True
        if new is not None and old is None:
            self._stamps[key] = new
            return True
        # both present
        assert new is not None and old is not None
        if new.mtime_ns != old.mtime_ns or new.size != old.size:
            self._stamps[key] = new
            return True
        return False

    def _mark_all_current(self):
        for k, p in self.paths.items():
            self._stamps[k] = self._stat(p)

    def _reload_arrays(self):
        # Simple, safe loader for each expected file
        self.composition = _read_tiff(self.paths["composition"])  # 3D float, -inf..inf
        self.concentration = _read_tiff(self.paths["concentration"])  # 3D float, 0..inf

        in_amp = _read_tiff(self.paths["input_field_amplitude"])  # 2D float, 0..inf
        in_ph = _read_tiff(self.paths["input_field_phase"])  # 2D float, 0..2pi
        self.input_field_complex = _amp_phase_to_complex(in_amp, in_ph)

        des_amp = _read_tiff(self.paths["desired_output_field_amplitude"])  # 2D float, -inf..inf
        des_ph = _read_tiff(self.paths["desired_output_field_phase"])  # 2D float, 0..2pi
        self.desired_output_field_complex = _amp_phase_to_complex(des_amp, des_ph)

        calc_amp = _read_tiff(self.paths["calculated_field_amplitude"])  # 3D float, -inf..inf
        calc_ph = _read_tiff(self.paths["calculated_field_phase"])  # 3D float, 0..2pi
        self.calculated_field_complex = _amp_phase_to_complex(calc_amp, calc_ph)

        err_amp = _read_tiff(self.paths["error_3d_amplitude"])  # 3D float, -inf..inf
        err_ph = _read_tiff(self.paths["error_3d_phase"])  # 3D float, 0..2pi
        self.error_3d_complex = _amp_phase_to_complex(err_amp, err_ph)

        grad_amp = _read_tiff(self.paths["gradient_amplitude"])  # 3D float, -inf..inf
        grad_ph = _read_tiff(self.paths["gradient_phase"])  # 3D float, 0..2pi
        self.gradient_complex = _amp_phase_to_complex(grad_amp, grad_ph)

        # PNG
        self.loss_history_image = _read_png_qimage(self.paths["loss_history"])  # may be None

    def initial_load(self) -> None:
        """Load everything without change checks (used on startup)."""
        self._reload_arrays()
        self._mark_all_current()
        self.version += 1
        logger.info("Initial data load complete (version %d).", self.version)

    def reload_if_changed(self) -> bool:
        """Check for any file changes and reload what is needed.

        For simplicity and consistency across derived visualizations, we reload the whole set if any
        tracked file changed. Returns True if anything changed.
        """
        changed_any = False
        for key in self.paths:
            if self._changed(key):
                changed_any = True
        if changed_any:
            logger.info("Detected changes in 'output' — reloading data…")
            self._reload_arrays()
            self.version += 1
            logger.info("Reload complete (version %d).", self.version)
        return changed_any
