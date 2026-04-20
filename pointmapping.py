import numpy as np
from typing import Callable
from beam_propagation import gaussian_beam_2d

from scipy.interpolate import RegularGridInterpolator

def compute_output_plane(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    input_plane: np.ndarray,
    map_reverse: Callable[[float, float], tuple[float, float]],
) -> np.ndarray:
    x_grid = x_grid.flatten()
    y_grid = y_grid.flatten()

    interp_real = RegularGridInterpolator(
        (y_grid, x_grid), input_plane.real, bounds_error=False, fill_value=0.0
    )
    interp_imag = RegularGridInterpolator(
        (y_grid, x_grid), input_plane.imag, bounds_error=False, fill_value=0.0
    )

    # Vectorized evaluation: build meshgrid of (x',y')
    Xp, Yp = np.meshgrid(x_grid, y_grid, indexing="xy")
    coords_out = np.stack([Xp.ravel(), Yp.ravel()], axis=-1)

    # Apply reverse mapping to each (x',y')
    mapped = np.array([map_reverse(xp, yp) for xp, yp in coords_out])
    Y_in, X_in = mapped[:,1], mapped[:,0]  # order = (y, x)

    pts = np.stack([Y_in, X_in], axis=-1)

    Er = interp_real(pts)
    Ei = interp_imag(pts)

    output_plane = (Er + 1j*Ei).reshape(len(y_grid), len(x_grid))
    return output_plane


class MyTrainingData_withMapping:
    """Training data generator for 2D imaging with custom reverse mapping.

    Generates input/output field pairs:
      - Input: Gaussian beam centered at (x0,y0).
      - Output: The input field remapped through `map_reverse`.
    """

    def __init__(self, coordinates, radius, map_reverse: Callable[[float,float],tuple[float,float]]):
        assert radius > 0
        self.coordinates = coordinates
        self.radius = radius
        self.map_reverse = map_reverse
        return None

    def random_point_in_a_circle(self):
        R, sin, cos, pi, sqrt = self.radius, np.sin, np.cos, np.pi, np.sqrt
        rand = np.random.random_sample
        r, phi = R*sqrt(rand()), 2*pi*rand()
        x, y = r*cos(phi), r*sin(phi)
        return x, y

    def input_output_pair(
        self,
        x0,
        y0,
        wavelength,
        divergence_angle_degrees,
        phi=0,
        theta=0,
    ):
        x0, y0 = float(x0), float(y0)
        wavelength = float(wavelength)
        divergence_angle = np.deg2rad(divergence_angle_degrees)
        w = wavelength / (np.pi * divergence_angle)

        # Local grids
        x, y, _ = self.coordinates.xyz

        # Generate input Gaussian beam
        input_field = gaussian_beam_2d(
            x=x, y=y, x0=x0, y0=y0,
            phi=phi, theta=theta,
            wavelength=wavelength, w=w,
        )

        # Output field: remap with user-provided mapping
        desired_output_field = compute_output_plane(
            x_grid=self.coordinates.x,
            y_grid=self.coordinates.y,
            input_plane=input_field,
            map_reverse=self.map_reverse,
        )

        return input_field, MyTrainingData_withMapping.normalize_power(input_field, desired_output_field)
    
    @staticmethod
    def total_power(field: np.ndarray) -> float:
        """
        Approximate total power through an image (arbitrary units).
        """

        # Intensity is |E|^2
        intensity = np.abs(field)**2

        return intensity.sum()

    @staticmethod
    def normalize_power(input_field: np.ndarray, output_field: np.ndarray) -> np.ndarray:
        power_input = MyTrainingData_withMapping.total_power(input_field)
        power_output = MyTrainingData_withMapping.total_power(output_field)
        power_ratio = power_input / power_output

        return output_field * np.sqrt(power_ratio)
