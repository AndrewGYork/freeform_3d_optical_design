import numpy as np
from train_multiscale.beam_propagation_multiscale import Coordinates, BeamPropagation, gaussian_beam_2d
from train_multiscale.beam_propagation_multiscale import to_tif, from_tif, plot_loss_history, smooth

"""
Design a 3D refractive optic that mimics the input/output behavior of a
pair of ideal lenses which are cofocal but not coaxial.

What is the input-output behavior of a pair of ideal lenses which are
co-focal but not co-axial?

Short answer: orthographically project your input image onto a sphere,
and project that sphere back onto a (tilted) plane.

Skip to the end of this file for the long answer.
"""

def main():
    # Specify our coordinate system, organized via a Coordinates object:
    coords = Coordinates(xyz_i=(-10, -10,   0),
                         xyz_f=(+10, +10, +20),
                         n_xyz=(101, 101, 101))
    print("Voxel dimensions:", coords.d_xyz)

    # Use these coordinates to initialize an instance of BeamPropagation
    # that will simulate how light changes as it passes through our
    # refractive object:
    bp = BeamPropagation(coords)

    # Initialize our object.
    try: # If there's an object saved to disk, pick up where we left off:
        initial_density = from_tif('1_density.tif')
    except FileNotFoundError:
        # If not, initialize with a simple GRIN fiber:
        print("Using default initial density.")
        nx, ny, nz = coords.n_xyz
        x,   y,  z = coords.xyz
        r = np.sqrt(x**2 + y**2)
        initial_density = np.zeros((nz, ny, nx))
        initial_density[:, :, :] = 0.45 / (np.cosh(0.2*r)**2)
    bp.set_3d_density(initial_density)

    # Make a source to generate training data.
    data_source = TrainingData_for_Tilted_Telescope(
        coords, radius=4, tilt_degrees=20)

    wavelength = 1
    divergence_angle_degrees = 15
    loss_history = []
    for iteration in range(10000):
        # Use our data source to generate random input/output pairs:
        x0, y0 = data_source.random_point_in_a_circle(radius=2)
        if iteration == 0: x0, y0 = 0, 0
        input_field, desired_field = data_source.input_output_pair(
            x0, y0, wavelength, divergence_angle_degrees)
        bp.set_2d_input_field(input_field, wavelength)
        bp.set_2d_desired_output_field(desired_field)

        # Simulate propagation through our 3D refractive object,
        # calculate loss, and calculate a gradient that hopefully will
        # reduce the loss:
        bp.calculate_3d_field()
        bp.calculate_loss(z_planes=(1, 2, 3))
        bp.calculate_gradient()

        # Output some intermediate state, so we can monitor our progress:
        print("At iteration", iteration, "the loss is", bp.loss)
        loss_history.append((x0, y0, bp.loss))
        if iteration % 10 == 0:
            print("Saving TIFs etc...", end='')
            to_tif('1_density.tif', bp.density)
            to_tif('2_input_field.tif', bp.input_field)
            to_tif('3_desired_field.tif', bp.desired_field)
            to_tif('4_calculated_field.tif', np.abs(bp.calculated_field))
            to_tif('5_error.tif', bp.error_3d)
            to_tif('6_gradient.tif', bp.gradient)
            plot_loss_history(loss_history, '7_loss_history.png')
            print("done.")

        # Update our 3D refractive object, using our calculated gradient:
        step_size = 1
        update = step_size * smooth(bp.gradient)
        bp.set_3d_density(bp.density - update)

class TrainingData_for_Tilted_Telescope:
    """2d imaging by an ideal lens pair which are cofocal but not coaxial.

    This generates input/output pairs that image a pointlike source at
    the 2d input plane to a pointlike (but shifted and distorted) image
    at the output plane.
    """
    def __init__(self, coordinates, radius, tilt_degrees):
        assert isinstance(coordinates, Coordinates)
        self.coordinates = coordinates
        assert radius > 0
        self.radius = radius
        self.tilt_degrees = float(tilt_degrees)
        self.tilt = np.deg2rad(tilt_degrees)
        return None

    def random_point_in_a_circle(self, radius):
        # Local nicknames:
        R, sin, cos, pi, sqrt = radius, np.sin, np.cos, np.pi, np.sqrt
        rand = np.random.random_sample
        # Simple math:
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
        divergence_angle, pi = np.deg2rad(divergence_angle_degrees), np.pi
        w = wavelength / (pi*divergence_angle)
        # Input beam is a focused point:
        x, y, _ = self.coordinates.xyz
        input_field = gaussian_beam_2d(
            x=x, y=y, x0=x0, y0=y0, phi=phi, theta=theta,
            wavelength=wavelength, w=w)
        # Desired output beam is a distorted image of the same point...
        old_x, old_y = self._old_xy(x, y)
        amplitude_scaling = self._amplitude_scaling()
        desired_field = amplitude_scaling * gaussian_beam_2d(
            x=old_x, y=old_y, x0=x0, y0=y0, phi=phi, theta=theta,
            wavelength=wavelength, w=w)
        desired_field = np.nan_to_num(desired_field)
        return input_field, desired_field

    def _old_xy(self, x, y):
        """Map xy-coordinates in the output plane to the input plane.
        """
        R, sin_t, cos_t = self.radius, np.sin(self.tilt), np.cos(self.tilt)
        # Orthographically project onto a sphere:
        with np.errstate(invalid='ignore'):
            z = np.sqrt(R**2 - x**2 - y**2)
        # Rotate by 'tilt' radians about the x-axis:
        new_z = y*sin_t + z*cos_t
        new_y = -y*cos_t + z*sin_t
        new_x = np.broadcast_to(-x, new_y.shape).copy()
        # Clip the points that were outside of the output circle, and
        # also points that rotate to the "dark side of the moon":
        nan_me = ~(new_z >= 0) # Accounts for negative z, but also NaN z.
        new_x[nan_me] = np.nan
        new_y[nan_me] = np.nan
        # To orthographically project into the input plane, just discard z:
        return new_x, new_y

    def _old_pixel_size(self):
        """How much did the distortion change the size of each pixel?

        The output plane is stored on a grid of square pixels, but these
        pixels don't originate from square regions in the input plane.
        """
        x,  y,  _ = self.coordinates.xyz
        dx, dy, _ = self.coordinates.d_xyz
        # Calculate the old (central) pixel widths and heights:
        old_dx = (self._old_xy(x+dx/2, y     )[0] -
                  self._old_xy(x-dx/2, y     )[0])
        old_dy = (self._old_xy(x,      y+dy/2)[1] -
                  self._old_xy(x,      y-dy/2)[1])
        return old_dx, old_dy

    def _amplitude_scaling(self):
        """
        A square output pixel of size (dx, dy) will not (in general)
        correspond to a square input pixel of the same area, so we have
        to compute the ratio of the sizes to conserve energy.
        """
        dx, dy, _ = self.coordinates.d_xyz
        old_dx, old_dy = self._old_pixel_size()
        output_pixel_area = dx*dy
        input_pixel_area = np.abs(old_dx*old_dy)
        # Assuming the intensity is ~uniform over a single pixel, the
        # product of intensity and area should be the same for an input
        # region and its corresponding output pixel:
        #
        # I_out * output_pixel_area = I_in * input_pixel_area
        with np.errstate(divide='ignore'):
            intensity_ratio = input_pixel_area / output_pixel_area
        # intensity = amplitude^2:
        amplitude_ratio = np.sqrt(intensity_ratio)
        return amplitude_ratio.squeeze()

"""
What is the input-output behavior of a pair of ideal lenses which are
co-focal but not co-axial?

Long anwswer:

***************************************************************
* Points in the input plane map to points at the output plane *
***************************************************************

Two simple facts:
1. The 1st lens turns point foci in its 1st focal plane into plane waves.
2. The 2nd lens turns plane waves into point foci in its 2nd focal plane.

So, the mapping from the 1st focal plane of the 1st lens to the 2nd
focal plane of the 2nd lens is point-to-point. It's just photon
reassignment!

*************************************************************************
* To find the position of an output point, project the corresponding    *
* input point onto a sphere, then project that sphere onto the (tilted) *
* output plane.                                                         *
*************************************************************************

Which points in the first focal plane correspond to which points in the
last focal plane?

Three simple facts:

1. The first lens turns point foci in its first focal plane into plane
   waves. The lateral components of the input point's position are
   proportional to the lateral component of the output plane wave's
   k-vector:

(x_i, y_i) = (-f_1*k_x_o, -f_1*k_y_o)
    ...where:
    *(x_i, y_i) is the input point's position in the first focal plane.
    * (k_x_o, k_y_o) are the lateral components of the output plane
      wave's k-vector.
    * f_1 is the focal length of the first lens.

2. Because the lenses are not coaxial, we have to apply a rotation to the
   k-vector of the plane waves output by the first lens, to get the
   k-vectors of the plane waves input to the second lens, in the second
   lens' coordinate system. Let's suppose the z-direction is the axis of
   the first lens, and the rotation is about the x-axis:

k_z^2 = k^2 - k_x^2 - k_y^2
k_x' = k_x
k_y' = k_y*cos(t) - k_z*sin(t)
k_z' = k_y*sin(t) + k_z*cos(t)

3. The second lens turns plane waves into point foci in its second focal
   plane. The lateral components of the output point's position are
   proportional to the lateral component of the input plane wave's
   k-vector:

(-f_2*k_x_i, -f_2*k_y_i) = (x_o, y_o)
    ...where:
    * (k_x_i, k_y_i) are the lateral components of the input plane
      wave's k-vector.
    *(x_o, y_o) is the output point's position in the last focal plane.
    * f_2 is the focal length of the second lens.

This might seem like a lot of math, but the geometrical interpretation
is really simple. Project your input image onto a sphere, and project
that sphere back onto a (tilted) plane.
"""

if __name__ == '__main__':
    main()
