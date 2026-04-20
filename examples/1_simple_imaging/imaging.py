import numpy as np
from train_multiscale.beam_propagation_multiscale import Coordinates, BeamPropagation
from train_multiscale.beam_propagation_multiscale import TrainingData_for_2dImaging
from train_multiscale.beam_propagation_multiscale import to_tif, from_tif, plot_loss_history, smooth

"""
Design a 3D refractive optic that mimics the input/output behavior of an
ideal imaging system.
"""

def main():
    """Example code: design a 3D refractive optic with specified input/output.

    In this example, the input/output is simple plane-to-plane imaging
    (with inversion). This is the same input-output you'd expect from a
    pair of ideal lenses which are cofocal and coaxial.

    We start with some (suboptimal) 3D refractive optic, and we generate
    "training data": 2D arrays of complex numbers that represent the
    amplitude and phase of optical inputs to our 3D optic. For each
    input, we specify the output that we WISH our optic would deliver,
    and then calculate the output it ACTUALLY delivers, for our current
    3D refractive optic. We use the difference between desired and
    calculated output to calculate our "loss", and use gradients of this
    loss to update our 3D refractive optic.
    """
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
        # If not, initialize either an empty array or a simple
        # pseudoparabolic GRIN lens:
        print("Using default initial density.")
        nx, ny, nz = coords.n_xyz
        x,   y,  z = coords.xyz
        r = np.sqrt(x**2 + y**2)
        initial_density = np.zeros((nz, ny, nx))
        # If you want to initialize with a pretty good initial guess,
        # uncomment this line:
        ##initial_density[:, :, :] = 0.45 / (np.cosh(0.2*r)**2)
    bp.set_3d_density(initial_density)

    # Make a source to generate training data. In this case, the
    # training data is for a simple plane-to-plane inverting imaging
    # system:
    data_source = TrainingData_for_2dImaging(coords, radius=3)

    wavelength = 1
    divergence_angle_degrees = 15
    loss_history = []
    for iteration in range(10000):
        # Use our data source to generate random input/output pairs:
        x0, y0 = data_source.random_point_in_a_circle()
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

if __name__ == '__main__':
    main()
