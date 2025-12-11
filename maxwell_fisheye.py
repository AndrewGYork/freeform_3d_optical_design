import time
import numpy as np
import torch
from ray_propagation import (
    Coordinates, Refractive3dOptic, FixedIndexMaterial, RayBundle,
    TrainingData_for_2dImaging, from_tif, to_tif, plot_loss_history)

def example_of_usage():
    """Example code: design a 3D refractive optic with specified input/output.

    Consider copy-pasting this example code to get you started.

    In this example, the input/output is simple plane-to-plane imaging
    (with inversion). This is the same input-output you'd expect from a
    pair of ideal lenses which are cofocal and coaxial.

##    We start with some (suboptimal) 3D refractive optic, and we generate
##    "training data": 2D arrays of complex numbers that represent the
##    amplitude and phase of optical inputs to our 3D optic. For each
##    input, we specify the output that we WISH our optic would deliver,
##    and then calculate the output it ACTUALLY delivers, for our current
##    3D refractive optic. We use the difference between desired and
##    calculated output to calculate our "loss", and use gradients of this
##    loss to update our 3D refractive optic.
    """

    # Specify our coordinate system, organized via a Coordinates object:
    coords = Coordinates(xyz_i=( -10, -10, -10),
                         xyz_f=( +10, +10, +10),
                         n_xyz=( 205, 203, 201))
    print("Voxel dimensions: %0.3f, %0.3f, %0.3f"%(coords.d_xyz))

    # Use these coordinates to initialize an instance of Refractive3dOptic
    # that will simulate how light changes as it passes through our
    # refractive optic:
    ro = Refractive3dOptic(coords)

    # Each voxel of our refractive optic is a mixture of materials:
    air     = FixedIndexMaterial(1)
    polymer = FixedIndexMaterial(2)
    ro.set_materials((air, polymer))

    # Initialize our optic.
    try: # If there's a concentration saved to disk, pick up where we left off:
        fname = '01_concentration.tif'
        initial_concentration = from_tif(fname)
        print("Using initial concentration from:", fname)
    except FileNotFoundError:
        print("Initializing refractive object to Maxwell's Fisheye")
        def maxwell_fisheye_index(coordinates, n0=2):
            x, y, z = coordinates.xyz
            r_sq = x**2 + y**2 + z**2
            R_sq = (coordinates.z_i)**2
            index = n0 / (1 + r_sq/R_sq)
            index[r_sq > R_sq] = n0/2
            return index
        initial_index = maxwell_fisheye_index(coords)
        n0, n1 = air.get_index(0.5), polymer.get_index(0.5)
        initial_concentration = (initial_index - n0) / (n1 - n0)
    ro.set_3d_concentration(initial_concentration)

##    # Make a source to generate training data. In this case, the
##    # training data is for a simple plane-to-plane inverting imaging
##    # system:
##    data_source = TrainingData_for_2dImaging(coords, radius=3)
##
    wavelength = 1
    divergence_angle_degrees = 15
##    loss_history = []
##    for iteration in range(int(1e6)): # Run for a loooong time
    start_time = time.perf_counter()
        
##        # Use our data source to generate random input/output pairs:
##        x0, y0 = data_source.random_point_in_a_circle()
##        input_field, desired_output_field = data_source.input_output_pair(
##            x0, y0, wavelength, divergence_angle_degrees)
    input_raybundle = ray_fan(coords, divergence_angle_degrees)
    ro.set_input_raybundle(input_raybundle)
    desired_output_raybundle = ray_fan(coords, divergence_angle_degrees)
    desired_output_raybundle.xyz[    2, :] = coords.z_f
    desired_output_raybundle.xyz[  0:2, :] *= -1
    desired_output_raybundle.v_xyz[0:2, :] *= -1
    ro.set_desired_output_raybundle(desired_output_raybundle)

    # Simulate propagation through our 3D refractive optic:
    start = time.perf_counter()
    rbs = ro._calculate_3d_propagation()
    end = time.perf_counter()
    print(end - start)

##    xyz_vs_z = torch.stack([rb.xyz.detach() for rb in rbs],
##                           dim=2)
##    import matplotlib.pyplot as plt
##    plt.figure()
##    for which_ray in range(xyz_vs_z.shape[1]):
##        plt.plot(xyz_vs_z[2, which_ray, :],
##                 xyz_vs_z[0, which_ray, :], '.-')
##    plt.grid('on')
##    plt.xlim(9.9, 10.1)
##    plt.ylim(-0.03, 0.03)
##    plt.show()

    xyz_error = rbs[-1].xyz - ro.desired_output_raybundle.xyz
    print(xyz_error)
    loss = (xyz_error[:, 0:2]**2).sum()
    start = time.perf_counter()
    loss.backward()
    end = time.perf_counter()
    print(end - start, 's')
    print(loss)
    to_tif('gradient.tif', ro._composition_tensor.grad)

##    x = np.array([rb.x.numpy() for rb in rbs])
##    z = np.array([rb.z.numpy() for rb in rbs])
##    import matplotlib.pyplot as plt
##    for which_ray in range(x.shape[1]):
##        plt.plot(z[:, which_ray], x[:, which_ray])
##    plt.grid('on')
##    plt.show()


##    acceleration_on_grid = np.stack((ro._to_numpy(ro._x_acceleration),
##                                     ro._to_numpy(ro._y_acceleration),
##                                     ro._to_numpy(ro._z_acceleration)), axis=0)
##    to_tif('acceleration.tif', acceleration_on_grid)
##    x = (1*coords.x + 0*coords.y + 0*coords.z).ravel() * (1-1e-12)
##    y = (0*coords.x + 1*coords.y + 0*coords.z).ravel() * (1-1e-12)
##    z = (0*coords.x + 0*coords.y + 1*coords.z).ravel() * (1-1e-12)
##    x, y, z = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(z)
##    a_xyz = ro._get_acceleration_at_points(x, y, z).reshape(3,
##                                                            coords.nz,
##                                                            coords.ny,
##                                                            coords.nx)
##    to_tif('acceleration_interpolated.tif', a_xyz)
##    error = a_xyz.numpy() - acceleration_on_grid
##    print(error.min(), error.max())

##        # Simulate propagation through our 3D refractive optic,
##        # calculate loss, and calculate a gradient that hopefully will
##        # reduce the loss:
##        ro.gradient_update(
##            step_size=100,
##            z_planes=(1, 2, 3),
##            smoothing_sigma=5)
##        loss_history.append((x0, y0, ro.loss))
##
##        end_time = time.perf_counter()
##        print("At iteration", iteration, "the loss is %0.4f"%(ro.loss),
##              "(%0.2f ms elapsed)"%(1000*(end_time - start_time)))
##
##        # Every so often, output some intermediate state, so we can
##        # monitor our progress. You can use ImageJ
##        # ( https://imagej.net/ij/ ) to view the TIF files:
##        if iteration % 50 == 0:
##            ro.update_attributes()
##            print("Saving TIFs etc...", end='')
##            to_tif('00_composition.tif',          ro.composition)
##            to_tif('01_concentration.tif',        ro.concentration)
##            to_tif('02_concentration_xz.tif',
##                   ro.concentration[:, ro.coordinates.ny//2, :])
##            to_tif('03_input_field.tif',          ro.input_field)
##            to_tif('04_desired_output_field.tif', ro.desired_output_field)
##            to_tif('05_calculated_field.tif',
##                   np.abs(ro.calculated_field))
##            to_tif('06_desired_output_field_3d',
##                   np.abs(ro.desired_output_field_3d))
##            to_tif('07_calculated_output_field_3d',
##                   np.abs(ro.calculated_output_field_3d))
##            to_tif('08_error_3d.tif', ro.error_3d)
##            to_tif('09_gradient.tif', ro.gradient)
##            plot_loss_history(loss_history, '10_loss_history.png')
##            print("done.")

def ray_fan(coordinates, divergence_angle_degrees):
    n_rays = 20000
    theta_radians = np.deg2rad(np.linspace(-divergence_angle_degrees,
                                            divergence_angle_degrees, n_rays))
    vx = np.sin(theta_radians)
    vy = np.zeros((n_rays,))
    vz = np.cos(theta_radians)
    x = np.zeros((n_rays,))
    y = np.zeros((n_rays,))
    z = np.ones((n_rays,))*coordinates.z_i
    xyz = np.stack((x, y, z), axis=0)
    v_xyz = np.stack((vx, vy, vz), axis=0)
    return RayBundle(xyz, v_xyz, wavelength_um=0.5)
    
if __name__ == '__main__':
    example_of_usage()
