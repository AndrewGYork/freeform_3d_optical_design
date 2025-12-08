import time
import numpy as np
from beam_propagation import (
    Coordinates, Refractive3dOptic, FixedIndexMaterial,
    TrainingData_for_2dImaging, from_tif, to_tif, plot_loss_history, show_ray_diagram, show_ray_bundle)

def example_of_usage(): 
    """Example code: design a 3D refractive optic with specified input/output.

    Consider copy-pasting this example code to get you started.

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
    coords = Coordinates(xyz_i=(-12.7, -12.7,  0    ),
                         xyz_f=(+12.7, +12.7, +25.4 ),
                         n_xyz=(  128,   128,   128 ))
    print("Voxel dimensions: %0.3f, %0.3f, %0.3f"%(coords.d_xyz))

    # Use these coordinates to initialize an instance of Refractive3dOptic
    # that will simulate how light changes as it passes through our
    # refractive optic:
    ro = Refractive3dOptic(coords)

    # Each voxel of our refractive optic is a mixture of materials:
    mat_low = FixedIndexMaterial(0.5)
    mat_high = FixedIndexMaterial(2.5)
    ro.set_materials((mat_low, mat_high))

    # Initialize our optic.
    try: # If there's a concentration saved to disk, pick up where we left off:
        fname = '01_concentration.tif'
        initial_concentration = from_tif(fname)
        ro.set_3d_concentration(initial_concentration)
        print("Using initial concentration from:", fname)
    except FileNotFoundError:
        initial_concentration = maxwell_fisheye(coords, mat_high, mat_low)
        ro.set_3d_concentration(initial_concentration)
        print("Using default concentration (Maxwell's Fisheye).")

    # Make a source to generate training data. In this case, the
    # training data is for a simple plane-to-plane inverting imaging
    # system:
    data_source = TrainingData_for_2dImaging(coords, radius=5)

    wavelength = 1
    divergence_angle_degrees = 20
    loss_history = []
    for iteration in range(int(1e6)): # Run for a loooong time
        start_time = time.perf_counter()
        
        # Use our data source to generate random input/output pairs:
        x0, y0 = data_source.random_point_in_a_circle()
        num_thetas = 40
        num_phis = 40
        input_raybundle, desired_output_raybundle = data_source.input_output_pair(
            x0, y0, wavelength, num_thetas, num_phis, divergence_angle_degrees)
        ro.set_input_raybundle(input_raybundle, wavelength)
        ro.set_desired_output_raybundle(desired_output_raybundle)
        show_ray_bundle(coords, input_raybundle, "input")
        show_ray_bundle(coords, desired_output_raybundle, "desired_output")

        # Simulate propagation through our 3D refractive optic,
        # calculate loss, and calculate a gradient that hopefully will
        # reduce the loss:
        ro.gradient_update(
            step_size=1e-2,
            smoothing_sigma=5)
        loss_history.append((x0, y0, ro.loss))

        end_time = time.perf_counter()
        print("At iteration", iteration, "the loss is %0.4f"%(ro.loss),
              "(%0.2f ms elapsed)"%(1000*(end_time - start_time)))

        # Every so often, output some intermediate state, so we can
        # monitor our progress. You can use ImageJ
        # ( https://imagej.net/ij/ ) to view the TIF files:
        if iteration % 50 == 0:
            ro.update_attributes()
            print("Saving TIFs etc...", end='')
            to_tif('00_composition.tif',          ro.composition)
            to_tif('01_concentration.tif',        ro.concentration)
            to_tif('02_concentration_xz.tif',
                   ro.concentration[:, ro.coordinates.ny//2, :].transpose())
            show_ray_diagram(coords, ro.calculated_raybundle_sequence, '03_raybundle_sequence.png')
            to_tif('04_gradient.tif', ro.gradient)
            plot_loss_history(loss_history, '05_loss_history.png')
            print("done.")

def maxwell_fisheye(coords: Coordinates, high_ri_material: FixedIndexMaterial, low_ri_material: FixedIndexMaterial, center: tuple[float, float, float]=None, radius: float=None):
    """Design a Maxwell fisheye (Luneberg lens) using the specified materials."""
    # Calculate the radius of the lens
    n_high = high_ri_material.index
    n_low = low_ri_material.index
    assert n_high > n_low, "High index material must have a higher index than low index material."
    assert n_high / n_low >= 2, "Index ratio must be at least 2."
    
    Dx = coords.x_f - coords.x_i + coords.dx
    Dy = coords.y_f - coords.y_i + coords.dy
    Dz = coords.z_f - coords.z_i + coords.dz
    
    R = min(Dx/2, Dy/2, Dz/2) if radius is None else radius
    
    x, y, z = coords.xyz
    if center is None:
        center = (np.mean(x), np.mean(y), np.mean(z))
    
    x = x - center[0]
    y = y - center[1]
    z = z - center[2]
    xyz = np.meshgrid(z, x, y, indexing='ij') # Note: meshgrid swaps y and z
    
    r = np.sqrt(xyz[0]**2 + xyz[1]**2 + xyz[2]**2)
    n = np.clip(n_high / (1 + (r / R)**2), a_min=max(n_low, n_high/2), a_max=None)

    concentration = (n - n_low) / (n_high - n_low) # Convert refractive index to concentration
    
    return concentration

if __name__ == '__main__':
    example_of_usage()
