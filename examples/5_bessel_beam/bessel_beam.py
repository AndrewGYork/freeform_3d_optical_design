import time
import numpy as np
import scipy.special as sp
from beam_propagation import (
    Coordinates, Refractive3dOptic, FixedIndexMaterial, gaussian_beam_2d,
    TrainingData_for_2dImaging, from_tif, to_tif, plot_loss_history)

def get_bessel_training_pair(coords, wavelength):
    """
    Generates a physically accurate, energy-conserved training pair for
    creating a Bessel beam generator.
    """
    # 1. Define the Input Field: A standard, centered Gaussian beam
    input_beam_waist = 5.0

    x, y, _ = coords.xyz
    input_field = gaussian_beam_2d(
        x=x, y=y, x0=0, y0=0, phi=0, theta=0,
        wavelength=float(wavelength), w=input_beam_waist)

    # --- START OF NEW BESSEL BEAM CODE ---
    # 2. Define the Desired Output Field: A Bessel beam
    from scipy.special import jv # Import the Bessel function

    radius = np.sqrt(x**2 + y**2)
    
    # Bessel beam parameters
    alpha = 1.5  # Controls the size of the central spot
    
    # The amplitude is defined by the Bessel function of the first kind, order zero
    desired_amplitude = jv(0, alpha * radius)
    
    # For a simple Bessel beam, the phase is flat (zero)
    desired_phase = np.zeros_like(desired_amplitude)
    
    desired_output_field = (desired_amplitude * np.exp(1j * desired_phase)).astype('complex128').squeeze()
    # --- END OF NEW BESSEL BEAM CODE ---
    
    # 3. Normalize both fields to have the same total power (energy)
    input_power = np.sum(np.abs(input_field)**2)
    output_power = np.sum(np.abs(desired_output_field)**2)

    if input_power > 1e-9:
        input_field /= np.sqrt(input_power)
    if output_power > 1e-9:
        desired_output_field /= np.sqrt(output_power)
            
    # return input_field, desired_output_field
    return input_field, desired_output_field

def bessel_simulation():
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
    coords = Coordinates(xyz_i=(-12.7, -12.7,     0),
                         xyz_f=(+12.7, +12.7, +25.4),
                         n_xyz=(  128,   128,   128))
    print("Voxel dimensions: %0.3f, %0.3f, %0.3f"%(coords.d_xyz))

    # Use these coordinates to initialize an instance of Refractive3dOptic
    # that will simulate how light changes as it passes through our
    # refractive optic:
    ro = Refractive3dOptic(coords)

    # Each voxel of our refractive optic is a mixture of materials:
    air     = FixedIndexMaterial(1)
    polymer = FixedIndexMaterial(1.5)
    ro.set_materials((air, polymer))

    # Initialize our optic.
    try: # If there's a concentration saved to disk, pick up where we left off:
        fname = '01_concentration.tif'
        initial_concentration = from_tif(fname)
        ro.set_3d_concentration(initial_concentration)
        print("Using initial concentration from:", fname)
    except FileNotFoundError:
        print("Using default concentration (50/50 mixture at each voxel).")


    wavelength = 1
    divergence_angle_degrees = 15
    loss_history = []
    for iteration in range(int(1e6)): # Run for a loooong time
        start_time = time.perf_counter()
     
        ### NEW CODE FOR Bessel Beam ###
        input_field, desired_output_field = get_bessel_training_pair(coords, wavelength)
        
        ro.set_2d_input_field(input_field, wavelength)
        ro.set_2d_desired_output_field(desired_output_field)

        # Simulate propagation through our 3D refractive optic,
        # calculate loss, and calculate a gradient that hopefully will
        # reduce the loss:
        ro.gradient_update(
            step_size=100,
            z_planes=(1, 2, 3),
            smoothing_sigma=5)
        # Centered input in this example (x0=y0=0)
        loss_history.append((0.0, 0.0, ro.loss))

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
                   ro.concentration[:, ro.coordinates.ny//2, :])
            to_tif('03_input_field.tif',          ro.input_field)
            to_tif('04_desired_output_field.tif', ro.desired_output_field)
            to_tif('05_calculated_field.tif',
                   np.abs(ro.calculated_field))
            to_tif('06_desired_output_field_3d.tif',
                   np.abs(ro.desired_output_field_3d))
            to_tif('07_calculated_output_field_3d.tif',
                   np.abs(ro.calculated_output_field_3d))
            to_tif('08_error_3d.tif', ro.error_3d)
            to_tif('09_gradient.tif', ro.gradient)
            plot_loss_history(loss_history, '10_loss_history.png')
            to_tif('11_input_phase.tif',             np.angle(ro.input_field))
            to_tif('12_desired_output_phase.tif',    np.angle(ro.desired_output_field))
            to_tif('13_calculated_output_phase.tif', np.angle(ro.calculated_field[-1]))
            # 14-16: 3D phase volumes
            # - Desired/calculated phase across propagated z-planes at the output
            # - Phase inside the optic across z (from calculated_field)
            to_tif('14_desired_output_field_3d_phase.tif',    np.angle(ro.desired_output_field_3d))
            to_tif('15_calculated_output_field_3d_phase.tif', np.angle(ro.calculated_output_field_3d))
            to_tif('16_calculated_field_phase_3d.tif',        np.angle(ro.calculated_field))

            print("done.")
        # if iteration % 5 == 0:
        #     ro.update_attributes()
        #     print("Saving TIFs etc...", end='')
        #     to_tif('composition.tif',          ro.composition)
        #     to_tif('concentration.tif',        ro.concentration)
        #     to_tif('input_field.tif',          ro.input_field)
        #     #to_tif('desired_output_field_amplitude.tif', np.real(ro.desired_output_field))
        #     to_tif('desired_output_field_amplitude.tif', np.abs(ro.desired_output_field))

        #     to_tif('desired_output_field_phase.tif', np.angle(ro.desired_output_field))
        #     to_tif('calculated_field_amplitude.tif',
        #            np.abs(ro.calculated_field))
        #     to_tif('calculated_field_phase.tif',
        #            np.angle(ro.calculated_field))
        #     to_tif('desired_output_field_3d.tif',
        #            np.abs(ro.desired_output_field_3d))
        #     to_tif('calculated_output_field_3d.tif',
        #            np.abs(ro.calculated_output_field_3d))
        #     to_tif('error_3d.tif', ro.error_3d)
        #     to_tif('gradient.tif', ro.gradient)
        #     plot_loss_history(loss_history, '10_loss_history.png')
        #     to_tif('14_desired_output_field_3d_phase.tif',    np.angle(ro.desired_output_field_3d))
        #     to_tif('15_calculated_output_field_3d_phase.tif', np.angle(ro.calculated_output_field_3d))
        #     to_tif('16_calculated_field_phase_3d.tif',        np.angle(ro.calculated_field))

        #     print("done.")

if __name__ == '__main__':
    bessel_simulation()
