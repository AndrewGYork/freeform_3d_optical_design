import time
import numpy as np
import torch
from beam_propagation import (
    Coordinates, Refractive3dOptic, FixedIndexMaterial,
    TrainingData_for_2dImaging, from_tif, to_tif)

def validate_our_algorithm():
    """Example code: validate our forward model with a known 3D optic.

    We'd like to trust our algorithm for simulating propagation (the
    `Interpolated WPM` method). So, lets validate it! Our algorithm also
    has a tunable parameter (the `_refractive_index_bin_size` attribute
    of our `RefractiveOptic` object) that trades off speed vs. accuracy;
    let's get a feel for how to choose its value.

    Maxwell's Fisheye:    
      Simple link: wikipedia.org/wiki/Luneburg_lens#Maxwell's_fish-eye_lens
      Permalink:  https://w.wiki/F962

    ...is a spherically-symmetric gradient-index refractive object that
    perfectly images points on its surface to the opposite side of the
    sphere. Qualitatively, if our forward model is accurate, it should
    convert a centered, focused input beam to a centered, focused output
    beam.

    In order for Maxwell's Fisheye to do what we expect:
       * The input beam should be propagating, not evanescent. This means
         the spot size should be large compared to the wavelength.
       * The input beam should be a focused spot, on the surface of the
         sphere. This means the spot size should be small compared to the
         radius of the sphere.
       * The WPM should be accurate. This means the voxels should be small
         compared to the wavelength.       
    If these conditions are satisfied, we expect the calculated output
    should be a focused spot, the same size and intensity as the input
    spot.

    We also use a (simple, but very slow) implementation of the WPM
    algorithm ( doi.org/10.1364/AO.32.004984 ) as a benchmark for
    quantitative accuracy.
    """
    # Specify our coordinate system, organized via a Coordinates object:
    coords = Coordinates(xyz_i=(-15.0, -15.0, -10.0),
                         xyz_f=(+15.0, +15.0, +10.0),
                         n_xyz=(  301,   301,   201))
    print("Voxel dimensions: %0.3f, %0.3f, %0.3f"%(coords.d_xyz))

    # Use these coordinates to initialize an instance of Refractive3dOptic
    # that will simulate how light changes as it passes through our
    # refractive optic:
    ro = Refractive3dOptic(coords)

    # Each voxel of our refractive optic is a mixture of materials:
    air     = FixedIndexMaterial(1)
    glass = FixedIndexMaterial(2) # This is pretty high for glass, but whatever
    ro.set_materials((air, glass))

    # Initialize our optic.
    diameter = (coords.z_f - coords.z_i) + coords.dz
    R_sq = (diameter/2)**2
    print("Maxwell's Fisheye radius:", diameter/2)
    x, y, z = coords.xyz
    r_sq = x**2 + y**2 + z**2
    n0, n1, n2 = 2, air.get_index(1), glass.get_index(1)
    maxwell_fisheye_index = n0 / (1 + r_sq/R_sq)
    maxwell_fisheye_concentration = (maxwell_fisheye_index - n1)/(n2-n1)
    ro.set_3d_concentration(maxwell_fisheye_concentration)

    # Make a source to generate input and expected output for our optic:
    data_source = TrainingData_for_2dImaging(coords, radius=3)

    wavelength = 0.5
    divergence_angle_degrees = 25

    # Use our data source to generate a single input/output pair:
    x0, y0 = 0, 0
    input_field, desired_output_field = data_source.input_output_pair(
        x0, y0, wavelength, divergence_angle_degrees)
    ro.set_2d_input_field(input_field, wavelength)
    ro.set_2d_desired_output_field(desired_output_field)

    # Simulate propagation through our optic with slow but accurate WPM:
    wpm_filename = 'calculated_field_wpm.tif'
    try:
        calculated_field_wpm_abs = from_tif(wpm_filename)
        print("WPM simulation results loaded from `%s`"%(wpm_filename))
    except FileNotFoundError:
        print("WPM simulation results not found on disk...")
        calculated_field_wpm = wpm(
            input_field=input_field,
            wavelength=wavelength,
            index_of_refraction=maxwell_fisheye_index,
            coords=coords)
        calculated_field_wpm_abs = np.abs(calculated_field_wpm)
        to_tif(wpm_filename, calculated_field_wpm_abs)

    # Simulate propagation through our 3D refractive optic,
    # calculate loss, and calculate a gradient that would
    # reduce the loss. Don't actually update the object (step_size ~= 0)
    print("Simulating Maxwell's Fisheye with our (fast, accurate(?))",
          "`Interpolated WPM`\nalgorithm...")
    for i, bin_size in enumerate(
        (0.32, 0.16, 0.08, 0.04, 0.02, 0.01, 0.005)):
        print("Bin size:", bin_size)
        ro._refractive_index_bin_size = bin_size
        start_time = time.perf_counter()
        ro.gradient_update(
            step_size=1e-12,
            z_planes=(1, 2, 3),
            smoothing_sigma=5)
        end_time = time.perf_counter()
        print("Done. (%0.2f ms elapsed)"%(1000*(end_time - start_time)))
        ro.update_attributes()
        to_tif('calculated_field_interpolated_wpm_%02d.tif'%(i),
               np.abs(ro.calculated_field))
        error = np.abs(ro.calculated_field)**2 - calculated_field_wpm_abs**2
        to_tif('intensity_error_%02d.tif'%(i), error)
        print("Error min/max/mean:, %0.2e %0.2e %0.2e"%(
            error.min(), error.max(), error.mean()))
        to_tif('intensity_error_xz_%02d.tif'%(i), error[:, coords.ny//2, :])
                
    
    # Output some additional results for context. You can use
    # ImageJ ( https://imagej.net/ij/ ) to view the TIF files:
    ro.update_attributes()
    print("Saving TIFs...", end=' ')
    to_tif('00_composition.tif',          ro.composition)
    to_tif('01_concentration.tif',        ro.concentration)
    to_tif('02_concentration_xz.tif',
           ro.concentration[:, ro.coordinates.ny//2, :])
    to_tif('03_input_field.tif',          ro.input_field)
    to_tif('04_desired_output_field.tif', ro.desired_output_field)
    to_tif('05_calculated_field.tif',
           np.abs(ro.calculated_field))
    to_tif('06_desired_output_field_3d',
           np.abs(ro.desired_output_field_3d))
    to_tif('07_calculated_output_field_3d',
           np.abs(ro.calculated_output_field_3d))
    to_tif('08_error_3d.tif', ro.error_3d)
    print("done.")


def wpm(input_field, wavelength, index_of_refraction, coords, try_cuda=True):
    """Calculate light propagation in a 3D refractive object with the WPM

    `WPM` is the "plane wave propagation method", doi.org/10.1364/AO.32.004984
    """
    device = torch.device('cpu')
    if try_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    d = device
    def to_torch(x):
        return torch.from_numpy(x).to(d)
    input_field = to_torch(input_field)
    index_of_refraction = to_torch(index_of_refraction)
    sqrt, exp, pi, c128 = torch.sqrt, torch.exp, torch.pi, torch.complex128
    k, fft, fftfreq = 2*pi/wavelength, torch.fft.fftn, torch.fft.fftfreq
    dx, dy, dz = coords.d_xyz
    nx, ny, nz = coords.n_xyz
    xp = torch.arange(nx, device=d).reshape(1, nx)
    yp = torch.arange(ny, device=d).reshape(ny, 1)
    calculated_field = [input_field]
    print("Calculating propagation with the (slow, but accurate(?)) WPM...")
    for which_z in range(nz):
        print("Slice ", which_z, "...", sep='', end='')
        last_field = calculated_field[-1]
        last_field_ft = fft(last_field) / (ny*nx)
        next_field = torch.zeros_like(last_field, device=d)
        n = index_of_refraction[which_z, :, :]
        kn_sq = ((k*n)**2).to(torch.complex128) # so the sqrt can give imaginary
        for i, fx in enumerate(fftfreq(nx, device=d)):
            print('.', sep='', end='')
            kx = 2*pi*fx/dx
            for j, fy in enumerate(fftfreq(ny, device=d)):
                ky = 2*pi*fy/dy
                kz = sqrt(kn_sq - kx**2 - ky**2)
                next_field += (last_field_ft[j, i] *
                               exp(1j*(kx*xp*dx + ky*yp*dy + kz*dz)))
        calculated_field.append(next_field)
        print('done.')
    calculated_field = torch.stack(calculated_field)
    return calculated_field.cpu().detach().numpy()

if __name__ == '__main__':
    validate_our_algorithm()
