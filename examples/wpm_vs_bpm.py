import torch
import numpy as np


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    # Coordinates
    xi, xf = -20, 20
    yi, yf = -20, 20
    zi, zf = -20, 20
    nx, ny, nz = 128, 128, 128
    x = torch.linspace(xi, xf, nx).reshape( 1,  1, nx).to(DEVICE)
    y = torch.linspace(yi, yf, ny).reshape( 1, ny,  1).to(DEVICE)
    z = torch.linspace(zi, zf, nz).reshape(nz,  1,  1).to(DEVICE)
    dx = (xf - xi)/(nx-1)
    dy = (yf - yi)/(ny-1)
    dz = (zf - zi)/(nz-1)
    dx = torch.tensor(dx, device=DEVICE)
    dy = torch.tensor(dy, device=DEVICE)
    dz = torch.tensor(dz, device=DEVICE)
    print("Voxel dimensions:", dx, dy, dz)

    # Refractive object: Maxwell's Fisheye
    r_sq = x**2 + y**2 + z**2
    n0 = 2
    R_sq = 20**2
    maxwell_fisheye = n0 / (1 + r_sq/R_sq)
    to_tif('index_of_refraction.tif', maxwell_fisheye)

    # Input field: Gaussian beam
    wavelength = 1
    input_field = gaussian_beam_2d(
        x=x, y=y, x0=0, y0=0, phi=0, theta=0, wavelength=wavelength, w=0.5)
    to_tif('input_field.tif', input_field)
    input_field = input_field.to(DEVICE)

    # # Simulate with slow but accurate WPM:
    # try:
    #     calculated_field_wpm_abs = from_tif('calculated_field_wpm.tif').to(torch.float64)
    # except FileNotFoundError:
    #     calculated_field_wpm = wpm(
    #         input_field=input_field,
    #         wavelength=wavelength,
    #         index_of_refraction=maxwell_fisheye,
    #         d_xyz=(dx, dy, dz))
    #     calculated_field_wpm_abs = torch.abs(calculated_field_wpm)
    #     to_tif('calculated_field_wpm.tif', calculated_field_wpm_abs)

    # # Simulate with fast but inaccurate BPM:
    # calculated_field_bpm = bpm(
    #     input_field=input_field,
    #     wavelength=wavelength,
    #     index_of_refraction=maxwell_fisheye,
    #     d_xyz=(dx, dy, dz))
    # calculated_field_bpm_abs = torch.abs(calculated_field_bpm)
    # amplitude_error = calculated_field_bpm_abs - calculated_field_wpm_abs
    # to_tif('amplitude_error_bpm.tif', amplitude_error)
    # to_tif('calculated_field_bpm.tif', calculated_field_bpm_abs)

    prof = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        profile_memory=True,
    )
    prof.start()

    # Simulate with fast(?) but accurate(?) BPM:
    for nn_bucket_size_factor in range(8):
        calculated_field = fast_wpm(
            input_field=input_field,
            wavelength=wavelength,
            index_of_refraction=maxwell_fisheye,
            d_xyz=(dx, dy, dz),
            n_bucket_size=2.0 ** -nn_bucket_size_factor)
        calculated_field = torch.stack(calculated_field)
        calculated_field_abs = torch.abs(calculated_field)
        # amplitude_error = calculated_field_abs - calculated_field_wpm_abs
        # to_tif('amplitude_error_%03d.tif' % (nn_bucket_size_factor), amplitude_error)
##        to_tif('calculated_field_%03d.tif'%(nn), np.abs(calculated_field))
        to_tif('calculated_field_%03d_xz.tif' % (nn_bucket_size_factor),
               torch.abs(calculated_field).sum(axis=1))
    prof.step()
    prof.stop()
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))

def bpm(input_field, wavelength, index_of_refraction, d_xyz):
    """Calculate light propagation in a 3D refractive object with the BPM

    (BPM is the "beam propagation method")
    """
    sqrt, exp, pi = torch.sqrt, torch.exp, torch.pi
    k = 2*pi/wavelength
    fft, ifft = torch.fft.fftn, torch.fft.ifftn
    dx, dy, dz = d_xyz
    nz, ny, nx = index_of_refraction.shape
    calculated_field = [input_field]
    print("Calculating fast (but wrong?) BPM propagation...", sep='', end='')
    for which_z in range(nz):
        print(".", sep='', end='')
        last_field = calculated_field[-1]
        last_field_ft = fft(last_field)
        n = index_of_refraction[which_z, :, :]
        n_mean = n.mean()
        delta_n = (n - n_mean)
        kn_sq = (k*n_mean)**2 + 0j # Complex so the sqrt can give imaginary
        kx = (2*pi/dx)*torch.fft.fftfreq(nx, device=DEVICE).reshape(1, nx)
        ky = (2*pi/dy)*torch.fft.fftfreq(ny, device=DEVICE).reshape(ny, 1)
        kz = sqrt(kn_sq - kx**2 - ky**2)
        phase_shifts = exp(1j*k*delta_n*dz)
        next_field = phase_shifts * ifft(last_field_ft * exp(1j*kz*dz))
        calculated_field.append(next_field)
    print('done')
    calculated_field = torch.stack(calculated_field)
    return calculated_field


def wpm(input_field, wavelength, index_of_refraction, d_xyz):
    """Calculate light propagation in a 3D refractive object with the WPM

    (WPM is the "plane wave propagation method")
    """
    sqrt, exp, pi = torch.sqrt, torch.exp, torch.pi
    k, fft = 2*pi/wavelength, torch.fft.fftn
    dx, dy, dz = d_xyz
    nz, ny, nx = index_of_refraction.shape
    xp = torch.arange(nx).reshape(1, nx)
    yp = torch.arange(ny).reshape(ny, 1)
    calculated_field = [input_field]
    print("Calculating (faster? but accurate?) WPM propagation...",
          sep='', end='')
    for which_z in range(nz):
        print("Slice ", which_z, "...", sep='', end='')
        last_field = calculated_field[-1]
        last_field_ft = fft(last_field) / (ny*nx)
        next_field = torch.zeros_like(last_field)
        n = index_of_refraction[which_z, :, :]
        kn_sq = (k*n)**2 + 0j # Make it complex so the sqrt can give imaginary
        for i, fx in enumerate(torch.fft.fftfreq(nx)):
            print('.', sep='', end='')
            kx = 2*pi*fx/dx
            for j, fy in enumerate(torch.fft.fftfreq(ny)):
                ky = 2*pi*fy/dy
                kz = sqrt(kn_sq - kx**2 - ky**2)
                next_field += (last_field_ft[j, i] *
                               exp(1j*(kx*xp*dx + ky*yp*dy + kz*dz)))
        calculated_field.append(next_field)
        print('done')
    calculated_field = torch.stack(calculated_field)
    return calculated_field


def fast_wpm(input_field, wavelength, index_of_refraction, d_xyz, n_bucket_size: float = 1 / 64):
    # Try to WPM? ...the sorta-fast way
    fft, ifft, fftfreq = torch.fft.fftn, torch.fft.ifftn, torch.fft.fftfreq
    sqrt, exp, pi = torch.sqrt, torch.exp, torch.pi
    k = 2*pi/wavelength
    dx, dy, dz = d_xyz
    nz, ny, nx = index_of_refraction.shape
    index_of_refraction = [i.requires_grad_(True) for i in index_of_refraction]
    calculated_field = [input_field]
    print("Calculating (faster?) WPM propagation...", sep='', end='')
    kx_sq = ((2*pi/dx)*fftfreq(nx, device=DEVICE).reshape(1, 1, nx)).square()
    ky_sq = ((2*pi/dy)*fftfreq(ny, device=DEVICE).reshape(1, ny, 1)).square()
    for n in index_of_refraction:
        print(".", sep='', end='')
        last_field = calculated_field[-1]
        last_field_ft = fft(last_field)
        # It's expensive to propagate in arbitrary inhomogenous
        # refractive indices, so instead, we'll simulate propagation in
        # a limited number of homogenous 'reference' materials:
        n_min, n_max = n.min(), n.max() + 1e-6
        n_buckets = max(2, int((n_max-n_min)/n_bucket_size))
        n_range = torch.linspace(n_min, n_max, n_buckets, device=DEVICE)[:, None, None]
        # print(f"n_min: {n_min}, n_max: {n_max}, n_bucket_size: {n_bucket_size}, n_buckets: {n_buckets}")
        # print(f"n_range: {n_range.shape}")
        kz_sq = (k*n_range).square() - kx_sq - ky_sq # Might be negative, so...
        kz = sqrt(kz_sq.to(torch.complex64)) # complex input -> complex output
        next_field_reference_stack_ft = last_field_ft * exp(1j*kz*dz)
        next_field_reference_stack = ifft(next_field_reference_stack_ft,
                                          dim=(1, 2))
        next_field = z_interpolate(known_values=next_field_reference_stack,
                                   known_z=n_range, desired_z=n)
        calculated_field.append(next_field.clone())
    print('done')
    return calculated_field

"""
I have a vector of values, and a vector of evenly-spaced coordinates of
those values. Given a single new coordinate, I want to find the weighted
sum of the two values that bracket that coordinate, with weights given
by the linear proximity.
"""

def z_interpolate(known_values, known_z, desired_z):
    """Interpolate a 3D stack in the z-direction.
    """
    # 'known_values' is a 3D stack of 2D images, each taken at some
    # fixed, known z-coordinate. At each xy position in 2D, we want to
    # interpolate in the z-direction, to give a value at an unknown,
    # intermediate z-coordinate:
    assert known_values.ndim == 3
    num_values, ny, nx = known_values.shape
    num_intervals = num_values - 1
    assert num_intervals >= 1
    # 'known_z' is a 1D vector, giving the z-coordinate of each 2D image
    # in 'known_values'. 'known_z' must be monotonic increasing, and
    # uniformly spaced:
    assert known_z.shape in ((num_values,), (num_values, 1, 1))
    zi, zf = known_z[0].squeeze(), known_z[-1].squeeze()
    # assert torch.allclose(known_z.squeeze().cpu(), torch.linspace(zi, zf, num_values))
    # 'desired_z' is a 2D array, with the same dimensions as a single
    # slice of 'known_values'. The entries in 'desired_z' are the
    # (unknown) z-coordinates at which we want to estimate values via
    # interpolation. All these z-values must be within the interval
    # covered by 'known_z':
    assert desired_z.shape == (ny, nx)
    assert zi <= desired_z.min() 
    assert desired_z.max() < zf
    
    desired_z_normalized = (desired_z - zi) * (num_intervals / (zf - zi))
    which_interval = torch.floor(desired_z_normalized)
    remainder = desired_z_normalized - which_interval
    which_interval = which_interval.to(torch.int64).reshape(1, ny, nx)
##    to_tif('which_interval.tif', which_interval)
##    to_tif('remainder.tif', remainder)

    lo_bound_vals = torch.gather(known_values, 0, which_interval)
    hi_bound_vals = torch.gather(known_values, 0, which_interval+1)
##    to_tif('lower_bound_values.tif', lo_bound_vals)
##    to_tif('upper_bound_values.tif', hi_bound_vals)

    result = lo_bound_vals*(1-remainder) + hi_bound_vals*remainder
##    to_tif('result.tif', result)
    return result[0, :, :]
    

def output_directory():
    # Put all the files that the demo code makes in their own folder:
    from pathlib import Path
    output_dir = Path(__file__).parent / 'output'
    output_dir.mkdir(exist_ok=True)
    return output_dir
    
def to_tif(filename, x):
    import tifffile as tf
    x = x.detach().cpu().numpy().real.astype('float32')
    if x.ndim == 3:
        x = np.expand_dims(x, axis=(0, 2))
    tf.imwrite(output_directory() / filename, x, imagej=True)

def from_tif(filename):
    import tifffile as tf
    arr = tf.imread(output_directory() / filename)
    return torch.from_numpy(arr).to(DEVICE)

def gaussian_beam_2d(x, y, x0, y0, phi, theta, wavelength, w):
    # Local nicknames:
    exp, sin, cos, pi = torch.exp, torch.sin, torch.cos, torch.pi
    # Simple math:
    k = 2*pi/wavelength
    theta = torch.tensor(theta)
    phi = torch.tensor(phi)
    kx = k*sin(theta)*cos(phi)
    ky = k*sin(theta)*sin(phi)
    r_sq = (x-x0)**2 + (y-y0)**2
    phase = exp(1j*(kx*(x-x0) + ky*(y-y0)))
    amplitude = exp(-r_sq / w**2)
    field = phase * amplitude
    return field.squeeze()

if __name__ == '__main__':
    main()
