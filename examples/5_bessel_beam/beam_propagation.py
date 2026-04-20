import numpy as np
import torch # For calculating gradients

"""
v1.0.6

Techniques for fabricating freeform 3D refractive optics are rapidly
maturing. By 'freeform', I don't just mean the shape - I mean optics
where at each voxel, we can specify the refractive index of the
material. This unlocks crazy new possibilities for optical design - but
how shall we design these optics?

Using gradient search, just like training a neural network!

This module defines a `Refractive3dOptic` class for designing freeform
3D refractive optics, and includes some example code for how to use this
class in the `example_of_usage` string below.

Written by Andrew G. York, licensed CC-BY 4.0.

Inspired and informed by conversations with Shwetadwip Chowdhury, Tanner
Fadero, Dakota Britton, and (presumably) others I'm forgetting. Credit
them for what's good here, and blame me for what's bad. Please tell me
if I should add your name to this list!
"""

##############################################################################
## BEGIN EXAMPLE CODE
##
## The following block of code is an example of usage.
##
## If you execute this module, rather than importing it, it will write a
## copy of this example code to disk as a separate python script. Use
## this as your starting point for learning to use the module.
##
##############################################################################

example_of_usage = """import time
import numpy as np
from beam_propagation import (
    Coordinates, Refractive3dOptic, FixedIndexMaterial,
    TrainingData_for_2dImaging, from_tif, to_tif, plot_loss_history)

def example_of_usage():
    \"""Example code: design a 3D refractive optic with specified input/output.

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
    \"""

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

    # Make a source to generate training data. In this case, the
    # training data is for a simple plane-to-plane inverting imaging
    # system:
    data_source = TrainingData_for_2dImaging(coords, radius=3)

    wavelength = 1
    divergence_angle_degrees = 15
    loss_history = []
    for iteration in range(int(1e6)): # Run for a loooong time
        start_time = time.perf_counter()
        
        # Use our data source to generate random input/output pairs:
        x0, y0 = data_source.random_point_in_a_circle()
        input_field, desired_output_field = data_source.input_output_pair(
            x0, y0, wavelength, divergence_angle_degrees)
        ro.set_2d_input_field(input_field, wavelength)
        ro.set_2d_desired_output_field(desired_output_field)

        # Simulate propagation through our 3D refractive optic,
        # calculate loss, and calculate a gradient that hopefully will
        # reduce the loss:
        ro.gradient_update(
            step_size=100,
            z_planes=(1, 2, 3),
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
            to_tif('09_gradient.tif', ro.gradient)
            plot_loss_history(loss_history, '10_loss_history.png')
            print("done.")

if __name__ == '__main__':
    example_of_usage()
"""

##############################################################################
## END EXAMPLE CODE
##############################################################################


##############################################################################
## The following blocks of code are the heart of the module. You should
## import the module and use these classes, similar to the demo code
## above.
##############################################################################

class Refractive3dOptic:
    """Simulate light propagation through a 3D refractive optic, with autograd.

    We use the resulting gradients to update the 3D optic, to
    (hopefully) design an optic with a desired input/output behavior.
    """
    def __init__(self, coordinates, try_cuda=True):
        assert isinstance(coordinates, Coordinates)
        self.coordinates = coordinates
        assert try_cuda in (True, False)
        self.device = torch.device('cpu')
        if try_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
        self.set_3d_concentration()
        return None

    def set_materials(self, material_list):
        """What materials are we mixing to control the index of refraction?

        The index of refraction at each voxel is the weighted average of
        two (or more) materials, our 'base' material, and our
        'mixer(s)'. Use this function to specify those materials.

        For real materials, the index of refraction depends on the
        wavelength, and you probably want to use a SellmeierMaterial
        object. For example, if each voxel was a mixture of air and
        fused silica, we could write:
        
            air = SellmeierMaterial(
                B=(0.05792105, 0.00167917),
                C=(238.0185, 57.362))
            fused_silica = SellmeierMaterial(
                B=(0.6961663, 0.4079426, 0.8974794),
                C=(0.004679148, 0.01351206, 97.934))
            material_list = [air, fused_silica]

        For simpler simulations that neglect dispersion, you could use a
        fictitious but convenient FixedIndexMaterial:

            air =          FixedIndexMaterial(1)
            fused_silica = FixedIndexMaterial(1.46)
            material_list = [air, fused_silica]
        """
        # For now, we only allow binary mixtures:
        assert len(material_list) == 2
        for m in material_list:
            assert hasattr(m, 'get_index')
        self.material_list = material_list
        self._invalidate(( # Remove these attributes, if they exist:
            'calculated_field', 'calculated_output_field_3d',
            'desired_output_field_3d', 'error_3d', 'loss', 'gradient'))
        return None

    def set_3d_concentration(self, concentration=None):
        """`concentration` is a 3D numpy array describing our refractive optic.

        A concentration of 0 corresponds to a voxel that's entirely the
        'base' material. A concentration of 1 corresponds to a voxel
        that's entirely the 'mixer' material.

        `concentration` is nice for human interpretation, but
        inconvenient for gradient search, since a concentration outside
        the range (0, 1) isn't possible, but gradient search will
        explore outside this range.

        Our current beam propagation model is only accurate if
        `concentration` is very smoothly varying, so caveat emptor.
        """
        nx, ny, nz = self.coordinates.n_xyz
        if concentration is None: # Default to a 50/50 mixture at every voxel
            concentration = np.broadcast_to(0.5, (nz, ny, nx))
        assert concentration.shape == (nz, ny, nx)
        assert np.isrealobj(concentration)
        self.concentration = concentration.astype('float64', copy=True)
        self._invalidate(( # Remove these attributes, if they exist:
            'composition', 'calculated_field', 'calculated_output_field_3d',
            'desired_output_field_3d', 'error_3d', 'loss', 'gradient'))
        return None

    def _set_3d_composition(self, composition):
        """`composition` is a 3D numpy array describing our refractive optic.

        A composition of -inf corresponds to a voxel that's entirely the
        'base' material. A composition of +inf corresponds to a voxel
        that's entirely the 'mixer' material.
        
        `composition` is nice for gradient search, but inconvenient for
        human interpretation.
        """
        # This function is just for convenience; the business logic is
        # in other functions:
        concentration = _to_concentration(composition)
        self.set_3d_concentration(concentration)
        return None

    def set_2d_input_field(self, input_field, wavelength):
        """What light are we shining on our refractive optic?

        `input_field` is a 2D numpy array of complex numbers, specifying
        the amplitude and phase of the input light vs. 2D position at
        the input plane of our refractive optic.

        `wavelength` is a positive number in the same units as our
        Coordinates object (e.g. microns). Note that we're specifying
        the wavelength of light our input field in *vacuum*, not in our
        base material. This is used to calculate how the light spreads
        out as it propagates through each layer of our refractive
        optic, and also used to convert composition to index of refraction.

        If you're simulating dispersion using a SellmeierMaterial, then
        the units of `wavelength` need to be microns.
        """
        self._require('material_list', 'set_materials')
        nx, ny, nz = self.coordinates.n_xyz
        assert input_field.shape == (ny, nx)
        assert (isinstance(input_field, np.ndarray) or
                isinstance(input_field, torch.Tensor))
        assert input_field.dtype in ('complex128', torch.complex128)
        assert wavelength > 0
        warning_string = ("""
    You're using a SellmeierMaterial, which expects the units of
    'wavelength' to be in microns, but your specified wavelength (%0.2f)
    seems to be outside the visible spectrum. Hopefully you know what
    you're doing!\n"""%(wavelength))
        if any([isinstance(m, SellmeierMaterial) for m in self.material_list]):
            if wavelength < 0.3 or 0.9 < wavelength:
                if not hasattr(self, '_SellmeierMaterial_warning'):
                    print(warning_string)
                    self._SellmeierMaterial_warning = True
        self._invalidate(('input_field',)) # Deletes both array and tensor
        if isinstance(input_field, np.ndarray):
            self.input_field = input_field
        elif isinstance(input_field, torch.Tensor):
            self._input_field_tensor = input_field
        self.wavelength = wavelength
        self._invalidate(( # Remove these attributes, if they exist:
            'desired_output_field', 'calculated_field',
            'calculated_output_field_3d', 'desired_output_field_3d',
            'error_3d', 'loss', 'gradient'))
        return None

    def set_2d_desired_output_field(self, desired_output_field):
        """What light do we wish would exit our refractive optic?

        `desired_output_field` is a 2D numpy array of complex numbers,
        specifying the amplitude and phase of the light vs. 2D position
        that we WISH would be produced at the output plane of our
        refractive optic. We use this to calculate loss (aggregate
        error between desired and calculated fields), and we take
        gradients of this loss to update our optic to (hopefully) get
        closer to yielding our desired output.
        """
        self._require('wavelength', 'set_2d_input_field')
        nx, ny, nz = self.coordinates.n_xyz
        assert desired_output_field.shape == (ny, nx)
        assert desired_output_field.dtype == 'complex128'
        self.desired_output_field = desired_output_field
        self._invalidate(( # Remove these attributes, if they exist:
            'desired_output_field_3d', 'error_3d', 'loss', 'gradient'))
        return None

    def gradient_update(self, step_size, z_planes=(1, 2, 3), smoothing_sigma=5):
        """Update our optic to get closer to our desired behavior.

        This is multiple steps rolled into one:
         * Calculate light propagation through our refractive optic.
         * Calculate the loss (aggregate difference between calculated
           and desired behavior).
         * Calculate the gradient of this loss (how can we modify our
           refractive optic to improve its performance?).
         * Update our optic with a smoothed (gaussian filter with
           kernel size = `smoothing_sigma`), scaled (multiplied
           by `step_size`) version of this gradient.

        If you know what you're doing, you can do these steps
        individually, but I often prefer having them rolled into one.

        See `_calculate_loss()` for an explanation of `z_planes`.
        """
        assert step_size > 0
        assert smoothing_sigma >= 0
        step_size = float(step_size)
        smoothing_sigma = float(smoothing_sigma)
        z_planes = [float(z) for z in z_planes]
        # These steps involve pytorch tensors, possibly on the GPU. I
        # find these more annoying to interact with than numpy arrays,
        # but copying to and from the GPU is expensive, so we stay
        # entirely in torch for these steps:        
        self._calculate_3d_field()
        self._calculate_loss(z_planes=z_planes)
        self._calculate_gradient()
        # The gradient usually has high-spatial-frequency content that
        # isn't desirable or manufacturable, so we update our refractive
        # optic with a scaled, smoothed version of the gradient:
        for g, c in zip(self._gradient_tensor, self._composition_tensor):
            update = step_size * smooth_2d(g, sigma=smoothing_sigma)
            c.requires_grad_(False)
            c.subtract_(update)

        self._invalidate( # Most of our numpy attributes become invalid.
            ('composition', 'concentration', 'calculated_field',
             'desired_output_field_3d', 'calculated_output_field_3d',
             'error_3d', 'gradient'),
            # ...but the corresponding tensor attributes are still ok:
            also_invalidate_tensors=False)
        return None

    def update_attributes(self, delete_tensors=True):
        """Convert our private torch tensors to public numpy arrays.

        A typical workflow is to call `gradient_update()` multiple times
        in a loop, and occasionally call `update_attributes()` to copy
        data off of the GPU for visualization and sanity checks.

        By default, we delete the private torch tensors. This can be
        important if you don't want to leave large tensors on a GPU, for
        example.
        """
        for numpy_name in ('composition',
                           'input_field',
                           'calculated_field',
                           'desired_output_field_3d',
                           'calculated_output_field_3d',
                           'error_3d',
                           'gradient'):
            torch_name = '_' + numpy_name + '_tensor'
            if hasattr(self, torch_name):
                tensor = getattr(self, torch_name)
                setattr(self, numpy_name, self._to_numpy(tensor))
                if delete_tensors:
                    delattr(self, torch_name)
        if hasattr(self, 'composition'):
            self.concentration = _to_concentration(self.composition)
        return None

    def _calculate_3d_field(self):
        """Propagate the input field through each z-slice of the volume.

        I think doi.org/10.1364/AO.17.003990 is the OG reference for the
        algorithm we're currently using here to simulate propagation.
        """
        try:
            self._require('_composition_tensor', 'set_3d_concentration')
        except AttributeError:
            self._require('concentration',       'set_3d_concentration')
        try:
            self._require('_input_field_tensor', 'set_2d_input_field')
        except AttributeError:
            self._require('input_field',         'set_2d_input_field')
        self._require('material_list', 'set_materials')
        self._require('wavelength',    'set_2d_input_field')
        # How do amplitude and phase change from one slice to the next?
        # Regardless of the optic, we want to propagate "between"
        # slices as if we were in a homogenous medium with absorbing
        # boundary conditions:
        phase_mask     = self._propagation_phase_mask(self.coordinates.dz)
        amplitude_mask = self._apodization_amplitude_mask()
        # Use Torch so we can calculate gradients:
        fft, ifft, exp = torch.fft.fftn, torch.fft.ifftn, torch.exp
        if not hasattr(self, '_composition_tensor'):
            # Note that this is a list of 2D tensors, not a 3D tensor
            # like you might expect. I think this important for the
            # performance of backpropagation, but maybe I just don't
            # understand pytorch:
            self._composition_tensor = [_to_composition(self._to_torch(c))
                                        for c in self.concentration]
        if not hasattr(self, '_input_field_tensor'):
            self._input_field_tensor = self._to_torch(self.input_field)
        # Propagate the light through the optic, one slice at a time:
        calculated_field = [self._input_field_tensor]
        for c in self._composition_tensor:
            # The composition is the quantity we ultimately want to
            # update via gradient search, so it `requires_grad`:
            c.requires_grad_(True)
            # Convert the composition to phase shifts:
            phase_shifts = self._composition_to_phase_shifts(c)
            # This step accounts for 99% of the computational burden of
            # this entire module:
            # fft, multiply, ifft, multiply
            calculated_field.append(
                amplitude_mask * exp(1j * phase_shifts) *
                ifft(phase_mask * fft(calculated_field[-1])))
        # Save results as attributes, not return values:
        self._calculated_field_tensor = calculated_field
        return None

    def _calculate_loss(self, z_planes=(1, 2, 3)):
        """How well does our calculated field match our desired field?

        `z_planes` is a list of z-offsets (in the same units as our
        `Coordinates` object, relative to the output plane of our 3D
        refractive optic) at which we'll calculate the intensity
        mismatch between our calculated field and our desired field.

        This 3D intensity-only penalty is a convenient way to penalize
        erros in both position (intensity) and direction (phase) of our
        calculated field, without ever directly referring to phase. In
        my hands, it works better than any 2D penalty that refers
        directly to both intensity and phase.
        """
        self._require('desired_output_field', 'set_2d_desired_output_field')
        self._require('_calculated_field_tensor', '_calculate_3d_field')
        # We want gradients, so we'll calculate our loss using Torch:
        desired_output_field = self._to_torch(self.desired_output_field)
        calculated_output_field = self._calculated_field_tensor[-1]
        # Since our fields are complex, we have to decide how to
        # penalize both intensity and phase errors. My favorite way to
        # do this is to simulate propagation in free space for both the
        # calculated and the desired fields, and compare the intensity
        # mismatch at multiple different z-planes.
        # These are lists of 2D tensors:
        desired_output_field_3d    = [desired_output_field]
        calculated_output_field_3d = [calculated_output_field]
        for dz in z_planes:
            d_at_dz = self._freespace_propagation(desired_output_field,    dz)
            c_at_dz = self._freespace_propagation(calculated_output_field, dz)
            desired_output_field_3d.append(   d_at_dz)
            calculated_output_field_3d.append(c_at_dz)

        loss = torch.zeros(1, device=self.device, dtype=torch.float64)
        error_3d = [] # Useful for visualization
        for d, c in zip(desired_output_field_3d, calculated_output_field_3d):
            desired_intensity    = d.abs()**2
            calculated_intensity = c.abs()**2
            intensity_error = (calculated_intensity - desired_intensity)
            error_3d.append(intensity_error)
            worst_case_intensity_error = (desired_intensity +
                                          calculated_intensity).sum()
            loss += intensity_error.abs().sum() / worst_case_intensity_error
        loss = loss / (len(z_planes) + 1)
        # Save our results as attributes, not return values:
        self._desired_output_field_3d_tensor = desired_output_field_3d
        self._calculated_output_field_3d_tensor = calculated_output_field_3d
        self._error_3d_tensor = error_3d
        self._loss_tensor = loss
        self.loss = self._to_numpy(loss)[0]
        return None

    def _calculate_gradient(self):
        """How might we change `composition` in order to improve `loss`?
        """
        self._require('_composition_tensor', '_calculate_3d_field')
        self._require('_loss_tensor', '_calculate_loss')
        for c in self._composition_tensor:
            if c.grad is not None:
                c.grad.zero_()
        self._loss_tensor.backward()
        self._gradient_tensor = [c.grad for c in self._composition_tensor]
        return None

    def _propagation_phase_mask(self, distance):
        """For simulating propagation in homogenous materials

        This is only used inside other private methods, so its
        implemented in torch Tensors, not numpy arrays.
        """
        self._require('wavelength', 'set_2d_input_field')
        # Local nicknames:
        nx, ny, nz = self.coordinates.n_xyz
        dx, dy, dz = self.coordinates.d_xyz
        wavelength, d = self.wavelength, self.device
        fftfreq, nan_to_num = torch.fft.fftfreq, torch.nan_to_num
        pi, sqrt, exp, float64 = torch.pi, torch.sqrt, torch.exp, torch.float64
        # Spatial frequencies as a function of position in FFT space:
        k  = 2*pi / wavelength
        kx = 2*pi * fftfreq(nx, dx, dtype=float64, device=d).reshape( 1, nx)
        ky = 2*pi * fftfreq(ny, dy, dtype=float64, device=d).reshape(ny,  1)
        # with np.errstate(invalid='ignore'): # I don't need this for torch?
        kz = nan_to_num(sqrt(k**2 - kx**2 - ky**2))
        phase_mask = exp(1j*kz*distance)
        return phase_mask

    def _apodization_amplitude_mask(self):
        """For simulating propagation with non-periodic boundary conditions.

        We want absorptive boundary conditions, not reflective or
        periodic boundary conditions, so we smoothly reduce the
        transmission amplitude near the lateral edges of the simulation
        volume.

        This is only used inside other private methods, so its
        implemented in torch Tensors, not numpy arrays.
        """
        if not hasattr(self, '_apodization_edge_pixels'):
            self._apodization_edge_pixels = 5
        if not hasattr(self, '_apodization_edge_value'):
            self._apodization_edge_value = 0.5
        edge_pixels = int(self._apodization_edge_pixels)
        edge_value = float(self._apodization_edge_value)
        assert edge_pixels > 0
        assert edge_value >= 0
        assert edge_value <= 1
        nx, ny, nz = self.coordinates.n_xyz
        assert nx > 2*edge_pixels
        assert ny > 2*edge_pixels
        # Local nicknames:
        ones, linspace, float64 = torch.ones, torch.linspace, torch.float64
        def linear_taper(n):
            mask = ones(n, dtype=float64, device=self.device)
            mask[0:edge_pixels] = linspace(edge_value, 1, edge_pixels)
            mask[-edge_pixels:] = linspace(1, edge_value, edge_pixels)
            return mask
        amplitude_mask = (linear_taper(nx).reshape(1, nx) *
                          linear_taper(ny).reshape(ny, 1))
        return amplitude_mask

    def _composition_to_phase_shifts(self, composition):
        """Convert our `composition` tensor to phase shifts at each voxel.

        The propagation simulation wants to know phase shifts at each
        voxel due to our refractive optic, which depends on the
        `composition` at each voxel, the materials that we're mixing, the
        size of the voxel, and the wavelength of the propagating light.
        """
        # `composition` must be a tensor, to allow autograd:
        assert isinstance(composition, torch.Tensor)
        # Local nicknames for the scalars:
        dz, wavelength, pi = self.coordinates.dz, self.wavelength, np.pi

        # The index of refraction is a weighted average of our
        # materials. For now, we only implement binary mixtures:
        concentration = _to_concentration(composition)
        self.index_list = [m.get_index(wavelength) for m in self.material_list]
        assert len(self.index_list) == 2
        index_1, index_2 = self.index_list
        index_minus_1 = (index_1 - 1) + (index_2 - index_1)*concentration
        
        # The phase shift scales with dz and inversely with wavelength:
        phase_shifts = 2*pi * dz * index_minus_1 / wavelength
        
        return phase_shifts # This is a pytorch Tensor (which allows autograd)

    def _freespace_propagation(self, field, distance):
        """
        Like `_calculate_3d_propagation()`, but for a single step, with
        no edge absorption and no phase shifts. We use this internally
        to calculate the loss function.
        """
        nx, ny, nz = self.coordinates.n_xyz
        assert field.shape == (ny, nx)
        phase_mask = self._propagation_phase_mask(distance)
        if isinstance(field, np.ndarray):     # Numpy input, return an array
            assert np.iscomplexobj(field)
            fft, ifft = np.fft.fftn, np.fft.ifftn
            phase_mask = self._to_numpy(phase_mask)
        elif isinstance(field, torch.Tensor): # Torch input, return a tensor
            assert torch.is_complex(field)
            fft, ifft = torch.fft.fftn, torch.fft.ifftn
        field_after_propagation = ifft(phase_mask * fft(field))
        return field_after_propagation

    def _invalidate(
        self,
        iterable_of_attribute_names,
        also_invalidate_tensors=True
        ):
        # Many of the methods above need to invalidate (i.e. delete)
        # multiple attributes. This makes it a little more convenient:
        for attr in iterable_of_attribute_names:
            if hasattr(self, attr): delattr(self, attr)
            if also_invalidate_tensors:
                tensor_attr = '_' + attr + '_tensor'
                if hasattr(self, tensor_attr): delattr(self, tensor_attr)
        return None

    def _require(self, attribute_name, prerequisite_function_name):
        # Many of the methods above need to be called in the expected
        # order, to create attributes that later methods depend on.
        # Check for a required attribute, and try to print a useful
        # error message if it's not present:
        if not hasattr(self, attribute_name):
            raise AttributeError(
                "No attribute `%s`. Did you call `%s()` yet?"%(
                    attribute_name, prerequisite_function_name))
        return None

    def _to_torch(self, x):
        """We convert 2D numpy arrays directly to 2D tensors, but we
        convert 3D numpy arrays to lists of 2D tensors rather than a 3D
        tensor, to avoid quadratic complexity during backpropagation.
        """
        assert x.ndim in (2, 3)
        if x.ndim == 2:
            return torch.from_numpy(x).to(self.device)
        elif x.ndim == 3:
            return [torch.from_numpy(x[i, :, :]).to(self.device)
                    for i in range(x.shape[0])]

    def _to_numpy(self, x):
        if isinstance(x, torch.Tensor): # Convert directly to a 2D array
            assert x.ndim in (1, 2)
            return x.cpu().detach().numpy()
        elif isinstance(x, list): # Convert to a 3D numpy array
            # This is kinda janky but seems correct at least.
            x0 = x[0].cpu().detach().numpy()
            nz = len(x)
            ny, nx = x0.shape
            out = np.zeros((nz, ny, nx), dtype=x0.dtype)
            out[0, :, :] = x0
            for i in range(1, nz):
                out[i, :, :] = x[i].cpu().detach().numpy()
            return out
        else:
            assert type(x) in (torch.Tensor, list)

def _to_concentration(composition):
    """See `set_3d_concentration` for details
    """
    if isinstance(composition, torch.Tensor):
        arctan = torch.arctan
    elif isinstance(composition, np.ndarray):
        arctan = np.arctan
    # Since -pi/2 < arctan < pi/2, this guarantees 0 < concentration < 1
    concentration = 0.5 + arctan(composition) / np.pi
    return concentration

def _to_composition(concentration):
    """See `set_3d_composition` for details.
    """
    if isinstance(concentration, torch.Tensor):
        tan, clip = torch.tan, torch.clip
    elif isinstance(concentration, np.ndarray):
        tan, clip = np.tan, np.clip
    # We enforce 0 < concentration < 1, which guarantees finite values
    # for `composition`:
    eps = 1e-6
    concentration = clip(concentration, eps, 1-eps)
    composition = tan(np.pi*(concentration - 0.5))
    return composition

def smooth_2d(a, sigma=5):
    """Smooth a 2D torch tensor via convolution with a small Gaussian kernel

    Similar to scipy.ndimage.gaussian_filter(), but (potentially)faster via GPU.
    """
    assert a.ndim == 2
    assert isinstance(a, torch.Tensor)
    # Local nicknames:
    arange, exp, conv2d = torch.arange, torch.exp, torch.nn.functional.conv2d
    # Make a gaussian kernel:
    radius = int(4*sigma + 0.5)
    x = arange(-radius, radius+1, dtype=a.dtype, device=a.device)
    gaussian = exp((-0.5/sigma**2) * x**2)
    gaussian = gaussian / gaussian.sum()
    # Use pytorch for 2d convolution.
    # The shapes that conv2d expects are all minibatch-silly:
    s0, s1 = a.shape, (1, 1) + a.shape
    for s2 in ((1, 1, 1, len(x)), (1, 1, len(x), 1)):
        # We use a pair of 2D convolutions with a pair of 1D kernels. Wheee!
        a = conv2d(a.view(s1), gaussian.view(s2), padding='same')
    return a.view(s0) # Clip off the silly extra dimensions.

class SellmeierMaterial:
    """The Sellmeier equation is an empirical relationship between
    refractive index and wavelength for a particular transparent medium.

    https://en.wikipedia.org/wiki/Sellmeier_equation
    """
    def __init__(self, B=(0, 0, 0), C=(0, 0, 0)):
        """Three terms is typical (one per resonance), but we might as
        well allow any number of resonances.
        """
        assert len(B) == len(C)
        B = [float(x) for x in B]
        C = [float(x) for x in C]
        self.B = B
        self.C = C
        return None

    def get_index(self, wavelength_um):
        lamda_sq = wavelength_um * wavelength_um
        index_sq = 1
        for b, c in zip(self.B, self.C):
            index_sq += b * lamda_sq / (lamda_sq - c)
        index = np.sqrt(index_sq)
        return index

class FixedIndexMaterial:
    """A conceptually simple material with no dispersion.

    In real life, the index of refraction for a material depends on the
    wavelength. However, sometimes it's nice to simulate simple things,
    so here's a (fictitious) material that has the same index of
    refraction for all wavelengths.
    """
    def __init__(self, index):
        index = float(index)
        self.index = index
        return None

    def get_index(self, wavelength_um):
        return self.index

class Coordinates:
    """A convenience class for keeping track of the coordinates of our voxels.

     - xyz_i: a 3-element tuple storing the coordinates of our initial voxel
     - xyz_f: a 3-element tuple storing the coordinates of our final voxel
     - n_xyz: a 3-element tuple specifying how many voxels we have in x, y, z

    There's nothing complicated here, but this is the type of detail I
    tend to get wrong if I don't make a convenience class to organize it.
    """
    def __init__(self, xyz_i, xyz_f, n_xyz):
        # Sanitize and organize...
        # - Inputs:
        xi, yi, zi = map(float, xyz_i)
        xf, yf, zf = map(float, xyz_f)
        nx, ny, nz = map(  int, n_xyz)
        # If your simulation cross section is too small, you're
        # dominated by edge effects; we demand at least 1x21x21 pixels:
        assert nx > 20
        assert ny > 20
        assert nz > 0
        # - Initial and final voxel positions:
        self.xyz_i = xi, yi, zi
        self.x_i, self.y_i, self.z_i = self.xyz_i
        self.xyz_f = xf, yf, zf
        self.x_f, self.y_f, self.z_f = self.xyz_f
        # - Position of voxels:
        self.xyz = (np.linspace(xi, xf, nx).reshape( 1,  1, nx),
                    np.linspace(yi, yf, ny).reshape( 1, ny,  1),
                    np.linspace(zi, zf, nz).reshape(nz,  1,  1))
        self.x, self.y, self.z = self.xyz
        # - Shape of voxels:
        dx, dy, dz = (xf-xi)/(nx-1), (yf-yi)/(ny-1), (zf-zi)/(nz-1)
        self.d_xyz = dx, dy, dz
        self.dx, self.dy, self.dz = self.d_xyz
        # - Number of voxels:
        self.n_xyz = nx, ny, nz
        self.nx, self.ny, self.nz = self.n_xyz
        return None

class RefractiveOpticSequence:
    def __init__(self, optics_list):
        """Multiple Refractive3dOptics in a row.

        The output field of the Nth Refractive3dOptic is the input field
        to the N+1th Refractive3dOptic.

        For example, you might want to simulate a big air gap in between
        two 3D printed optics:
        
        air = FixedIndexMaterial(1)
        polymer = FixedIndexMaterial(1.5)
        coords = Coordinates(xyz_i=(-12.7, -12.7, -12.7),
                             xyz_f=(+12.7, +12.7, +12.7),
                             n_xyz=(  128,   128,   128))

        printed_optic_1 = Refractive3dOptic(coords)
        printed_optic_1.set_materials((air, polymer))

        air_gap = Refractive3dOptic(coords)
        air_gap.set_materials((air, air))
        air_gap.allow_gradient_update = False

        printed_optic_2 = Refractive3dOptic(coords)
        printed_optic_2.set_materials((air, polymer))

        my_optics = RefractiveOpticSequence((printed_optic_1,
                                             air_gap,
                                             printed_optic_2))
        """
        assert len(optics_list) > 0
        for o in optics_list:
            assert isinstance(o, Refractive3dOptic)
            if not hasattr(o, 'allow_gradient_update'):
                o.allow_gradient_update = True
            assert o.allow_gradient_update in (True, False)
        for o1, o2 in zip(optics_list[ :-1], optics_list[1:  ]):
            # We don't check that the z-coordinates of our consecutive
            # optics are consistent, but be thoughtful about the fact
            # that the z-positions of the refractive voxels and the
            # z-positions of the calculated field have a "fencepost"
            # relationship:
            #
            # https://en.wikipedia.org/wiki/Off-by-one_error#Fencepost_error
            #
            # We *do* check that consecutive optics share XY coordinates:
            c1, c2 = o1.coordinates, o2.coordinates
            assert c1.nx == c2.nx # Consecutive optics have same # of x-pixels
            assert c1.ny == c2.ny # Consecutive optics have same # of y-pixels
            assert np.allclose(c1.x, c2.x) # Consecutive optics share x coords
            assert np.allclose(c1.y, c2.y) # Consecutive optics share y coords
        self.optics = optics_list
        return None

    def disable_gradient_update(self, optic):
        """We might not want to optimize all of the optics in our sequence.
        """
        assert optic in self.optics
        optic.allow_gradient_update = False
        return None

    def set_2d_input_field(self, input_field, wavelength):
        """See `Refractive3dOptic.set_2d_input_field()` for details
        """
        first_optic, last_optic = self.optics[0], self.optics[-1]
        first_optic.set_2d_input_field(input_field, wavelength)
        last_optic.wavelength = wavelength
        return None

    def set_2d_desired_output_field(self, desired_output_field):
        """See `Refractive3dOptic.set_2d_desired_output_field()` for details
        """
        # This will get overwritten, it's just to trigger input sanitization:
        self.optics[-1].set_2d_desired_output_field(desired_output_field)
        # This gets remembered, though:
        self.desired_output_field = desired_output_field
        return None

    def gradient_update(self, step_size, z_planes=(1, 2, 3), smoothing_sigma=5):
        """See `Refractive3dOptic.gradient_update()` for details
        """
        assert step_size > 0
        assert smoothing_sigma >= 0
        step_size = float(step_size)
        smoothing_sigma = float(smoothing_sigma)
        z_planes = [float(z) for z in z_planes]
        
        pairs_of_optics = zip(self.optics[:-1], self.optics[1:])
        final_optic = self.optics[-1]
        try:
            for i, (current_optic, next_optic) in enumerate(pairs_of_optics):
                # Calculate propagation through the current optic:
                current_optic._calculate_3d_field()
                # Use the output field of the current optic as the input
                # field of the next optic:
                next_optic.set_2d_input_field(
                    input_field=current_optic._calculated_field_tensor[-1],
                    wavelength=current_optic.wavelength)
            # Calculate loss just for the final optic:
            final_optic.set_2d_desired_output_field(self.desired_output_field)
            delattr(self, 'desired_output_field')
            final_optic._calculate_3d_field()
            final_optic._calculate_loss(z_planes=z_planes)
            self.loss = np.copy(final_optic.loss)
            # Zero the gradient for all the optics:
            for i, o in enumerate(self.optics):
                for c in o._composition_tensor:
                    if c.grad is not None:
                        c.grad.zero_()
            # Backpropagate to populate our gradients:
            final_optic._loss_tensor.backward()
            for i, o in enumerate(self.optics):
                o._gradient_tensor = [c.grad for c in o._composition_tensor]
            # Update our optics using our gradients:
            for i, o in enumerate(self.optics):
                for g, c in zip(o._gradient_tensor, o._composition_tensor):
                    c.requires_grad_(False)
                    if o.allow_gradient_update:
                        update = step_size*smooth_2d(g, sigma=smoothing_sigma)
                        c.subtract_(update)
        except:
            print("While calculating with optic %i, an exception occured:"%(i))
            raise
        
        return None

    def update_attributes(self, delete_tensors=True):
        for op in self.optics:
            op.update_attributes(delete_tensors=delete_tensors)

##############################################################################
## The following utility code is used for the demo in the 'main' block,
## it's not critical to the module.
##############################################################################

def output_directory():
    # Put all the files that the demo code makes in their own folder:
    from pathlib import Path
    output_dir = Path(__file__).parent / 'output'
    output_dir.mkdir(exist_ok=True)
    return output_dir

def to_tif(filename, x):
    import tifffile as tf
    if hasattr(x, 'detach'):
        x = x.detach()
    x = np.asarray(x).real.astype('float32')
    if x.ndim == 3:
        x = np.expand_dims(x, axis=(0, 2))
    tf.imwrite(output_directory() / filename, x, imagej=True)

def from_tif(filename):
    import tifffile as tf
    return tf.imread(output_directory() / filename)

def attributes_to_tifs(refractive_optic_sequence, list_of_attributes):
    assert isinstance(refractive_optic_sequence, RefractiveOpticSequence)
    for n, optic in enumerate(refractive_optic_sequence.optics):
        for i, attribute_name in enumerate(list_of_attributes):
            if hasattr(optic, attribute_name):
                attr = getattr(optic, attribute_name)
                if np.iscomplexobj(attr):
                    attr = np.abs(attr)
                filename = "%02d_optic_%02d_%s.tif"%(
                    i, n, attribute_name)
                to_tif(filename, attr)

def plot_loss_history(loss_history, filename):
    import matplotlib as mpl
    mpl.use('agg') # Prevents a memory leak from repeated plotting
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from scipy import ndimage as ndi

    loss_history = np.asarray(loss_history)
    x0, y0, loss = loss_history.T
    r = np.sqrt(x0**2 + y0**2)
    smooth_loss = ndi.gaussian_filter(loss, sigma=30)
    fig = plt.figure()
    plt.scatter(range(len(loss)), loss, s=7, c=r)
    plt.plot(smooth_loss)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.colorbar(label="Input radial position")
    plt.grid('on', alpha=0.1)
    plt.savefig(output_directory() / filename)
    plt.close(fig)
    return None

class TrainingData_for_2dImaging:
    """An example of how to generate training data for an imaging optic.

    This generates input/output pairs that image a pointlike source at
    the 2d input plane to an inverted (but otherwise identical) image of
    the input plane to the output plane.
    """
    def __init__(self, coordinates, radius):
        assert isinstance(coordinates, Coordinates)
        self.coordinates = coordinates
        assert radius > 0
        self.radius = radius
        return None

    def random_point_in_a_circle(self):
        # Local nicknames:
        R, sin, cos, pi, sqrt = self.radius, np.sin, np.cos, np.pi, np.sqrt
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
        # Desired output beam is an inverted image of the same point:
        desired_output_field = input_field[::-1, ::-1].copy()
        return input_field, desired_output_field
        
def gaussian_beam_2d(x, y, x0, y0, phi, theta, wavelength, w):
    # Local nicknames:
    exp, sin, cos, pi = np.exp, np.sin, np.cos, np.pi
    # Simple math:
    k = 2*pi/wavelength
    kx = k*sin(theta)*cos(phi)
    ky = k*sin(theta)*sin(phi)
    r_sq = (x-x0)**2 + (y-y0)**2
    phase = exp(1j*(kx*(x-x0) + ky*(y-y0)))
    amplitude = exp(-r_sq / w**2)
    field = phase * amplitude
    return field.squeeze()

if __name__ == '__main__':
    """Save our example code to disk, so you can execute it.
    """
    from pathlib import Path

    print("Saving 'example_of_usage.py' to disk... ", end='')
    this_module = Path(__file__).name
    working_directory = Path(__file__).parent
    filename = working_directory / "example_of_usage.py"
    with open(filename, 'w') as f:
        f.write(example_of_usage)
    print("done.")
    print("\nOpen, read, and execute 'example_of_usage.py'",
          "for an example of how to import\nand use the objects defined in",
          this_module, "\n")
    print("If you edit 'example_of_usage.py', make sure you rename it.")