import numpy as np
import torch # For calculating gradients

"""
v1.0.0

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
Fadero, Dakota Britton, Jordão Bragantini, Gabriel (Gav) Sturm, Seth
Hinz, Vincent Selhorst-Jones, Megan Fu, and (presumably) others I'm
forgetting. Credit them for what's good here, and blame me for what's
bad. Please tell me if I should add your name to this list!
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
    TrainingData_for_2dImaging, from_tif, to_tif, plot_loss_history, show_ray_diagram, show_ray_bundle)

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
    \"""Design a Maxwell fisheye (Luneberg lens) using the specified materials.\"""
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
            'calculated_raybundle_sequence', 'calculated_output_raybundle_3d',
            'desired_output_raybundle_3d', 'error_3d', 'loss', 'gradient'))
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
            'composition', 'calculated_raybundle_sequence', 'calculated_output_raybundle_3d',
            'desired_output_raybundle_3d', 'error_3d', 'loss', 'gradient'))
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
    
    def set_input_raybundle(self, input_raybundle, wavelength):
        """What rays are we shining on our refractive optic?

        `input_raybundle` is a RayBundle object, specifying the positions
        and directions of rays at the input plane of our refractive optic.
        
        `wavelength` is a positive number in the same units as our 
        Coordinates object (e.g. microns). Note that we're specifying
        the wavelength of light our input rays in *vacuum*, not in our
        base material. This is used to convert composition to index of
        refraction.
        
        If you're simulating dispersion using a SellmeierMaterial, then
        the units of `wavelength` need to be microns.
        """
        self._require('material_list', 'set_materials')
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
        assert isinstance(input_raybundle, RayBundle)
        self._invalidate(('input_raybundle',)) # Deletes both array and tensor
        self.input_raybundle = input_raybundle
        self.wavelength = wavelength
        self._invalidate(( # Remove these attributes, if they exist:
            'desired_output_raybundle', 'calculated_raybundle_sequence', 'error', 'loss', 'gradient'))
        return None
    
    def set_desired_output_raybundle(self, desired_output_raybundle):
        """What rays do we wish would exit our refractive optic?

        `desired_output_raybundle` is a RayBundle object, specifying the
        positions and directions of rays at the output plane of our
        refractive optic. We use this to calculate loss (aggregate
        error between desired and calculated raybundles), and we take
        gradients of this loss to update our optic to (hopefully) get
        closer to yielding our desired output.
        """
        assert isinstance(desired_output_raybundle, RayBundle)
        self.desired_output_raybundle = desired_output_raybundle
        self._invalidate(( # Remove these attributes, if they exist:
            'calculated_raybundle_sequence', 'error', 'loss', 'gradient'))
        return None

    def gradient_update(self, step_size, smoothing_sigma=5):
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
        # These steps involve pytorch tensors, possibly on the GPU. I
        # find these more annoying to interact with than numpy arrays,
        # but copying to and from the GPU is expensive, so we stay
        # entirely in torch for these steps:        
        self._calculate_3d_raybundle_sequence()
        self._calculate_loss()
        self._calculate_gradient()
        # The gradient usually has high-spatial-frequency content that
        # isn't desirable or manufacturable, so we update our refractive
        # optic with a scaled, smoothed version of the gradient:
        g = self._gradient_tensor
        c = self._composition_tensor
        update = step_size * smooth_3d(g, sigma=smoothing_sigma)
        c.requires_grad_(False)
        c.subtract_(update)

        self._invalidate( # Most of our numpy attributes become invalid.
            ('composition', 'concentration', 'calculated_raybundle_sequence',
             'desired_output_raybundle', 'calculated_output_raybundle',
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
                           'input_raybundle',
                           'calculated_raybundle_sequence',
                           'desired_output_raybundle',
                           'calculated_output_raybundle',
                           'error_3d',
                           'gradient'):
            torch_name = '_' + numpy_name + '_tensor'
            if hasattr(self, torch_name):
                tensor = getattr(self, torch_name)
                if isinstance(tensor, RayBundle) or (isinstance(tensor, list) and isinstance(tensor[0], RayBundle)):
                    setattr(self, numpy_name, self._to_numpy_raybundle(tensor))
                else: 
                    setattr(self, numpy_name, self._to_numpy(tensor))
                if delete_tensors:
                    delattr(self, torch_name)
        if hasattr(self, 'composition'):
            self.concentration = _to_concentration(self.composition)
        return None
    
    def _get_acceleration(self, acceleration, xyz):
        """Estimate acceleration at arbitrary xyz positions via interpolation.
        """
        assert xyz.device == acceleration.device
        c, a_xyz = self.coordinates, acceleration
        # The 3D interpolation routine in torch wants xyz scaled to the
        # range (-1, 1). This is a little silly, but whatever:
        xyz_i = torch.tensor(c.xyz_i, device=acceleration.device)
        xyz_f = torch.tensor(c.xyz_f, device=acceleration.device)
        center_point = 0.5*(xyz_f + xyz_i)
        radius       = 0.5*(xyz_f - xyz_i)
        xyz_scaled = (xyz - center_point) * (1/radius)
        # The 3D interpolation routine in torch wants `xyz` and
        # `acceleration` to have a few extra dimensions that we're not
        # using. This is a little silly, but whatever:
        xyz_scaled = xyz_scaled.reshape((1, 1, 1,) + xyz.shape)
        a_xyz = a_xyz.reshape((1,) + a_xyz.shape)
        # Now we can interpolate, and strip off the extra dimensions:
        a_interpolated = torch.nn.functional.grid_sample(
            a_xyz, xyz_scaled, mode='bilinear', align_corners=True
            ).reshape(3, xyz.shape[0])
        # The interpolation routine wants acceleration shaped (3, nz,
        # ny, nx), and gives output shaped (3, num_rays), but we want
        # our output shaped (nrays, 3):
        a_interpolated = torch.transpose(a_interpolated, 0, 1)
        return a_interpolated
    
    def _step_rays(self, raybundle, acceleration, dt):
        """Step a raybundle forward by time interval dt"""
        # Shorter nicknames
        xyz, v_xyz, a = raybundle.xyz, raybundle.v_xyz, lambda xyz: self._get_acceleration(acceleration, xyz)
        # Calculate A, B, C, for Sharma's algorithm:
        A = dt * a(xyz)
        B = dt * a(xyz + (1/2)*dt*v_xyz + (1/8)*dt*A)
        C = dt * a(xyz +       dt*v_xyz + (1/2)*dt*B)
        # Calculate the position/velocity update for Sharma's algorithm:
        xyz_f = xyz + dt*(v_xyz + (1/6)*(A + 2*B))
        v_xyz_f = v_xyz + (1/6)*(A + 4*B + C)
        return RayBundle(xyz_f, v_xyz_f)

    def _calculate_3d_raybundle_sequence(self):
        """Propagate our input raybundle through our 3D refractive optic 
        using Runge-Kutta ray tracing.
        
        See Sharma (1982) for details of the algorithm ( doi.org/10.1364/AO.21.000984 )
        """
        try:
            self._require('_composition_tensor', 'set_3d_concentration')
        except AttributeError:
            self._require('concentration',       'set_3d_concentration')
        try:
            self._require('_input_raybundle_tensor', 'set_2d_input_raybundle')
        except AttributeError:
            self._require('input_raybundle',         'set_2d_input_raybundle')
        self._require('material_list', 'set_materials')
        # Use Torch so we can calculate gradients:
        if not hasattr(self, '_composition_tensor'):
            self._composition_tensor = self._to_3d_torch(_to_composition(self.concentration))
        if not hasattr(self, '_input_raybundle_tensor'):
            self._input_raybundle_tensor = self._to_torch_raybundle(self.input_raybundle)
        self._composition_tensor.requires_grad_(True)            
        # Nicknames:
        x, y, z = self.coordinates.xyz
        nx, ny, nz = self.coordinates.n_xyz
        dx, dy, dz = self.coordinates.d_xyz
        x_f, y_f, z_f = self.coordinates.xyz_f
        # Parameters
        dt = dz/5 # Step size for propagation; smaller = more accurate
        max_steps = int(20*nz*dz/dt) # Max steps in case of total internal reflection 
        # Calculate acceleration
        acceleration = self._composition_to_acceleration(self._composition_tensor)
        # Propagate the light through the optic, one slice at a time:
        num_rays = self._input_raybundle_tensor.xyz.shape[0]
        raybundle_sequence = [self._input_raybundle_tensor]
        for which_step in range(max_steps):
            # Get the current raybundle:
            rb = raybundle_sequence[-1]
            z_min  = rb.xyz[:, 2].min()
            if (z_min - z_f)*dz > 0: # Trying to account for z_f < z_i
                break
            raybundle_sequence.append(self._step_rays(rb, acceleration, dt))
        else:
            print("After %d steps, not all rays have reached z=%0.2f"%(
                max_steps, z_f))
            print("If you need to propagate for longer, increase `max_steps`")
        # Resample our trajectories onto the z-positions of our
        # coordinate grid. I'm being lazy about this, because there
        # aren't great interpolation functions in pytorch.
        #  Variables sample uniformly in t:
        xyz_vs_t   = torch.stack([rb.xyz   for rb in raybundle_sequence], dim=1)
        v_xyz_vs_t = torch.stack([rb.v_xyz for rb in raybundle_sequence], dim=1)
        #  Variables sampled uniformly in z:
        c_z = torch.from_numpy(z).to(xyz_vs_t.device)
        c_z = c_z.reshape(1, nz, 1).broadcast_to((num_rays, nz, 3))
        z_vs_t = xyz_vs_t[:, :, 2:3].broadcast_to(xyz_vs_t.shape)
        xyz_vs_z   = interp(x=c_z, xp=z_vs_t, fp=xyz_vs_t,   dim=1)
        v_xyz_vs_z = interp(x=c_z, xp=z_vs_t, fp=v_xyz_vs_t, dim=1)
        # Convert per-z positions and velocities back into an array of RayBundle objects
        raybundle_sequence_tensor = [
            RayBundle(xyz_vs_z[:, i, :], v_xyz_vs_z[:, i, :])
            for i in range(nz)
        ]
        self._calculated_raybundle_sequence_tensor = raybundle_sequence_tensor

    def _calculate_loss(self):
        """How well does our calculated raybundle match our desired raybundle?
        """
        self._require('desired_output_raybundle', 'set_2d_desired_output_raybundle')
        self._require('_calculated_raybundle_sequence_tensor', '_calculate_3d_raybundle_sequence')
        # We want gradients, so we'll calculate our loss using Torch:
        desired_output_raybundle = self._to_torch_raybundle(self.desired_output_raybundle)
        calculated_output_raybundle = self._calculated_raybundle_sequence_tensor[-1]
        # Error vectors:
        error_xyz = calculated_output_raybundle.xyz - desired_output_raybundle.xyz
        error_v_xyz = calculated_output_raybundle.v_xyz - desired_output_raybundle.v_xyz
        # Calculate loss to be sum of euclidean distance for both the final ray position and
        # velocity vectors. Euclidean distance was chosen over squared euclidean distance so
        # that convergence does not slow down over time, and to improve stability for far rays.
        loss_xyz = (error_xyz**2).sum(dim=1).sqrt().sum() 
        loss_v_xyz = (error_v_xyz**2).sum(dim=1).sqrt().sum()
        # Note that we may want to add an optical path delay term and loss term weight hyperparameters
        # in the future.
        # Save our results as attributes, not return values:
        self._loss_tensor = loss_xyz + loss_v_xyz
        self.loss = self._to_numpy(self._loss_tensor)
        return None

    def _calculate_gradient(self):
        """How might we change `composition` in order to improve `loss`?
        """
        self._require('_composition_tensor', '_calculate_3d_raybundle_sequence')
        self._require('_loss_tensor', '_calculate_loss')
        composition_tensor = self._composition_tensor
        # Zero out any existing gradients:
        if composition_tensor.grad is not None:
            composition_tensor.grad.zero_()
        self._loss_tensor.backward()
        self._gradient_tensor = composition_tensor.grad
        return None

    def _composition_to_refractive_index(self, composition):
        """Convert our `composition` tensor to refractive index at each voxel.

        The propagation simulation wants to know the index at each voxel
        due to our refractive optic, which depends on the `composition`
        at each voxel, and the materials that we're mixing.
        """
        # `composition` must be a tensor, to allow autograd:
        assert isinstance(composition, torch.Tensor)

        # The index of refraction is a weighted average of our
        # materials. For now, we only implement binary mixtures:
        concentration = _to_concentration(composition)
        self.index_list = [m.get_index(self.wavelength)
                           for m in self.material_list]
        assert len(self.index_list) == 2
        index_1, index_2 = self.index_list
        refractive_index = index_1 + (index_2 - index_1)*concentration
                
        return refractive_index # A pytorch Tensor (which allows autograd)
    
    def _composition_to_acceleration(self, composition):
        """Convert our `composition` tensor  to ray 'acceleration' at each voxel.
        """
        # `composition` must be a tensor, to allow autograd:
        assert isinstance(composition, torch.Tensor) 
        
        coords = self.coordinates
        dx, dy, dz = coords.d_xyz
        
        # The acceleration is proportional to (n - n_base):
        refractive_index = self._composition_to_refractive_index(composition)
        
        dn_dxyz = torch.stack(torch.gradient(refractive_index, spacing=(dz, dy, dx)), axis=0)
        acceleration = torch.flip(refractive_index * dn_dxyz, [0])    
        
        return acceleration # A pytorch Tensor (which allows autograd)

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
            
    def _to_3d_torch(self, x):
        """Convert a 3D numpy array to a 3D torch tensor.
        """
        assert x.ndim == 3
        return torch.from_numpy(x).to(self.device)
    
    def _to_torch_raybundle(self, raybundle):
        """Convert a RayBundle with numpy arrays to one with torch tensors.
        """
        xyz_tensor   = self._to_torch(raybundle.xyz)
        v_xyz_tensor = self._to_torch(raybundle.v_xyz)
        return RayBundle(xyz_tensor, v_xyz_tensor)

    def _to_numpy(self, x):
        if isinstance(x, torch.Tensor): # Convert directly to an array
            assert x.ndim in (0, 1, 2, 3)
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
            
    def _to_numpy_raybundle(self, raybundle):
        """Convert a RayBundle with torch tensors to one with numpy arrays.
        """
        if isinstance(raybundle, list):
            return [self._to_numpy_raybundle(rb) for rb in raybundle]
        xyz_array   = self._to_numpy(raybundle.xyz)
        v_xyz_array = self._to_numpy(raybundle.v_xyz)
        return RayBundle(xyz_array, v_xyz_array)

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

def interp(x, xp, fp, dim=-1):
    """Linear interpolation of one dimension of an n-dimensional tensor.

    See github.com/pytorch/pytorch/issues/50334#issuecomment-2304751532
    """
    # Move the interpolation dimension to the last axis
    x  =  x.movedim(dim, -1).contiguous()
    xp = xp.movedim(dim, -1).contiguous()
    fp = fp.movedim(dim, -1).contiguous()
    # Calculate slopes and offsets:
    m = torch.diff(fp) / torch.diff(xp) # slope
    b = fp[..., :-1] - m*xp[..., :-1] # offset
    indices = torch.searchsorted(xp, x, right=False)
    # Pad m and b to get constant values outside of xp range
    m = torch.cat([torch.zeros_like(m)[..., :1], m,
                   torch.zeros_like(m)[..., :1]], dim=-1)
    b = torch.cat([fp[..., :1], b, fp[..., -1:]], dim=-1)

    values = x*m.gather(-1, indices) + b.gather(-1, indices)
    return values.movedim(-1, dim)

def smooth_3d(a, sigma=5):
    """Smooth a 3D torch tensor via convolution with a small Gaussian kernel

    Similar to scipy.ndimage.gaussian_filter(), but (potentially)faster via GPU.
    """
    assert a.ndim == 3
    assert isinstance(a, torch.Tensor)
    # Local nicknames:
    arange, exp, conv3d = torch.arange, torch.exp, torch.nn.functional.conv3d
    # Make a gaussian kernel:
    radius = int(4*sigma + 0.5)
    x = arange(-radius, radius+1, dtype=a.dtype, device=a.device)
    gaussian = exp((-0.5/sigma**2) * x**2)
    gaussian = gaussian / gaussian.sum()
    # Use pytorch for 3d convolution.
    # The shapes that conv3d expects are all minibatch-silly:
    s0, s1 = a.shape, (1, 1) + a.shape
    for s2 in ((1, 1, 1, 1, len(x)), (1, 1, 1, len(x), 1), (1, 1, len(x), 1, 1)):
        # We use a trio of 3D convolutions with a trio of 1D kernels. Wheee!
        a = conv3d(a.view(s1), gaussian.view(s2), padding='same')
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
    
class RayBundle:
    def __init__(self, xyz, v_xyz):
        assert xyz.ndim     == 2
        assert xyz.shape[1] == 3
        assert v_xyz.shape  == xyz.shape
        self.xyz   = xyz
        self.v_xyz = v_xyz
        return None

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
    if hasattr(x, 'cpu'):
        x = x.cpu()
    x = np.asarray(x).real.astype('float32')
    if x.ndim == 3:
        x = np.expand_dims(x, axis=(0, 2))
    tf.imwrite(output_directory() / filename, x, imagej=True)

def from_tif(filename):
    import tifffile as tf
    return tf.imread(output_directory() / filename)

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

def show_ray_diagram(coords, raybundle_array, filename):
    assert isinstance(raybundle_array, list) and len(raybundle_array) > 0
    assert all(isinstance(rb, RayBundle) for rb in raybundle_array)
    assert isinstance(raybundle_array[0].xyz, np.ndarray)
    x_i, x_f = coords.x_i, coords.x_f
    z_i, z_f = coords.z_i, coords.z_f
    Dx, Dz = x_f-x_i, z_f-z_i
    extra_margin_x = 0.1
    extra_margin_z = 0.15
    import matplotlib.pyplot as plt
    # Extract xyz positions from raybundle_array into shape (n_rays, n_z, 3)
    xyz_vs_z = np.stack([rb.xyz for rb in raybundle_array], axis=1)
    fig = plt.figure()
    num_rays = xyz_vs_z.shape[0]
    for which_ray in range(0, num_rays, 1+int(num_rays/100)):
        z = xyz_vs_z[which_ray, :, 2]
        x = xyz_vs_z[which_ray, :, 0]
        plt.plot(z, x, '.-')
    plt.grid('on')
    plt.xlim((z_i-Dz*extra_margin_z, z_f+Dz*extra_margin_z))
    plt.ylim((x_i-Dx*extra_margin_x, x_f+Dx*extra_margin_x))
    plt.axvline(x=z_i, color='r', linestyle='--', linewidth=1)
    plt.axvline(x=z_f, color='r', linestyle='--', linewidth=1)
    plt.savefig(output_directory() / filename)
    plt.close(fig)

def show_ray_bundle(coords, raybundle, filename):
    assert isinstance(coords, Coordinates)
    assert isinstance(raybundle, RayBundle)
    assert isinstance(raybundle.xyz, np.ndarray)
    x_i, x_f = coords.x_i, coords.x_f
    z_i, z_f = coords.z_i, coords.z_f
    Dx, Dz = x_f-x_i, z_f-z_i
    extra_margin_x = 0.1
    extra_margin_z = 0.15
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.quiver(raybundle.xyz[:,2], raybundle.xyz[:,0], raybundle.v_xyz[:,2], raybundle.v_xyz[:,0], range(raybundle.xyz.shape[0]), angles='xy', units='height', width=0.005)
    plt.grid('on')
    plt.xlim((z_i-Dz*extra_margin_z, z_f+Dz*extra_margin_z))
    plt.ylim((x_i-Dx*extra_margin_x, x_f+Dx*extra_margin_x))
    plt.axvline(x=z_i, color='r', linestyle='--', linewidth=1)
    plt.axvline(x=z_f, color='r', linestyle='--', linewidth=1)
    plt.savefig(output_directory() / filename)
    plt.close(fig)

class TrainingData_for_2dImaging:
    """An example of how to generate training data for an imaging optic.

    This generates input/output pairs that image a pointlike source at
    the 2d input plane to an inverted (but otherwise identical) image of
    the input plane to the output plane.
    """
    def __init__(self, coordinates, radius):
        assert isinstance(coordinates, Coordinates)
        self.coordinates = coordinates
        assert radius >= 0
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
        num_thetas,
        num_phis,
        divergence_angle_degrees,
        ):
        x0, y0 = float(x0), float(y0)
        wavelength = float(wavelength)
        zi, zf = self.coordinates.z_i, self.coordinates.z_f
        num_rays = num_thetas * num_phis
        # Input beam is a focused point:
        xyz = np.zeros((num_rays, 3))
        xyz[:, 0] = x0
        xyz[:, 1] = y0
        xyz[:, 2] = zi
        #   Direction:
        v_xyz = np.zeros((num_rays, 3))
        # Higher density sampling for higher angles to provide uniform sampling by area
        theta = np.deg2rad(divergence_angle_degrees*np.sqrt(np.random.random_sample(size=num_thetas)))
        phi = np.deg2rad(np.random.uniform(0, 360, num_phis))
        theta, phi = np.meshgrid(theta, phi, indexing='ij')
        theta = theta.ravel()
        phi = phi.ravel()
        v_xyz[:, 0] = np.sin(theta) * np.cos(phi)
        v_xyz[:, 1] = np.sin(theta) * np.sin(phi)
        v_xyz[:, 2] = np.cos(theta)
        input_raybundle = RayBundle(xyz, v_xyz)
        # Desired output ray bundle is an inverted image of the same point:
        out_xyz = np.zeros((num_rays, 3))
        out_xyz[:, 0] = -x0
        out_xyz[:, 1] = -y0
        out_xyz[:, 2] = zf
        out_v_xyz = np.zeros((num_rays, 3))
        out_v_xyz[:, 0] = -v_xyz[:, 0]
        out_v_xyz[:, 1] = -v_xyz[:, 1]
        out_v_xyz[:, 2] = v_xyz[:, 2]
        desired_output_raybundle = RayBundle(
            xyz=np.copy(out_xyz), 
            v_xyz=np.copy(out_v_xyz))
        return input_raybundle, desired_output_raybundle

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
