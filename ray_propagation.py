import numpy as np
import torch # For calculating gradients
import torch.nn.functional

"""
v0.0.2

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
## DO NOT MODIFY THIS MODULE. Show some couth. Run this module, modify
## the resulting example code, import this module.
##
##############################################################################

example_of_usage = """import time
import numpy as np
from ray_propagation import (
    Coordinates, Refractive3dOptic, FixedIndexMaterial,
    TrainingData_for_2dImaging, from_tif, to_tif,
    plot_loss_history, plot_simple_ray_diagram)

def example_of_usage():
    \"""Example code: design a 3D refractive optic with specified input/output.

    Consider copy-pasting this example code to get you started.

    In this example, the input/output is simple plane-to-plane imaging
    (with inversion). This is the same input-output you'd expect from a
    pair of ideal lenses which are cofocal and coaxial.

    We start with some (suboptimal) 3D refractive optic, and we generate
    "training data": bundles of rays that represent the position and
    direction of optical inputs to our 3D optic. For each input, we
    specify the output that we WISH our optic would deliver, and then
    calculate the output it ACTUALLY delivers, for our current 3D
    refractive optic. We use the difference between desired and
    calculated output to calculate our "loss", and use gradients of this
    loss to update our 3D refractive optic.
    \"""

    # Specify our coordinate system, organized via a Coordinates object:
    coords = Coordinates(xyz_i=(-10.0, -10.0,   0.0),
                         xyz_f=(+10.0, +10.0, +20.0),
                         n_xyz=(  101,   101,   101))
    print("Voxel size: %0.3f, %0.3f, %0.3f"%(coords.dx, coords.dy, coords.dz))

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
    data_source = TrainingData_for_2dImaging(
        coords, radius=3, max_z_angle_deg=15, x0=0, y0=0, num_rays=1000)
    loss_history = []
    for iteration in range(int(1e6)): # Run for a loooong time
        start_time = time.perf_counter()
        
        # Use our data source to generate random input/output pairs:
        wavelength = 0.5
        input_rays, desired_output_rays = (
            data_source.input_output_pair(wavelength))
        ro.set_input_raybundle(input_rays)
        ro.set_desired_output_raybundle(desired_output_rays)

        # Simulate propagation through our 3D refractive optic,
        # calculate loss, and calculate a gradient that hopefully will
        # reduce the loss:
        ro.gradient_update(
            dt=0.1,
            step_size=10,
            z_planes=(0, 1),
            smoothing_sigma=5)
        x0, y0 = 0, 0 # TODO: remove this cruft
        loss_history.append((x0, y0, ro.loss))

        end_time = time.perf_counter()
        print("At iteration", iteration, "the loss is %0.4f"%(ro.loss),
              "(%0.2f ms elapsed)"%(1000*(end_time - start_time)))

        # Every so often, output some intermediate state, so we can
        # monitor our progress. You can use ImageJ
        # ( https://imagej.net/ij/ ) to view the TIF files:
        if iteration % 10 == 0:
            ro.update_attributes()
            print("Saving TIFs etc...", end='')
            to_tif('00_composition.tif',          ro.composition)
            to_tif('01_concentration.tif',        ro.concentration)
            to_tif('02_concentration_xz.tif',
                   ro.concentration[:, ro.coordinates.ny//2, :])
            to_tif('03_gradient.tif', ro.gradient)
            plot_simple_ray_diagram(ro, '04_ray_diagram.png')
            plot_loss_history(loss_history, '05_loss_history.png')
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

class RayBundle:
    def __init__(self, xyz, v_xyz, wavelength_um=0.5):
        """Rays of light. Six numbers per ray: 3 for position, 3 for velocity.
        en.wikipedia.org/wiki/Ray_(optics)

        xyz and v_xyz can be either numpy arrays or torch tensors, with
        shape (3, num_rays).

        'wavelength_um' is the wavelength of the rays, in microns.

        Note that in this definition of "velocity", the "speed" of a ray
        is equal to the index of refraction at the ray's position. This
        is counterintuitive, since the speed of light is _slower_ in a
        higher index, but this "velocity" is _faster_ in a higher index.
        See equation 5 of Sharma (1982) for details:
        doi.org/10.1364/AO.21.000984
        """
        assert xyz.ndim     == 2
        assert xyz.shape[0] == 3
        assert v_xyz.shape  == xyz.shape
        self.xyz   =   xyz
        self.v_xyz = v_xyz
        self.x,  self.y,  self.z  =    xyz
        self.vx, self.vy, self.vz =  v_xyz
        self.wavelength_um = float(wavelength_um)
        assert self.wavelength_um > 0
        return None

    def to(self, array_type, device='cpu'):
        """Return a copy of this RayBundle, as either torch or numpy."""
        assert array_type in ('numpy', 'torch')
        xyz, v_xyz = self.xyz, self.v_xyz
        if hasattr(xyz, 'detach'):
            xyz, v_xyz = xyz.detach().cpu(), v_xyz.detach().cpu()
        if array_type == 'torch':
            device = torch.device(device)
            xyz   = torch.tensor(xyz  ).to(device)
            v_xyz = torch.tensor(v_xyz).to(device)
        elif array_type == 'numpy':
            assert device == 'cpu'
            xyz   = np.array(xyz,   copy=True)
            v_xyz = np.array(v_xyz, copy=True)
        return RayBundle(xyz, v_xyz, self.wavelength_um)

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
        self._invalidate(('raypaths',    '_raypaths_tensor'
                          'loss',        '_loss_tensor',
                          'gradient',    '_gradient_tensor'))
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

        Our current ray propagation model is only tested for the case
        where `concentration` is very smoothly varying, so caveat emptor.
        """
        nx, ny, nz = self.coordinates.n_xyz
        if concentration is None: # Default to a 50/50 mixture at every voxel
            concentration = np.broadcast_to(0.5, (nz, ny, nx))
        assert concentration.shape == (nz, ny, nx)
        assert np.isrealobj(concentration)
        self.concentration = concentration.astype('float64', copy=True)
        self._invalidate(('composition', '_composition_tensor',
                          'raypaths',    '_raypaths_tensor'
                          'loss',        '_loss_tensor',
                          'gradient',    '_gradient_tensor'))
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

    def set_input_raybundle(self, input_raybundle):
        """What light are we shining on our refractive optic?

        `input_raybundle` is a RayBundle object, specifying the position
        and direction of light entering our optic.

        Note that the magnitude of the "velocity" of these rays should
        match the index of refraction of the medium we're entering from;
        in the case where the medium just before our optic has index = 1,
        the magnitude of velocity should equal 1.

        The rays also have a wavelength; note that we're specifying the
        wavelength of light our input field in *vacuum*, not in our base
        material. If you're simulating dispersion using a
        SellmeierMaterial, then the units of this `wavelength` need to
        be microns.
        """
        self._require('material_list', 'set_materials')
        assert isinstance(input_raybundle, RayBundle)
        wavelength_um = input_raybundle.wavelength_um
        warning_string = ("""
    You're using a SellmeierMaterial, which expects the units of
    wavelength to be in microns, but your specified wavelength (%0.2f)
    seems to be outside the visible spectrum. Hopefully you know what
    you're doing!\n"""%(wavelength_um))
        if any([isinstance(m, SellmeierMaterial) for m in self.material_list]):
            if wavelength_um < 0.3 or 0.9 < wavelength_um:
                if not hasattr(self, '_SellmeierMaterial_warning'):
                    print(warning_string)
                    self._SellmeierMaterial_warning = True
        if not np.allclose(input_raybundle.z, self.coordinates.z_i):
            print("WARNING: your input raybundle isn't in the input z-plane.")
            print("Hopefully you know what you're doing!")
        self.input_raybundle = input_raybundle
        self._invalidate(('desired_output_raybundle',
                          'raypaths', '_raypaths_tensor'
                          'loss',     '_loss_tensor',
                          'gradient', '_gradient_tensor'))
        return None

    def set_desired_output_raybundle(self, desired_output_raybundle):
        """What light do we wish would exit our refractive optic?

        `desired_output_raybundle` is RayBundle object specifying the
        position and direction of the light that we WISH would be
        produced at the output plane of our refractive optic. We use
        this to calculate loss (aggregate error between desired and
        calculated rays), and we take gradients of this loss to update
        our optic to (hopefully) get closer to yielding our desired
        output.
        """
        self._require('input_raybundle', 'set_input_raybundle')
        assert isinstance(desired_output_raybundle, RayBundle)
        assert (desired_output_raybundle.xyz.shape ==
                    self.input_raybundle.xyz.shape)
        desired_output_raybundle.wavelength_um = ( # Float equality is annoying!
            self.input_raybundle.wavelength_um)    # Just force them equal.
        self.desired_output_raybundle = desired_output_raybundle
        self._invalidate(('loss', '_loss_tensor',
                          'gradient', '_gradient_tensor'))
        return None

    def gradient_update(
        self,
        dt,
        step_size,
        z_planes=(0, 1),
        smoothing_sigma=5
        ):
        """Update our optic to get closer to our desired behavior.

        This is multiple steps rolled into one:
         * Calculate light propagation through our refractive optic
           (with time step `dt`)
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
        self._calculate_3d_propagation(dt=dt)
        self._calculate_loss(z_planes=z_planes)
        self._calculate_gradient()
        # The gradient usually has high-spatial-frequency content that
        # isn't desirable or manufacturable, so we update our refractive
        # optic with a scaled, smoothed version of the gradient:
        update = step_size * smooth_3d(self._gradient_tensor,
                                       sigma=smoothing_sigma)
        self._composition_tensor.requires_grad_(False)
        self._composition_tensor.subtract_(update)

        self._invalidate(('concentration', 'composition',
                          'raypaths', 'gradient'))
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
        if hasattr(self, '_composition_tensor'):
            self.composition = self._to_numpy(self._composition_tensor)
            if delete_tensors: del self._composition_tensor
            self.concentration = _to_concentration(self.composition)
        if hasattr(self, '_raypaths_tensor'):
            self.raypaths = [rb.to('numpy') for rb in self._raypaths_tensor]
            if delete_tensors: del self._raypaths_tensor
        if hasattr(self, '_gradient_tensor'):
            self.gradient = self._to_numpy(self._gradient_tensor)
            if delete_tensors: del self._gradient_tensor
        return None

    def _calculate_3d_propagation(self, dt=None):
        """Hybrid ray tracing: an RK method for accurate forward
        propagation, combined with an Euler method for efficient
        backpropagation.

        We store our refractive object on a 3D grid, and use
        interpolation to define the index of refraction between grid
        points. How shall we simulate ray propagation through this
        object? We'd like an accurate method, but every voxel of the
        object that we "touch" during each step of forward propagation
        adds to the computational graph.

        The Runge-Kutta method from Sharma (1982):
        doi.org/10.1364/AO.21.000984
        ...with trilinear interpolation and constant time step `dt`,
        gives fast, accurate, simple ray tracing in gradient-index
        refractive objects.
        
        Unfortunately, this doesn't combine well with the pytorch memory
        model. The constant time step means bundles of rays spread out
        in z, and we end up adding a large 3D chunk of the object to the
        computational graph for every time step. We also use many small
        time steps for high accuracy, so backpropagation is VERY
        expensive.

        Euler's method:
        https://en.wikipedia.org/wiki/Euler_method
        ...with nearest-neighbor interpolation and constant z-step size,
        gives lower accuracy, and the resulting error builds to
        intolerable levesl over many steps.

        However, this combines extremely well with the pytorch memory
        model. The constant z-step means bundles of rays stay in the
        same z-plane, so we add one 2D slice of the object to the
        computational graph for every z-step. We use also use large
        z-steps (equal to the z-spacing of our object's 3D grid), so
        backpropagation is VERY cheap.

        Here, we get the best of both worlds: the accuracy of RK, and
        the efficient backprop of Euler:
         * Turn off autograd and compute accurate ray paths with RK.
         * Turn on autograd and compute differentiable ray paths with Euler.
         * After each Euler z-step, we add a small non-differentiable
           correction based on the RK result. The resulting ray path is
           given by RK, but the differentiability is given by Euler.
         * Since our ray paths are accurate, the resulting gradients are
           nearly identical to the gradients given by backprop through
           RK, but MUCH cheaper to compute.

        I'm pretty happy with this algorithm. I think it's awesome.
        """
        try:
            self._require('_composition_tensor', 'set_3d_concentration')
        except AttributeError:
            self._require('concentration', 'set_3d_concentration')
        self._require('input_raybundle', 'set_input_raybundle')
        self._require('material_list', 'set_materials')
        # All the internal work of this function is done in torch. Convert:
        c = self.coordinates.to('torch', self.device)
        input_raybundle = self.input_raybundle.to('torch', self.device)
        if not hasattr(self, '_composition_tensor'):
            self._composition_tensor = _to_composition(
                self._to_torch(self.concentration))
        if dt is None: # The time step size for the RK method.
            dt = c.dz/5
        assert dt > 0
        #  Calculate the optical "acceleration" due to our object:
        #   composition -> concentration -> index -> gradient -> acceleration
        # We ultimately want to update the 'composition', so track gradients:
        self._composition_tensor.requires_grad_(True)
        concentration = _to_concentration(self._composition_tensor)
        n = self._concentration_to_refractive_index(concentration)
        dx, dy, dz = map(float, c.d_xyz) # torch.gradient is silly
        grad_z, grad_y, grad_x = torch.gradient(n, spacing=(dz, dy, dx))
        a_x, a_y, a_z = (n*grad_x, n*grad_y, n*grad_z)
        del concentration, grad_x, grad_y, grad_z, dx, dy, dz
        # The 'speed' of our input RayBundle must equal the index of
        # refraction that it starts in (see `set_input_raybundle()` for
        # details). Keep the *direction* of our input ray velocities,
        # but force the *magnitude* of our input ray velocities to equal
        # the local index of refraction:
        initial_index = sample_3d_grid_data_at_xyz(input_raybundle.xyz,
                                                   n.detach(), c)
        initial_speed = torch.sqrt((input_raybundle.v_xyz**2).sum(axis=0))
        input_raybundle.v_xyz *= initial_index / initial_speed        
        del n
        # We'll start by running a Runge-Kutta raytrace WITHOUT
        # automatic differentiation:
        rt = SharmaRaytracer(a_x.detach(), a_y.detach(), a_z.detach(), c)
        xyz_vs_z_RK, v_xyz_vs_z_RK = rt.propagate_rays(input_raybundle, dt)
        del rt
        # Now run an Euler's method raytrace WITH automatic differentiation:
        a_x = torch.unbind(a_x, dim=0)
        a_y = torch.unbind(a_y, dim=0) # We only touch one 2D slice at a time
        a_z = torch.unbind(a_z, dim=0)
        raypaths = [input_raybundle]
        for which_z in range(c.nz - 1): # Input/output planes are voxel centers
            rb = raypaths[-1]
            # Simple nearest-neighbor interpolation to get acceleration:
            x_scaled = (rb.x - c.x_i_edges)*(1/c.dx)
            y_scaled = (rb.y - c.y_i_edges)*(1/c.dy)
            which_x = torch.clip(x_scaled, 0, c.nx-1).to(torch.int32)
            which_y = torch.clip(y_scaled, 0, c.ny-1).to(torch.int32)
            a_x_in = a_x[which_z][which_y, which_x]
            a_y_in = a_y[which_z][which_y, which_x] # Interpolated values
            a_z_in = a_z[which_z][which_y, which_x]
            a_xyz_i = torch.stack((a_x_in, a_y_in, a_z_in), dim=0)
            # Calculate the position/velocity update for Euler's method:
            dt = c.dz / rb.vz # Variable t-step to give constant z-step = dz.
            xyz_f   = rb.xyz   + dt * rb.v_xyz
            v_xyz_f = rb.v_xyz + dt * a_xyz_i
            # Euler's method with crappy interpolation is simple but not
            # accurate. Use our (without-autodiff) Runge-Kutta raytrace
            # to apply a (hopefully small) correction factor to our
            # (with-autodiff) Euler raytrace:
            xyz_error   =   xyz_f.detach() -   xyz_vs_z_RK[which_z+1, :, :]
            v_xyz_error = v_xyz_f.detach() - v_xyz_vs_z_RK[which_z+1, :, :]
            xyz_f.subtract_(    xyz_error)
            v_xyz_f.subtract_(v_xyz_error)
            rb = RayBundle(xyz_f, v_xyz_f, wavelength_um=rb.wavelength_um)
            raypaths.append(rb)
        self._raypaths_tensor = raypaths
        self._invalidate(('raypaths', 'loss', '_loss_tensor',
                          'gradient', '_gradient_tensor'))
        return raypaths

    def _calculate_loss(self, z_planes=(0, 1)):
        """How close is our calculated output to our desired output?

        `z_planes` is a list of z-offsets (in the same units as our
        `Coordinates` object, relative to the output plane of our 3D
        refractive optic) at which we'll calculate the mismatch between
        our calculated rays and our desired rays.
        """
        self._require('desired_output_raybundle',
                      'set_desired_output_raybundle')
        self._require('_raypaths_tensor', '_calculate_3d_propagation')
        desired_rays = self.desired_output_raybundle.to('torch', self.device)
        calculated_rays = self._raypaths_tensor[-1] # The output raybundle
        def z_propagate(rb, z): # Propagate a raybundle to a given z-plane
            if z == 0: return rb.x, rb.y # Don't bother
            dt = (z - rb.z) / rb.vz # Each ray takes a different amount of time
            return rb.x + dt*rb.vx, rb.y + dt*rb.vy # x_final, y_final
        loss = torch.zeros(1, device=self.device, dtype=torch.float64)
        for zf in z_planes:
            xf_d, yf_d = z_propagate(   desired_rays, zf)
            xf_c, yf_c = z_propagate(calculated_rays, zf)
            distance_sq = (xf_d - xf_c)**2 + (yf_d - yf_c)**2
            loss += torch.mean(torch.sqrt(distance_sq))
        loss = loss / len(z_planes)
        # Save our results as attributes, not return values:
        self._loss_tensor = loss
        self.loss = self._to_numpy(loss)[0]
        self._invalidate(('gradient', '_gradient_tensor'))
        return None

    def _calculate_gradient(self):
        """How might we change `composition` in order to improve `loss`?
        """
        self._require('_composition_tensor', '_calculate_3d_propagation')
        self._require('_loss_tensor', '_calculate_loss')
        if self._composition_tensor.grad is not None:
            self._composition_tensor.grad.zero_()
        self._loss_tensor.backward() # This step is very expensive.
        self._gradient_tensor = self._composition_tensor.grad
        return None

    def _concentration_to_refractive_index(self, concentration):
        """Convert our `concentration` to refractive index at each voxel.

        The propagation simulation wants to know the index at each voxel
        due to our refractive optic, which depends on the `concentration`
        at each voxel, and the materials that we're mixing.
        """
        # The index of refraction is a weighted average of our
        # materials. For now, we only implement binary mixtures:
        assert len(self.material_list) == 2
        index_1, index_2 = [m.get_index(self.input_raybundle.wavelength_um)
                            for m in self.material_list]
        refractive_index = index_1 + (index_2 - index_1)*concentration
        return refractive_index

    def _invalidate(self, iterable_of_attribute_names):
        # Many of the methods above need to invalidate (i.e. delete)
        # multiple attributes. This makes it a little more convenient:
        for attr in iterable_of_attribute_names:
            if hasattr(self, attr): delattr(self, attr)
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
        return torch.from_numpy(np.asarray(x)).to(self.device)

    def _to_numpy(self, x):
        return x.cpu().detach().numpy()

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

def smooth_3d(a, sigma=5):
    """Smooth a 3D torch tensor via convolution with a small Gaussian kernel

    Similar to scipy.ndimage.gaussian_filter(), but potentially faster via GPU.
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
    for s2 in ((1, 1,      1,      1, len(x)),
               (1, 1,      1, len(x),      1),
               (1, 1, len(x),      1,      1)):
        # We use 3 3D convolutions, each with a 1D kernel. Wheee!
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

class SharmaRaytracer:
    """Simple Runge-Kutta ray tracing in an inhomogenous refractive object.

    See Sharma (1982) for details of the algorithm:
    doi.org/10.1364/AO.21.000984
    """
    def __init__(self, a_x, a_y, a_z, coords):
        """`a_x`, `a_y`, and `a_z` are (Nz x Ny x Nx) arrays, specifying
        the acceleration an optical ray will experience due to a 3D
        inhomogenous refractive object. If n(x, y, z) is the 3D
        refractive index of our object, the acceleration is given by:
          a_x = d/dx (n(x, y, z)**2)
          a_y = d/dy (n(x, y, z)**2)
          a_z = d/dz (n(x, y, z)**2)
        """
        for a in (a_x, a_y, a_z):
            assert isinstance(a, torch.Tensor)
            assert a.shape == (coords.nz, coords.ny, coords.nx)
        assert a_x.device == a_y.device == a_z.device
        self.a_x, self.a_y, self.a_z = (a_x, a_y, a_z)
        # Make sure our coordinates object is a tensor on the right device:
        assert isinstance(coords, Coordinates)
        self.coordinates = coords.to('torch', device=a_x.device)
        return None
        
    def _get_acceleration(self, xyz):
        """Estimate acceleration at arbitrary xyz positions via interpolation.
        """
        a_x_i, a_y_i, a_z_i = [
            sample_3d_grid_data_at_xyz(xyz, a, self.coordinates)
            for a in (self.a_x, self.a_y, self.a_z)]
        a_xyz = torch.stack((a_x_i, a_y_i, a_z_i), dim=0)
        return a_xyz

    def _step_rays(self, raybundle, dt):
        """Step a raybundle forward by time interval dt.

        Since we're a 'private' method, we don't bother with sanity
        checks, but here's what they would be:
        assert isinstance(raybundle, Raybundle)
        assert dt > 0
        dt = float(dt)
        """
        # Shorter nicknames
        xyz, v_xyz, a = raybundle.xyz, raybundle.v_xyz, self._get_acceleration
        # Calculate A, B, C, for Sharma's algorithm:
        A = dt * a(xyz)
        B = dt * a(xyz + (1/2)*dt*v_xyz + (1/8)*dt*A)
        C = dt * a(xyz +       dt*v_xyz + (1/2)*dt*B)
        # Calculate the position/velocity update for Sharma's algorithm:
        xyz_f   = xyz   + dt*(v_xyz + (1/6)*(A + 2*B))
        v_xyz_f = v_xyz +             (1/6)*(A + 4*B + C)
        return RayBundle(xyz_f, v_xyz_f)

    def propagate_rays(self, initial_raybundle, dt, max_steps=None):
        # Input sanitization
        assert isinstance(initial_raybundle, RayBundle)
        assert initial_raybundle.xyz.device   == self.a_x.device
        assert initial_raybundle.v_xyz.device == self.a_x.device
        dt = float(dt)
        assert dt > 0
        c = self.coordinates
        if max_steps is None:
            max_steps = int(10*c.nz*c.dz/dt)

        # Call '_step_rays()' in a loop... 
        raybundle_sequence = [initial_raybundle]
        for which_step in range(max_steps):
            rb = raybundle_sequence[-1]
            z_min = rb.xyz[2, :].min()
            # ...until our rays pass z = z_f:
            if (z_min - c.z_f)*c.dz > 0: # Trying to account for z_f < z_i
                break
            next_rb = self._step_rays(rb, dt)
            raybundle_sequence.append(next_rb)
        else:
            print("After %d steps, not all rays have reached z=%0.2f"%(
                max_steps, self.coordinates.z_f))
            print("If you need to propagate for longer, increase `max_steps`")
            raise TimeoutError("Maximum steps exceeded")

        # Resample our trajectories onto the z-positions of our
        # coordinate grid. I'm being lazy about this, because there
        # aren't great interpolation functions in pytorch.
        #  Variables sample uniformly in t:
        xyz_vs_t   = torch.stack([rb.xyz   for rb in raybundle_sequence], dim=0)
        v_xyz_vs_t = torch.stack([rb.v_xyz for rb in raybundle_sequence], dim=0)
        #  Variables sampled uniformly in z:
        c_z = c.z.reshape(c.nz, 1, 1).broadcast_to((c.nz,) + rb.xyz.shape)
        z_vs_t = xyz_vs_t[:, 2:3, :].broadcast_to(xyz_vs_t.shape)
        xyz_vs_z   = sample_1d_irregular_data(c_z, z_vs_t,   xyz_vs_t, dim=0)
        v_xyz_vs_z = sample_1d_irregular_data(c_z, z_vs_t, v_xyz_vs_t, dim=0)
        return xyz_vs_z, v_xyz_vs_z

def sample_1d_irregular_data(new_positions, old_positions, old_values, dim=-1):
    """Estimate values at arbitrary 1D positions via interpolation

    Suppose we know the values of some 1D function sampled at known
    positions in 1D. These values are stored in the 1D tensor
    `old_values`, and their positions are stored in the 1D tensor
    `old_positions`. We'd *like* to know values of this 1D function at
    intermediate positions stored in the 1D tensor `new_positions`.

    See github.com/pytorch/pytorch/issues/50334#issuecomment-2304751532
    """
    # TODO: Is this actually any good? Can we do a lot better?
    # Move the interpolation dimension to the last axis
    x  =  new_positions.movedim(dim, -1).contiguous()
    xp =  old_positions.movedim(dim, -1).contiguous()
    fp =     old_values.movedim(dim, -1).contiguous()
    # Calculate slopes and offsets:
    m = torch.diff(fp) / torch.diff(xp) # slope
    b = fp[..., :-1] - m*xp[..., :-1] # offset
    indices = torch.searchsorted(xp, x, right=False)
    # Pad m and b to get constant values outside of xp range
    m = torch.cat([torch.zeros_like(m)[..., :1], m,
                   torch.zeros_like(m)[..., :1]], dim=-1)
    b = torch.cat([fp[..., :1], b, fp[..., -1:]], dim=-1)

    new_values = x*m.gather(-1, indices) + b.gather(-1, indices)
    return new_values.movedim(-1, dim)

def sample_3d_grid_data_at_xyz(xyz, data, data_coordinates):
    """Estimate values at arbitrary 3D positions via interpolation.

    Suppose we know the values of some 3D function sampled at points on
    a regular 3D grid. These values are stored in the 3D tensor `a`, and
    the coordinates are stored in the `Coordinates' object `a_coordinates`.
    We'd *like* to know values of this 3D function at N intermediate
    points, given by the coordinates stored in the (3, N) tensor `xyz`.
    """
    assert isinstance(xyz,  torch.Tensor)
    assert isinstance(data, torch.Tensor)
    assert isinstance(data_coordinates, Coordinates)
    assert xyz.device == data.device == data_coordinates.device
    assert xyz.ndim == 2
    assert xyz.shape[0] == 3
    c, num_points = data_coordinates, xyz.shape[1]
    # The 3D interpolation routine in torch wants xyz scaled to the
    # range (-1, 1). This is a little silly, but whatever:
    center_point = 0.5*(c.xyz_f + c.xyz_i).reshape(3, 1)
    radius       = 0.5*(c.xyz_f - c.xyz_i).reshape(3, 1)
    xyz_scaled = (xyz - center_point) * (1/radius)
    # The 3D interpolation routine in torch wants `xyz` and `a` to have
    # a few extra dimensions that we're not using. This is a little
    # silly, but whatever.
    # (3, num_rays) -> (1, 1, 1, num_points, 3)
    xyz_scaled = xyz_scaled.reshape(3, 1, 1, num_points, 1).transpose(0, 4)
    # (nz, ny, nx) -> (1, 1, nz, ny, nx)
    data = data.reshape((1, 1, c.nz, c.ny, c.nx))
    # Now we can interpolate, and then strip off the extra dimensions:
    data_estimated_at_xyz = torch.nn.functional.grid_sample(
        data, xyz_scaled, mode='bilinear', align_corners=True
        ).reshape(num_points)
    return data_estimated_at_xyz

class Coordinates:
    """A convenience class for keeping track of the coordinates of our voxels.

     - xyz_i: a 3-element tuple storing the coordinates of our initial voxel
     - xyz_f: a 3-element tuple storing the coordinates of our final voxel
     - n_xyz: a 3-element tuple specifying how many voxels we have in x, y, z

    There's nothing complicated here, but this is the type of detail I
    tend to get wrong if I don't make a convenience class to organize it.
    """
    def __init__(self, xyz_i, xyz_f, n_xyz, array_type='numpy', device='cpu'):
        # Sanitize and organize...
        # - Inputs:
        xi, yi, zi = map(float, xyz_i)
        xf, yf, zf = map(float, xyz_f)
        nx, ny, nz = map(int,   n_xyz)
        assert array_type in ('numpy', 'torch')
        self.array_type = array_type
        self.device = torch.device(device)
        if array_type == 'numpy':
            linspace = np.linspace
            def as_floats(a): return np.asarray(a, dtype='float64')
            def as_ints(a):   return np.asarray(a, dtype='int32')
        elif array_type == 'torch':
            def linspace(start, stop, num):
                return torch.linspace(start, stop, num,
                                      dtype=torch.float64, device=device)
            def as_floats(a):
                return torch.as_tensor(a, dtype=torch.float64, device=device)
            def as_ints(a):
                return torch.as_tensor(a, dtype=torch.int32,   device=device)
            self.device = linspace(0, 1, 2).device
        # - Initial and final voxel positions:
        self.xyz_i = as_floats(xyz_i)
        self.xyz_f = as_floats(xyz_f)
        self.x_i, self.y_i, self.z_i = self.xyz_i
        self.x_f, self.y_f, self.z_f = self.xyz_f
        # - Position of voxels:
        self.xyz = (linspace(xi, xf, nx).reshape( 1,  1, nx),
                    linspace(yi, yf, ny).reshape( 1, ny,  1),
                    linspace(zi, zf, nz).reshape(nz,  1,  1))
        self.x, self.y, self.z = self.xyz
        # - Shape of voxels:
        dx, dy, dz = (xf-xi)/(nx-1), (yf-yi)/(ny-1), (zf-zi)/(nz-1)
        self.d_xyz = as_floats((dx, dy, dz))
        self.dx, self.dy, self.dz = self.d_xyz
        # - Number of voxels:
        self.n_xyz = as_ints(n_xyz)
        self.nx, self.ny, self.nz = self.n_xyz
        # So far, we've been referring to the positions of the *centers*
        # of our voxels. We often want to know the position of the *edges*
        # of our voxels, so let's cover that, too:
        #
        # - Initial and final voxel edge positions:
        self.xyz_i_edges = as_floats((xi-dx/2, yi-dy/2, zi-dz/2))
        self.xyz_f_edges = as_floats((xf+dx/2, yf+dy/2, zf+dz/2))
        self.x_i_edges, self.y_i_edges, self.z_i_edges = self.xyz_i_edges
        self.x_f_edges, self.y_f_edges, self.z_f_edges = self.xyz_f_edges
        # - Position of voxel edges:
        self.xyz_edges = (
            linspace(xi-dx/2, xf+dx/2, nx+1).reshape(1, 1, nx+1),
            linspace(yi-dy/2, yf+dy/2, ny+1).reshape(1, ny+1, 1),
            linspace(zi-dz/2, zf+dz/2, nz+1).reshape(nz+1, 1, 1))
        self.x_edges, self.y_edges, self.z_edges = self.xyz_edges
        # - Number of edges:
        self.n_xyz_edges = as_ints((nx+1, ny+1, nz+1))
        self.nx_edges, self.ny_edges, self.nz_edges = self.n_xyz_edges
        return None

    def to(self, array_type, device='cpu'):
        if self.array_type == array_type:
            if self.device == device:
                return self
        return Coordinates(self.xyz_i, self.xyz_f, self.n_xyz,
                           array_type=array_type, device=device)

        
##class RefractiveOpticSequence:
##    def __init__(self, optics_list):
##        """Multiple Refractive3dOptics in a row.
##
##        The output field of the Nth Refractive3dOptic is the input field
##        to the N+1th Refractive3dOptic.
##
##        For example, you might want to simulate a big air gap in between
##        two 3D printed optics:
##        
##        air = FixedIndexMaterial(1)
##        polymer = FixedIndexMaterial(1.5)
##        coords = Coordinates(xyz_i=(-12.7, -12.7, -12.7),
##                             xyz_f=(+12.7, +12.7, +12.7),
##                             n_xyz=(  128,   128,   128))
##
##        printed_optic_1 = Refractive3dOptic(coords)
##        printed_optic_1.set_materials((air, polymer))
##
##        air_gap = Refractive3dOptic(coords)
##        air_gap.set_materials((air, air))
##        air_gap.allow_gradient_update = False
##
##        printed_optic_2 = Refractive3dOptic(coords)
##        printed_optic_2.set_materials((air, polymer))
##
##        my_optics = RefractiveOpticSequence((printed_optic_1,
##                                             air_gap,
##                                             printed_optic_2))
##        """
##        assert len(optics_list) > 0
##        for o in optics_list:
##            assert isinstance(o, Refractive3dOptic)
##            if not hasattr(o, 'allow_gradient_update'):
##                o.allow_gradient_update = True
##            assert o.allow_gradient_update in (True, False)
##        for o1, o2 in zip(optics_list[ :-1], optics_list[1:  ]):
##            # We don't check that the z-coordinates of our consecutive
##            # optics are consistent, but be thoughtful about the fact
##            # that the z-positions of the refractive voxels and the
##            # z-positions of the calculated field have a "fencepost"
##            # relationship:
##            #
##            # https://en.wikipedia.org/wiki/Off-by-one_error#Fencepost_error
##            #
##            # We *do* check that consecutive optics share XY coordinates:
##            c1, c2 = o1.coordinates, o2.coordinates
##            assert c1.nx == c2.nx # Consecutive optics have same # of x-pixels
##            assert c1.ny == c2.ny # Consecutive optics have same # of y-pixels
##            assert np.allclose(c1.x, c2.x) # Consecutive optics share x coords
##            assert np.allclose(c1.y, c2.y) # Consecutive optics share y coords
##        self.optics = optics_list
##        return None
##
##    def disable_gradient_update(self, optic):
##        """We might not want to optimize all of the optics in our sequence.
##        """
##        assert optic in self.optics
##        optic.allow_gradient_update = False
##        return None
##
##    def set_2d_input_field(self, input_field, wavelength):
##        """See `Refractive3dOptic.set_2d_input_field()` for details
##        """
##        first_optic, last_optic = self.optics[0], self.optics[-1]
##        first_optic.set_2d_input_field(input_field, wavelength)
##        last_optic.wavelength = wavelength
##        return None
##
##    def set_2d_desired_output_field(self, desired_output_field):
##        """See `Refractive3dOptic.set_2d_desired_output_field()` for details
##        """
##        # This will get overwritten, it's just to trigger input sanitization:
##        self.optics[-1].set_2d_desired_output_field(desired_output_field)
##        # This gets remembered, though:
##        self.desired_output_field = desired_output_field
##        return None
##
##    def gradient_update(self, step_size, z_planes=(1, 2, 3), smoothing_sigma=5):
##        """See `Refractive3dOptic.gradient_update()` for details
##        """
##        assert step_size > 0
##        assert smoothing_sigma >= 0
##        step_size = float(step_size)
##        smoothing_sigma = float(smoothing_sigma)
##        z_planes = [float(z) for z in z_planes]
##        
##        pairs_of_optics = zip(self.optics[:-1], self.optics[1:])
##        final_optic = self.optics[-1]
##        try:
##            for i, (current_optic, next_optic) in enumerate(pairs_of_optics):
##                # Calculate propagation through the current optic:
##                current_optic._calculate_3d_field()
##                # Use the output field of the current optic as the input
##                # field of the next optic:
##                next_optic.set_2d_input_field(
##                    input_field=current_optic._calculated_field_tensor[-1],
##                    wavelength=current_optic.wavelength)
##            # Calculate loss just for the final optic:
##            final_optic.set_2d_desired_output_field(self.desired_output_field)
##            delattr(self, 'desired_output_field')
##            final_optic._calculate_3d_field()
##            final_optic._calculate_loss(z_planes=z_planes)
##            self.loss = np.copy(final_optic.loss)
##            # Zero the gradient for all the optics:
##            for i, o in enumerate(self.optics):
##                for c in o._composition_tensor:
##                    if c.grad is not None:
##                        c.grad.zero_()
##            # Backpropagate to populate our gradients:
##            final_optic._loss_tensor.backward()
##            for i, o in enumerate(self.optics):
##                o._gradient_tensor = [c.grad for c in o._composition_tensor]
##            # Update our optics using our gradients:
##            for i, o in enumerate(self.optics):
##                for g, c in zip(o._gradient_tensor, o._composition_tensor):
##                    c.requires_grad_(False)
##                    if o.allow_gradient_update:
##                        update = step_size*smooth_2d(g, sigma=smoothing_sigma)
##                        c.subtract_(update)
##        except:
##            print("While calculating with optic %i, an exception occured:"%(i))
##            raise
##        
##        return None
##
##    def update_attributes(self, delete_tensors=True):
##        for op in self.optics:
##            op.update_attributes(delete_tensors=delete_tensors)

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
    if x.ndim == 4:
        x = np.expand_dims(x, axis=(2,))
    tf.imwrite(output_directory() / filename, x, imagej=True)

def from_tif(filename):
    import tifffile as tf
    return tf.imread(output_directory() / filename)

##def attributes_to_tifs(refractive_optic_sequence, list_of_attributes):
##    assert isinstance(refractive_optic_sequence, RefractiveOpticSequence)
##    for n, optic in enumerate(refractive_optic_sequence.optics):
##        for i, attribute_name in enumerate(list_of_attributes):
##            if hasattr(optic, attribute_name):
##                attr = getattr(optic, attribute_name)
##                if np.iscomplexobj(attr):
##                    attr = np.abs(attr)
##                filename = "%02d_optic_%02d_%s.tif"%(
##                    i, n, attribute_name)
##                to_tif(filename, attr)

def plot_simple_ray_diagram(refractive_object, filename):
    import matplotlib as mpl
    mpl.use('agg') # Prevents a memory leak from repeated plotting
    import matplotlib.pyplot as plt

    c = refractive_object.coordinates

    raypaths = refractive_object.raypaths
    num_rays = raypaths[0].xyz.shape[1]
    max_plotted_rays = 300
    chunk_size = int(np.ceil(num_rays / max_plotted_rays))

    x_vs_t = [r.x[::chunk_size] for r in raypaths]
    z_vs_t = [r.z[::chunk_size] for r in raypaths]

    calculated_output = raypaths[-1]
    desired_output = refractive_object.desired_output_raybundle
    error_ray_x = np.stack((calculated_output.x[::chunk_size],
                               desired_output.x[::chunk_size]))
    error_ray_y = np.stack((calculated_output.y[::chunk_size],
                               desired_output.y[::chunk_size]))

    fig = plt.figure()
    fig.add_subplot(1, 2, 1, adjustable='box', aspect=1)
    plt.plot(z_vs_t, x_vs_t, '-', linewidth=0.5)
    plt.xlim(c.z_i, c.z_f)
    plt.ylim(c.x_i, c.x_f)
    plt.grid('on', alpha=0.1)
    fig.add_subplot(1, 2, 2, adjustable='box', aspect=1)
    plt.plot(error_ray_y, error_ray_x, '-', color='red', alpha=0.05)
    plt.plot(error_ray_y, error_ray_x, '.', color='red', alpha=0.5,
             markersize=2)
    plt.grid('on', alpha=0.1)
    plt.savefig(output_directory() / filename, dpi=2*fig.dpi)
    plt.close(fig)
    return None

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

    This generates input/output pairs that image rays at the 2d input
    plane to an inverted (but otherwise identical) image of the input
    plane to the output plane.
    """
    def __init__(
        self,
        coordinates,
        radius,
        max_z_angle_deg,
        x0, y0,
        num_rays,
        ):
        assert isinstance(coordinates, Coordinates)
        self.coordinates = coordinates
        assert radius > 0
        self.radius = float(radius)
        assert 0 <= max_z_angle_deg < 90
        self.max_z_angle_deg = float(max_z_angle_deg)
        self.x0, self.y0 = float(x0), float(y0)
        self.z0 = float(self.coordinates.z_i)
        assert num_rays > 0
        self.num_rays = int(num_rays)
        return None

    def random_points_in_a_circle(self):
        # Local nicknames:
        R, sin, cos, pi, sqrt = self.radius, np.sin, np.cos, np.pi, np.sqrt
        rand, n = np.random.random_sample, self.num_rays
        # Simple math:
        r, phi = R*sqrt(rand(n)), 2*pi*rand(n)
        xyz = np.empty((3, n), dtype='float64')
        xyz[0, :] = self.x0 + r*cos(phi)
        xyz[1, :] = self.y0 + r*sin(phi)
        xyz[2, :] = self.z0
        return xyz

    def random_directions_in_a_cone(self):
        th_max, pi, n = np.deg2rad(self.max_z_angle_deg), np.pi, self.num_rays
        # Point-picking on a sphere is easy, but also easy to do wrong:
        theta = np.arccos(np.random.uniform(np.cos(th_max), 1, n))
        phi   =           np.random.uniform(0,           2*pi, n)
        # Calculate our trig functions:
        sin_th, cos_th = np.sin(theta), np.cos(theta)
        sin_ph, cos_ph = np.sin(phi),   np.cos(phi)
        # Convert back to Cartesian:
        v_xyz = np.empty((3, n), dtype='float64')
        v_xyz[0, :] = sin_th * cos_ph
        v_xyz[1, :] = sin_th * sin_ph
        v_xyz[2, :] = cos_th
        return v_xyz        

    def input_output_pair(self, wavelength_um):
        c = self.coordinates
        assert wavelength_um > 0
        wavelength_um = float(wavelength_um)
        # Input beam is a focused point:
        input_raybundle = RayBundle(
            self.random_points_in_a_circle(),
            self.random_directions_in_a_cone(),
            wavelength_um)
        # Desired output is an inverted image of the same points:
        xyz = input_raybundle.xyz.copy()
        #  XY position flips, Z position is the output plane:
        xyz[0, :] = -1*xyz[0, :]
        xyz[1, :] = -1*xyz[1, :]
        xyz[2, :] = c.z_f
        #  XY velocity flips, Z velocity stays the same:
        v_xyz = input_raybundle.v_xyz.copy()
        v_xyz[0, :] = -1*v_xyz[0, :]
        v_xyz[1, :] = -1*v_xyz[1, :]
        desired_output_raybundle = RayBundle(xyz, v_xyz, wavelength_um)
        return input_raybundle, desired_output_raybundle

def random_conical_bundle(
    x=0, y=0, z=0,
    cone_angle=np.pi/32,
    num_rays=100,
    wavelength=0.5,
    ):
    """A pointlike emitter at (x, y, z), emitting in the z-direction.

    Note that 'cone_angle' is the full angle of the cone, not the half-angle.
    """
    # Point-picking on a sphere is easy, but also easy to do wrong:
    theta = np.arccos(np.random.uniform(np.cos(cone_angle/2), 1, num_rays))
    phi   =           np.random.uniform(0,              2*np.pi, num_rays)
    # Calculate our trig functions:
    sin_th, cos_th = np.sin(theta), np.cos(theta)
    sin_ph, cos_ph = np.sin(phi),   np.cos(phi)
    # Convert back to Cartesian:
    v_xyz = np.empty((3, num_rays), dtype='float64')
    v_xyz[0, :] = sin_th * cos_ph
    v_xyz[1, :] = sin_th * sin_ph
    v_xyz[2, :] = cos_th
    # Broadcast our xyz's:
    xyz = np.broadcast_to(np.array((x, y, z)).reshape(3, 1), (3, num_rays))
    return RayBundle(xyz.copy(), v_xyz, wavelength)

        
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
