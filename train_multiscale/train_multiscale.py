"""
train_multiscale.py

An N-stage training script for designing large-scale freeform 3D
refractive optics using a "full-volume interpolation" scaling strategy.

This script implements a scaling strategy that prioritizes a constant
physical pixel size across all stages. The grid resolution jumps in
powers of two, and the physical width of the optic is adjusted
accordingly to keep the pixel size consistent.

The process is as follows:
1.  **Start Small:** The first stage runs a baseline simulation to create
    a small, converged "seed" optic. This stage defines the target
    physical pixel size.
2.  **Interpolate and Stretch:** For each subsequent stage, the grid
    resolution is doubled. The physical width is increased proportionally
    to maintain the constant pixel size. The previous optic is stretched
    to fit this new grid.
3.  **Continue Training:** This stretched optic is used as the starting
    point for the next stage, allowing for refinement at a consistent
    physical resolution.

Written by Gav Sturm/Gemini
"""

import time
import numpy as np
from scipy.ndimage import zoom
from pathlib import Path # Import Path for modern path handling
import re

# Prefer the project-root beam_propagation.py; fall back to local copy if needed
try:
    from beam_propagation import (
        Coordinates, Refractive3dOptic, FixedIndexMaterial,
        from_tif, to_tif, plot_loss_history, gaussian_beam_2d, output_directory
    )
except Exception:
    # If running as a module or from a different CWD, add project root to sys.path
    import sys
    from pathlib import Path as _Path
    _THIS_DIR = _Path(__file__).parent
    _PROJECT_ROOT = _THIS_DIR.parent
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    from beam_propagation import (
        Coordinates, Refractive3dOptic, FixedIndexMaterial,
        from_tif, to_tif, plot_loss_history, gaussian_beam_2d, output_directory
    )

##############################################################################
## CONFIGURATION
##
## This section defines the parameters for the entire multi-stage training
## process. Instead of a long, hardcoded list, we now programmatically
## generate the stages for a more concise and flexible setup.
##############################################################################

def generate_training_schedule(
    num_stages,
    start_width_um, # Used to define the target pixel size
    start_n_xy, end_n_xy,
    iterations_per_stage,
    start_smoothing, end_smoothing
):
    """
    Programmatically generates a training schedule that maintains a constant
    pixel size while jumping grid resolution by powers of two.
    """
    # --- MODIFIED LOGIC: Constant Pixel Size ---
    # 1. Define the target pixel size from the initial stage's parameters.
    target_pixel_size = start_width_um / float(start_n_xy)
    print(f"Target pixel size locked to: {target_pixel_size:.4f} um/pixel")

    # 2. Generate grid sizes that are powers of 2.
    log_start = np.log2(start_n_xy)
    log_end = np.log2(end_n_xy)
    n_xys = np.logspace(log_start, log_end, num=num_stages, base=2).astype(int)

    # 3. Derive the physical width for each stage to maintain the target pixel size.
    widths = n_xys * target_pixel_size
    # --- END OF MODIFIED LOGIC ---

    smoothing_sigmas = np.linspace(start_smoothing, end_smoothing, num_stages)

    stages = []
    for i in range(num_stages):
        stage = {
            'name': f'Stage{i+1:02d}_Size{widths[i]:.1f}um_Grid{n_xys[i]}',
            'physical_width_um': widths[i],
            'n_xy': n_xys[i],
            'n_z': n_xys[i], # Keep n_z the same as n_xy for simplicity
            'iterations': iterations_per_stage,
            'step_size': 100, # Keep step size constant for stability
            'smoothing_sigma': smoothing_sigmas[i],
        }
        stages.append(stage)
    return stages

# --- Generate a 5-stage training schedule with constant pixel size ---
TRAINING_STAGES = generate_training_schedule(
    num_stages=5,
    start_width_um=25.4,   # Defines the pixel size for all stages
    start_n_xy=128,        # Start with baseline resolution
    end_n_xy=2048,         # End with high resolution
    iterations_per_stage=1000,
    start_smoothing=5.0,
    end_smoothing=2.0
)


# --- Define the physical properties of the optic ---
WAVELENGTH_UM = 1.0           # Wavelength of light to simulate

# --- Define the materials for the optic ---
AIR = FixedIndexMaterial(1.0)
POLYMER = FixedIndexMaterial(1.5)

# --- Define parameters for the imaging task ---
INPUT_RADIUS_FRACTION = 0.24
DIVERGENCE_ANGLE_DEGREES = 15


##############################################################################
## HELPER FUNCTIONS
##############################################################################

# Early stopping configuration: end a stage when loss stays below threshold
# for a number of consecutive iterations. The fixed iteration count remains as
# a safety cap.
CONVERGENCE_LOSS_THRESHOLD = 0.10
CONVERGENCE_PATIENCE = 10

def get_training_pair(coords, wavelength, x0, y0, divergence_angle_degrees):
    """
    Generates an energy-conserving input/output pair for the imaging task.
    """
    x, y, _ = coords.xyz
    divergence_angle_rad = np.deg2rad(divergence_angle_degrees)
    w = wavelength / (np.pi * divergence_angle_rad)
    input_field = gaussian_beam_2d(
        x=x, y=y, x0=x0, y0=y0, phi=0, theta=0,
        wavelength=wavelength, w=w)
    desired_output_field = input_field[::-1, ::-1].copy()
    return input_field.squeeze(), desired_output_field.squeeze()

def save_diagnostic_outputs(ro, output_dir, loss_history):
    """
    Saves a comprehensive set of TIF files and plots for diagnostics
    into a stage-specific subfolder.
    """
    print(f"Saving intermediate TIFs and loss plot to {output_dir}...")
    ro.update_attributes()

    # Match example_of_usage.py exact filenames and numbering inside this stage dir
    # 00-02: composition and concentration
    to_tif(output_dir / '00_composition.tif',          ro.composition)
    to_tif(output_dir / '01_concentration.tif',        ro.concentration)
    to_tif(output_dir / '02_concentration_xz.tif',
           ro.concentration[:, ro.coordinates.ny//2, :])

    # 03-05: fields at output plane (2D for 03/04; 05 is 3D magnitude of calculated_field)
    to_tif(output_dir / '03_input_field.tif',          ro.input_field)
    to_tif(output_dir / '04_desired_output_field.tif', ro.desired_output_field)
    to_tif(output_dir / '05_calculated_field.tif',     np.abs(ro.calculated_field))

    # 06-08: 3D diagnostic volumes (amplitudes for 06/07), exact names per example
    to_tif(output_dir / '06_desired_output_field_3d',  np.abs(ro.desired_output_field_3d))
    to_tif(output_dir / '07_calculated_output_field_3d', np.abs(ro.calculated_output_field_3d))
    to_tif(output_dir / '08_error_3d.tif',             ro.error_3d)

    # 09: gradient
    to_tif(output_dir / '09_gradient.tif', ro.gradient)

    # 10: loss history plot (PNG)
    plot_loss_history(loss_history, output_dir / '10_loss_history.png')

    # 11-13: additional phase maps as TIFs (quantitative, ImageJ-friendly)
    to_tif(output_dir / '11_input_phase.tif',             np.angle(ro.input_field))
    to_tif(output_dir / '12_desired_output_phase.tif',    np.angle(ro.desired_output_field))
    to_tif(output_dir / '13_calculated_output_phase.tif', np.angle(ro.calculated_field[-1]))

    # 14-16: 3D phase volumes
    # - Desired/calculated phase across propagated z-planes at the output
    # - Phase inside the optic across z (from calculated_field)
    to_tif(output_dir / '14_desired_output_field_3d_phase.tif',    np.angle(ro.desired_output_field_3d))
    to_tif(output_dir / '15_calculated_output_field_3d_phase.tif', np.angle(ro.calculated_output_field_3d))
    to_tif(output_dir / '16_calculated_field_phase_3d.tif',        np.angle(ro.calculated_field))


##############################################################################
## MULTI-STAGE TRAINING INFRASTRUCTURE
##############################################################################

def run_training_stage(stage_config, input_concentration_file=None):
    """
    Executes a single stage of the multi-scale training process.
    """
    stage_name = stage_config['name']
    # Create a dedicated subfolder for this stage's outputs under beam_propagation's output/
    base_output = output_directory()
    output_dir = base_output / stage_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}\n--- Starting Stage: {stage_name} ---\n{'='*80}")

    # 1. Setup coordinates for the current stage's physical size and resolution.
    width = stage_config['physical_width_um']
    depth = width # Maintain a 1:1 aspect ratio for simplicity
    n_xy = stage_config['n_xy']
    n_z = stage_config['n_z']
    
    coords = Coordinates(
        xyz_i=(-width/2, -width/2, 0),
        xyz_f=(+width/2, +width/2, depth),
        n_xyz=(n_xy, n_xy, n_z)
    )
    print(f"Physical Size: {width:.2f} x {width:.2f} x {depth:.2f} um")
    print(f"Grid Resolution: {coords.n_xyz}")
    print(f"Voxel Dimensions: {coords.d_xyz[0]:.3f}, {coords.d_xyz[1]:.3f}, {coords.d_xyz[2]:.3f} um")

    # 2. Initialize the Refractive3dOptic object.
    ro = Refractive3dOptic(coords)
    ro.set_materials((AIR, POLYMER))

    # 3. Initialize the optic's concentration profile.
    if input_concentration_file is None:
        # First stage: start from scratch with random noise.
        print("Initializing optic with random noise.")
        initial_concentration = 0.5 + np.random.randn(*coords.n_xyz[::-1]) * 0.01
    else:
        # Subsequent stages: load the previous result and resample (trilinear)
        # to stretch it to the full size of the new grid.
        print(f"Initializing from previous result (full-volume stretch): {input_concentration_file}")
        try:
            prev_concentration = from_tif(input_concentration_file)
            
            # --- Full-Volume Interpolation ("Stretching") ---
            # This logic remains the same, as we are always stretching from
            # one power-of-two grid to the next.
            prev_shape = prev_concentration.shape # (z, y, x)
            new_shape = coords.n_xyz[::-1]      # (z, y, x)

            # Compute zoom factors to stretch the previous volume to the new full volume
            zoom_factors = (
                new_shape[0] / float(prev_shape[0]),
                new_shape[1] / float(prev_shape[1]),
                new_shape[2] / float(prev_shape[2]),
            )
            print(f"Stretching previous result of shape {prev_shape} to new shape {new_shape}.")

            # Trilinear interpolation (order=1) to stretch the optic to the full new size
            initial_concentration = zoom(prev_concentration, zoom=zoom_factors, order=1, mode='nearest')
            initial_concentration = np.asarray(initial_concentration, dtype=np.float64)
            
            # Clip to valid range [0, 1] after interpolation
            initial_concentration = np.clip(initial_concentration, 0.0, 1.0)
            
        except FileNotFoundError:
            print(f"ERROR: Could not find file {input_concentration_file}. Aborting.")
            raise
            
    ro.set_3d_concentration(initial_concentration)

    # 4. Run the optimization loop for this stage.
    loss_history = []
    iterations = stage_config['iterations']
    max_radius = (width / 2.0) * INPUT_RADIUS_FRACTION
    print(f"Running up to {iterations} iterations (early stop if loss < {CONVERGENCE_LOSS_THRESHOLD} for {CONVERGENCE_PATIENCE} consecutive iters)...")
    consecutive_below_threshold = 0
    for i in range(iterations):
        start_time = time.perf_counter()

        r = max_radius * np.sqrt(np.random.random_sample())
        phi = 2 * np.pi * np.random.random_sample()
        x0, y0 = r * np.cos(phi), r * np.sin(phi)

        input_field, desired_output_field = get_training_pair(
            coords, WAVELENGTH_UM, x0, y0, DIVERGENCE_ANGLE_DEGREES
        )
        ro.set_2d_input_field(input_field, WAVELENGTH_UM)
        ro.set_2d_desired_output_field(desired_output_field)

        ro.gradient_update(
            step_size=stage_config['step_size'],
            smoothing_sigma=stage_config['smoothing_sigma']
        )
        loss_history.append((x0, y0, ro.loss))

        end_time = time.perf_counter()
        if np.isnan(ro.loss):
            print(f"ERROR: Loss is NaN at iteration {i+1}. Aborting.")
            return f"{stage_name}_FAILED.tif"

        print(f"Iter {i+1}/{iterations} | Loss: {ro.loss:.6f} | Time: {(end_time - start_time)*1000:.1f} ms")

        if (i + 1) % 100 == 0: # Save less frequently
            save_diagnostic_outputs(ro, output_dir, loss_history)

        # Early stopping check
        if ro.loss < CONVERGENCE_LOSS_THRESHOLD:
            consecutive_below_threshold += 1
        else:
            consecutive_below_threshold = 0
        if consecutive_below_threshold >= CONVERGENCE_PATIENCE:
            print(f"Converged: loss < {CONVERGENCE_LOSS_THRESHOLD} for {CONVERGENCE_PATIENCE} consecutive iterations at iter {i+1}.")
            break

    # 5. Save the final, converged result for this stage.
    print(f"Stage '{stage_name}' complete. Saving final concentration profile.")
    ro.update_attributes(delete_tensors=True)
    output_filename = output_dir / "final_concentration.tif"
    to_tif(output_filename, ro.concentration)

    return output_filename


def main():
    """
    Main function to orchestrate the entire multi-stage training process.
    """
    print("Starting multi-scale optic design process (Constant Pixel Size Method).")
    previous_stage_output_file = None
    for stage_config in TRAINING_STAGES:
        previous_stage_output_file = run_training_stage(
            stage_config,
            input_concentration_file=previous_stage_output_file
        )
        if "FAILED" in previous_stage_output_file.name:
            print("Stopping due to numerical instability.")
            break

    print(f"\n{'='*80}\nMulti-scale training finished!\n")
    if "FAILED" not in previous_stage_output_file.name:
        print(f"The final optic design is saved in:\n{previous_stage_output_file}\n{'='*80}")


if __name__ == '__main__':
    main()
