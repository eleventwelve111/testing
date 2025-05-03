#!/usr/bin/env python
# coding: utf-8

# In[19]:


#!/usr/bin/env python3
"""
Gamma-Ray Shielding Simulation with Air Channel

This script simulates gamma-ray transport through a concrete shield with a small
cylindrical air channel, using OpenMC. It analyzes dose dependence on detector
position and angle, and generates comprehensive visualizations.

Author: OpenMC Simulation Team
Version: 1.0.0
"""

import os
import json
import math
import time
import pickle
import logging
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import argparse
import multiprocessing as mp
import shutil
from scipy.interpolate import griddata, interp1d
from scipy.ndimage import gaussian_filter
from pathlib import Path
import traceback
import sys

# Try to import OpenMC, but don't fail if it's not available
try:
    import openmc
    HAS_OPENMC = True
    # Use a more dynamic way to find cross-section data
    try:
        import os
        if 'OPENMC_CROSS_SECTIONS' in os.environ:
            openmc.config['cross_sections'] = os.environ['OPENMC_CROSS_SECTIONS']
        else:
            # Default location
            openmc.config['cross_sections'] = '/Users/fantadiaby/Desktop/endfb-vii.1-hdf5/cross_sections.xml'
    except Exception as e:
        logger.warning(f"Could not set cross sections: {e}")
        logger.warning("Using default cross sections path")
except ImportError:
    print("Warning: OpenMC package is not installed. This script will run in demonstration mode only.")
    HAS_OPENMC = False

# Set up logging with more robust configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("simulation.log", mode='a'),  # Append mode to preserve previous logs
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=FutureWarning)

# Constants and parameters
# Dimensions (convert to cm)
CONCRETE_THICKNESS = 2 * 30.48  # 2 ft to cm
SOURCE_DISTANCE = 6 * 30.48  # 6 ft to cm
PHANTOM_DIAMETER = 30.0  # cm
WORLD_SIZE = 1000.0  # cm

# Channel diameters to study (cm)
CHANNEL_DIAMETERS = [0.05, 0.1, 0.5, 1.0]  # 0.5mm to 1cm

# Gamma-ray energies to study (MeV)
GAMMA_ENERGIES = [0.1, 0.5, 1.0, 2.0, 5.0]  # 100 keV to 5 MeV

# Detector positions (cm from back of wall)
DETECTOR_DISTANCES = [30, 40, 60, 80, 100, 150]

# Detector angles (degrees) - updated to match line 3719
DETECTOR_ANGLES = [0, 15, 30, 45, 60, 75]  # Include 0 degrees plus the angles used in the loop

# Number of particles to simulate (base value, will be scaled based on channel diameter)
BASE_PARTICLES = 5000000

# NCRP-38, ANS-6.1.1-1977 flux-to-dose conversion factors (rem/hr)/(p/cm²-s)
# Energy (MeV) -> conversion factor
FLUX_TO_DOSE = {
    0.01: 3.96e-6,
    0.03: 5.82e-7,
    0.05: 2.90e-7,
    0.07: 2.58e-7,
    0.1: 2.83e-7,
    0.15: 3.79e-7,
    0.2: 5.01e-7,
    0.25: 6.31e-7,
    0.3: 7.59e-7,
    0.35: 8.78e-7,
    0.4: 9.85e-7,
    0.45: 1.08e-6,
    0.5: 1.17e-6,
    0.55: 1.27e-6,
    0.6: 1.36e-6,
    0.65: 1.44e-6,
    0.7: 1.52e-6,
    0.8: 1.68e-6,
    1.0: 1.98e-6,
    1.4: 2.51e-6,
    1.8: 2.99e-6,
    2.2: 3.42e-6,
    2.6: 3.82e-6,
    2.8: 4.01e-6,
    3.25: 4.41e-6,
    3.75: 4.83e-6,
    4.25: 5.23e-6,
    4.75: 5.60e-6,
    5.0: 5.80e-6,
    5.25: 6.01e-6,
    5.75: 6.37e-6,
    6.25: 6.74e-6,
    6.75: 7.11e-6,
    7.5: 7.66e-6,
    9.0: 8.77e-6,
    11.0: 1.03e-5,
    13.0: 1.18e-5,
    15.0: 1.33e-5
}

# Create directories with proper error handling
RESULTS_DIR = Path('results')
CHECKPOINT_DIR = RESULTS_DIR / 'checkpoints'
PLOTS_DIR = RESULTS_DIR / 'plots'
DATA_DIR = RESULTS_DIR / 'data'


# Create directories with proper error handling
# Safely create directories
for directory in [RESULTS_DIR, CHECKPOINT_DIR, PLOTS_DIR, DATA_DIR]:
    try:
        directory.mkdir(exist_ok=True, parents=True)
    except Exception as e:
        logger.error(f"Failed to create directory {directory}: {e}")


def interpolate_flux_to_dose(energy):
    """
    Interpolate flux-to-dose conversion factor for a given energy.
    Handles both scalar and array inputs.

    Args:
        energy (float or np.ndarray): Gamma-ray energy in MeV

    Returns:
        float or np.ndarray: Interpolated flux-to-dose conversion factor
    """
    energies = np.array(list(FLUX_TO_DOSE.keys()))
    factors = np.array(list(FLUX_TO_DOSE.values()))

    # Handle array input
    if isinstance(energy, np.ndarray):
        # Create an output array of the same shape as energy
        result = np.zeros_like(energy, dtype=float)
        
        # Process each element in the array
        for i in range(energy.size):
            e = energy.flat[i]
            if e <= np.min(energies):
                result.flat[i] = FLUX_TO_DOSE[np.min(energies)]
            elif e >= np.max(energies):
                result.flat[i] = FLUX_TO_DOSE[np.max(energies)]
            else:
                # Use log-log interpolation for better accuracy
                log_energies = np.log(energies)
                log_factors = np.log(factors)
                
                # Create interpolation function
                interp_func = interp1d(log_energies, log_factors, kind='linear')
                
                # Interpolate and convert back from log
                log_result = interp_func(np.log(e))
                result.flat[i] = np.exp(log_result)
        
        return result
    
    # Handle scalar input (original behavior)
    else:
        if energy <= min(energies):
            return FLUX_TO_DOSE[min(energies)]
        elif energy >= max(energies):
            return FLUX_TO_DOSE[max(energies)]

    # Use log-log interpolation for better accuracy
    log_energies = np.log(energies)
    log_factors = np.log(factors)

    # Create interpolation function
    interp_func = interp1d(log_energies, log_factors, kind='linear')

    # Interpolate and convert back from log
    log_result = interp_func(np.log(energy))
    return np.exp(log_result)

def create_materials():
    """
    Create materials for the simulation with dynamic IDs to avoid conflicts.

    Returns:
        tuple: (materials collection, air, concrete, tissue)
    """
    logger.info("Creating materials...")

    # Generate random base for material IDs to avoid conflicts in parallel runs
    import random
    base_id = random.randint(10000, 99000)
    
    # Create materials with dynamic IDs to avoid conflicts
    air = openmc.Material(material_id=base_id + 1, name='Air')
    air.set_density('g/cm3', 0.001205)
    air.add_element('N', 0.784)
    air.add_element('O', 0.216)

    concrete = openmc.Material(material_id=base_id + 2, name='Concrete')
    concrete.set_density('g/cm3', 2.3)
    concrete.add_element('H', 0.01)
    concrete.add_element('O', 0.53)
    concrete.add_element('Si', 0.33)
    concrete.add_element('Ca', 0.12)
    concrete.add_element('Fe', 0.01)

    # Create tissue for detectors
    tissue = openmc.Material(material_id=base_id + 3, name='Tissue')
    tissue.set_density('g/cm3', 1.0)
    tissue.add_element('H', 0.101)
    tissue.add_element('C', 0.111)
    tissue.add_element('N', 0.026)
    tissue.add_element('O', 0.762)

    # Create materials collection
    materials = openmc.Materials([air, concrete, tissue])
    
    # Return materials
    return materials, air, concrete, tissue
def create_geometry(channel_diameter, detector_cells=None):
    """
    Create geometry for the simulation, with vacuum environment outside the shield
    to prevent ground and skyshine effects, ensuring symmetric radiation patterns.

    Args:
        channel_diameter (float): Diameter of the air channel in cm
        detector_cells (list): Optional list of detector cells to include in the geometry

    Returns:
        tuple: (geometry, surfaces dict)
    """
    logger.info(f"Creating geometry with channel diameter {channel_diameter} cm...")

    # Create materials
    materials, air, concrete, tissue = create_materials()

    # Create surfaces
    # World boundary with vacuum boundary condition
    world_min = -WORLD_SIZE/2
    world_max = WORLD_SIZE/2
    xmin = openmc.XPlane(world_min, boundary_type='vacuum')
    xmax = openmc.XPlane(world_max, boundary_type='vacuum')
    ymin = openmc.YPlane(world_min, boundary_type='vacuum')
    ymax = openmc.YPlane(world_max, boundary_type='vacuum')
    zmin = openmc.ZPlane(world_min, boundary_type='vacuum')
    zmax = openmc.ZPlane(world_max, boundary_type='vacuum')

    # Concrete shield
    shield_front = openmc.XPlane(0.0)
    shield_back = openmc.XPlane(CONCRETE_THICKNESS)

    # Air channel (cylindrical)
    channel_radius = channel_diameter / 2.0
    channel = openmc.ZCylinder(x0=0, y0=0, r=channel_radius)

    # Create regions
    # World region (all space within boundaries)
    world_region = +xmin & -xmax & +ymin & -ymax & +zmin & -zmax

    # Concrete shield region (with hole for channel)
    shield_region = +shield_front & -shield_back & +ymin & -ymax & +zmin & -zmax & ~(-channel)

    # Air channel region
    channel_region = +shield_front & -shield_back & -channel

    # Create cells with dynamic IDs to avoid conflicts in parallel runs
    import random
    base_cell_id = random.randint(1000, 9000)
    
    # World cell (filled with air instead of void)
    world_cell = openmc.Cell(cell_id=base_cell_id, name='World')
    world_cell.region = world_region
    world_cell.fill = air  # Using air outside the shield instead of void

    # Concrete shield cell
    shield_cell = openmc.Cell(cell_id=base_cell_id + 1, name='Concrete Shield')
    shield_cell.region = shield_region
    shield_cell.fill = concrete

    # Air channel cell
    channel_cell = openmc.Cell(cell_id=base_cell_id + 2, name='Air Channel')
    channel_cell.region = channel_region
    channel_cell.fill = air

    # Create cells list with the main geometry cells
    cells = [world_cell, shield_cell, channel_cell]
    
    # Add detector cells if provided
    if detector_cells:
        # We'll create a region outside the shield for the detectors
        outside_shield_region = -shield_front | +shield_back
        
        # Create a universe for the detectors
        for detector_cell in detector_cells:
            # Update the region to be contained within the world
            detector_cell.region = detector_cell.region & world_region & outside_shield_region
            cells.append(detector_cell)
            logger.info(f"Added detector cell ID {detector_cell.id} to geometry")

    # Create universe
    universe = openmc.Universe(cells=cells)

    # Create geometry
    geometry = openmc.Geometry(universe)

    # Store important surfaces for later use
    surfaces = {
        'shield_front': shield_front,
        'shield_back': shield_back,
        'channel': channel
    }

    # Export materials and geometry to XML files
    materials.export_to_xml()
    geometry.export_to_xml()

    return geometry, surfaces

def create_source(energy, channel_diameter):
    """
    Create a point source with optimized biased angular distribution to focus particles
    toward the channel, ensuring 100% of particles pass through without touching the concrete wall.

    Args:
        energy (float): Gamma-ray energy in MeV
        channel_diameter (float): Diameter of the air channel in cm

    Returns:
        openmc.Source: Source definition
    """
    logger.info(f"Creating optimized source with energy {energy} MeV for channel diameter {channel_diameter} cm...")

    # Source position (in front of shield)
    x = -SOURCE_DISTANCE
    y = 0.0
    z = 0.0

    # Calculate the maximum angle that ensures particles go through the channel
    # without touching the concrete walls using the isosceles triangle formed by 
    # the source and channel endpoints
    channel_radius = channel_diameter / 2.0
    
    # Calculate the angle subtended by the channel from the source point
    # Use a smaller angle than geometrically possible to account for floating point precision
    # This ensures no particles touch the concrete walls
    theta_max = math.atan(channel_radius / SOURCE_DISTANCE)
    
    # Apply a small safety margin (99% of max angle) to ensure particles stay within channel
    theta_max_safe = theta_max * 0.99
    
    logger.info(f"Maximum channel angle: {math.degrees(theta_max):.6f} degrees, Safe angle: {math.degrees(theta_max_safe):.6f} degrees")
    
    # Lower limit of cosine of polar angle
    mu_min = math.cos(theta_max_safe)
    
    # Calculate solid angle fraction for correct weight adjustment
    solid_angle_fraction = (1 - mu_min) / 2
    logger.info(f"Solid angle fraction: {solid_angle_fraction:.8f}")

    # Create spatial distribution (point source)
    space = openmc.stats.Point((x, y, z))

    # Create energy distribution (monoenergetic)
    energy_dist = openmc.stats.Discrete([energy], [1.0])
    
    # Create a precise angular distribution to focus particles within the channel
    angle_dist = openmc.stats.PolarAzimuthal(
        mu=openmc.stats.Uniform(mu_min, 1.0),  # Cosine of polar angle from mu_min to 1.0
        phi=openmc.stats.Uniform(0.0, 2*math.pi)  # Full azimuthal angle range
    )
    
    # Create source with biased angular distribution
    source = openmc.Source(space=space, energy=energy_dist, angle=angle_dist)
    source.particle = 'photon'
    
    # Set particle weight to account for biased sampling
    # This ensures the physics is not biased while improving efficiency
    source.strength = 1.0 / solid_angle_fraction

    return source


def create_tallies(channel_diameter):
    """
    Create optimized tallies for the simulation.

    Args:
        channel_diameter (float): Diameter of the air channel in cm

    Returns:
        tuple: (tallies, mesh dimensions and bounds)
    """
    logger.info(f"Creating optimized tallies for channel diameter {channel_diameter} cm...")

    # First, create the materials needed for detectors
    # This tissue material must be added to the materials collection to be properly referenced
    tissue = openmc.Material(material_id=9999, name='Tissue')  # Using ID 9999 to avoid conflicts
    tissue.set_density('g/cm3', 1.0)
    tissue.add_element('H', 0.101)
    tissue.add_element('C', 0.111)
    tissue.add_element('N', 0.026)
    tissue.add_element('O', 0.762)
    
    # Add tissue to the materials file if not already there
    try:
        # Try to load existing materials
        materials = openmc.Materials.from_xml()
        # Check if tissue is already in the materials
        if not any(m.id == 9999 for m in materials):
            materials.append(tissue)
    except:
        # If materials.xml doesn't exist, create a new Materials collection with tissue
        materials = openmc.Materials([tissue])
    
    # Export the materials to XML to ensure they're available
    materials.export_to_xml()

    tallies = openmc.Tallies()

    # Create mesh for 2D cartesian visualization with adaptive resolution
    # Mesh covering from source to beyond detector
    mesh_min_x = -SOURCE_DISTANCE - 10
    mesh_max_x = CONCRETE_THICKNESS + DETECTOR_DISTANCES[-1] + 20
    mesh_min_yz = -60  # Increased for better coverage
    mesh_max_yz = 60   # Increased for better coverage

    # Adaptive mesh resolution based on channel diameter
    # Smaller channels need finer mesh for accurate visualization
    mesh_resolution_factor = max(1.0, min(4.0, 0.5 / max(channel_diameter, 0.05)))
    
    # Base resolution
    n_x_base = 200
    n_yz_base = 100
    
    # Apply resolution factor with reasonable limits
    n_x = min(600, int(n_x_base * mesh_resolution_factor))
    n_yz = min(300, int(n_yz_base * mesh_resolution_factor))
    
    logger.info(f"Using mesh resolution: {n_x}x{n_yz}x{n_yz} (factor: {mesh_resolution_factor:.2f})")

    mesh = openmc.RegularMesh(name='FullMesh')
    mesh.dimension = [n_x, n_yz, n_yz]
    mesh.lower_left = [mesh_min_x, mesh_min_yz, mesh_min_yz]
    mesh.upper_right = [mesh_max_x, mesh_max_yz, mesh_max_yz]

    # Mesh tally for flux with both flux and dose scores
    # Use modern Tally API with mesh filter instead of MeshTally
    mesh_filter = openmc.MeshFilter(mesh)
    mesh_tally = openmc.Tally(name='FullMeshTally')
    mesh_tally.filters = [mesh_filter]
    mesh_tally.scores = ['flux']
    tallies.append(mesh_tally)

    # Fine mesh for detailed visualization near the channel exit
    # Adjust mesh resolution based on channel diameter
    fine_mesh_min_x = CONCRETE_THICKNESS - 5
    fine_mesh_max_x = CONCRETE_THICKNESS + 60

    # Smaller channels need finer mesh
    fine_mesh_size = max(0.05, channel_diameter / 10)  # Much finer mesh
    fine_mesh_min_yz = -25
    fine_mesh_max_yz = 25

    fine_n_x = int((fine_mesh_max_x - fine_mesh_min_x) / fine_mesh_size)
    fine_n_yz = int((fine_mesh_max_yz - fine_mesh_min_yz) / fine_mesh_size)

    # Ensure reasonable mesh size
    fine_n_x = min(fine_n_x, 200)
    fine_n_yz = min(fine_n_yz, 160)

    fine_mesh = openmc.RegularMesh(name='FineMesh')
    fine_mesh.dimension = [fine_n_x, fine_n_yz, fine_n_yz]
    fine_mesh.lower_left = [fine_mesh_min_x, fine_mesh_min_yz, fine_mesh_min_yz]
    fine_mesh.upper_right = [fine_mesh_max_x, fine_mesh_max_yz, fine_mesh_max_yz]

    # Fine mesh tally for flux
    fine_mesh_filter = openmc.MeshFilter(fine_mesh)
    fine_mesh_tally = openmc.Tally(name='FineMeshTally')
    fine_mesh_tally.filters = [fine_mesh_filter]
    fine_mesh_tally.scores = ['flux']
    tallies.append(fine_mesh_tally)

    # Create energy bins for spectrum analysis
    energy_bins = np.logspace(-2, 1, 100)  # 10 keV to 10 MeV
    energy_filter = openmc.EnergyFilter(energy_bins)

    # In OpenMC 0.14.0, we need to work with the model universe directly
    # Create a list of detector cells to add to our universe at the end
    detector_cells = []

    # Generate a high base ID for detector cells to avoid conflicts
    import random
    cell_id = random.randint(10000, 90000)  # Much higher range to avoid conflicts

    for distance in DETECTOR_DISTANCES:
        for angle in DETECTOR_ANGLES:
            # Calculate detector position
            angle_rad = math.radians(angle)
            detector_x = CONCRETE_THICKNESS + distance * math.cos(angle_rad)
            detector_y = distance * math.sin(angle_rad)
            detector_z = 0.0

            # Create spherical detector
            detector_sphere = openmc.Sphere(
                x0=detector_x, y0=detector_y, z0=detector_z, r=PHANTOM_DIAMETER/2
            )

            # Create detector cell with explicit ID and the tissue material
            detector_cell = openmc.Cell(cell_id=cell_id, name=f'Detector_D{distance}_A{angle}')
            detector_cell.region = -detector_sphere
            detector_cell.fill = tissue  # Use the tissue material created above
            
            # Increment cell ID for next detector
            cell_id += 1
            
            # Add to list of detector cells
            detector_cells.append(detector_cell)

            # Create cell tally for detector - use the specific cell object
            cell_filter = openmc.CellFilter([detector_cell])  # Pass cell object directly, not ID
            detector_tally = openmc.Tally(name=f'DetectorTally_D{distance}_A{angle}')
            detector_tally.filters = [cell_filter, energy_filter]
            detector_tally.scores = ['flux']
            tallies.append(detector_tally)

    # Export tallies to XML
    tallies.export_to_xml()

    # Return mesh dimensions and bounds for later use
    mesh_info = {
        'full_mesh': {
            'dimensions': [n_x, n_yz, n_yz],
            'bounds': [mesh_min_x, mesh_max_x, mesh_min_yz, mesh_max_yz]
        },
        'fine_mesh': {
            'dimensions': [fine_n_x, fine_n_yz, fine_n_yz],
            'bounds': [fine_mesh_min_x, fine_mesh_max_x, fine_mesh_min_yz, fine_mesh_max_yz]
        }
    }

    return tallies, mesh_info, detector_cells

def create_settings(energy, channel_diameter, particles=None):
    """
    Create simulation settings.

    Args:
        energy (float): Gamma-ray energy in MeV
        channel_diameter (float): Diameter of the air channel in cm
        particles (int, optional): User-specified number of particles, overrides default calculation

    Returns:
        openmc.Settings: Simulation settings
    """
    logger.info(f"Creating settings for energy {energy} MeV and channel diameter {channel_diameter} cm...")
    # Calculate number of particles based on channel diameter
    # Smaller channels need more particles for good statistics
    if particles is not None:
        NUM_PARTICLES = particles
        logger.info(f"Using user-specified particle count: {NUM_PARTICLES}")
    else:
        NUM_PARTICLES = int(BASE_PARTICLES * (0.5 / max(channel_diameter, 0.05)))
        logger.info(f"Using calculated particle count: {NUM_PARTICLES}")

    # Create settings object
    settings = openmc.Settings()

    # Set number of particles
    settings.particles = NUM_PARTICLES

    # Set number of batches
    settings.batches = 10
    settings.inactive = 0

    # Set run mode
    settings.run_mode = 'fixed source'

    # Set seed for reproducibility
    settings.seed = 42

    # Set cutoffs
    settings.cutoff = {
        'energy_photon': 0.001  # 1 keV cutoff for photons
    }

    # Set physics parameters
    settings.photon_transport = True

    # Set custom output filename based on energy and diameter
    # In some OpenMC versions, need to use path='filename.h5' format
    settings.statepoint = {'batches': [settings.batches]}
    
    # Use unique filenames for summary and other output files to avoid conflicts in parallel runs
    timestamp = int(time.time() * 1000) % 100000  # Use milliseconds for uniqueness
    unique_suffix = f"_E{energy}_D{channel_diameter}_{timestamp}"
    
    # Set the output directory for summary and statepoint files
    openmc.OPENMC_ARGS = ['--output-summary', f"summary{unique_suffix}.h5"]

    return settings

def save_checkpoint(results, energy, channel_diameter):
    """
    Save checkpoint of simulation results with robust error handling and multiple
    backup formats to ensure data persistence.

    Args:
        results (dict): Results dictionary
        energy (float): Gamma-ray energy in MeV
        channel_diameter (float): Diameter of the air channel in cm
    """
    # Create checkpoint directory if it doesn't exist
    if not CHECKPOINT_DIR.exists():
        try:
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create checkpoint directory: {e}")
            return
    
    # Create temporary files first, then rename them to avoid corruption
    checkpoint_file = CHECKPOINT_DIR / f'checkpoint_E{energy}_D{channel_diameter}.pkl'
    temp_file = CHECKPOINT_DIR / f'temp_checkpoint_E{energy}_D{channel_diameter}.pkl'
    json_file = CHECKPOINT_DIR / f'checkpoint_E{energy}_D{channel_diameter}.json'
    temp_json_file = CHECKPOINT_DIR / f'temp_checkpoint_E{energy}_D{channel_diameter}.json'
    
    # Create backup files with timestamps for additional safety
    backup_dir = CHECKPOINT_DIR / 'backups'
    try:
        backup_dir.mkdir(exist_ok=True)
    except Exception as e:
        logger.warning(f"Failed to create backup directory: {e}")
    
    backup_file = backup_dir / f'checkpoint_E{energy}_D{channel_diameter}_{int(time.time())}.pkl'
    
    try:
        # Save pickle file
        with open(temp_file, 'wb') as f:
            pickle.dump(results, f)
    except Exception as e:
        logger.error(f"Error saving pickle file: {e}")

    try:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                json_results[key] = {}
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, dict):
                        json_results[key][subkey] = {}
                        for subsubkey, subsubvalue in subvalue.items():
                            if isinstance(subsubvalue, dict):
                                json_results[key][subkey][subsubkey] = {}
                                for k, v in subsubvalue.items():
                                    if isinstance(v, np.ndarray):
                                        json_results[key][subkey][subsubkey][k] = v.tolist()
                                    else:
                                        json_results[key][subkey][subsubkey][k] = v
                            elif isinstance(subsubvalue, np.ndarray):
                                json_results[key][subkey][subsubkey] = subsubvalue.tolist()
                            else:
                                json_results[key][subkey][subsubkey] = subsubvalue
                    elif isinstance(subvalue, np.ndarray):
                        json_results[key][subkey] = subvalue.tolist()
                    else:
                        json_results[key][subkey] = subvalue

        # Save JSON file with better formatting for readability
        with open(temp_json_file, 'w') as f:
            json.dump(json_results, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving JSON file: {e}")

    try:
        # After successful writes, rename the temp files to the final filenames
        if temp_file.exists():
            try:
                if checkpoint_file.exists():
                    checkpoint_file.unlink()
                temp_file.rename(checkpoint_file)
            except OSError as e:
                logger.error(f"Error renaming pickle checkpoint: {e}")
                
        if temp_json_file.exists():
            try:
                if json_file.exists():
                    json_file.unlink()
                temp_json_file.rename(json_file)
            except OSError as e:
                logger.error(f"Error renaming JSON checkpoint: {e}")
        
        # Save backup file if backup directory exists
        if backup_dir.exists():
            try:
                with open(backup_file, 'wb') as f:
                    pickle.dump(results, f)
                logger.info(f"Backup checkpoint saved to {backup_file}")
            except Exception as e:
                logger.warning(f"Failed to save backup checkpoint: {e}")
    except Exception as e:
        logger.error(f"Error finalizing checkpoint save: {e}")

    logger.info(f"Checkpoint saved for energy {energy} MeV and channel diameter {channel_diameter} cm")

def load_checkpoint(energy, channel_diameter):
    """
    Load checkpoint of simulation results with robust error handling and fallback
    mechanisms for improved reliability.

    Args:
        energy (float): Gamma-ray energy in MeV
        channel_diameter (float): Diameter of the air channel in cm

    Returns:
        dict or None: Results dictionary if checkpoint exists, None otherwise
    """
    checkpoint_file = CHECKPOINT_DIR / f'checkpoint_E{energy}_D{channel_diameter}.pkl'
    json_file = CHECKPOINT_DIR / f'checkpoint_E{energy}_D{channel_diameter}.json'
    temp_file = CHECKPOINT_DIR / f'temp_checkpoint_E{energy}_D{channel_diameter}.pkl'
    
    # List of files to try in order of preference
    files_to_try = [
        (checkpoint_file, 'pickle'),
        (temp_file, 'pickle'),
        (json_file, 'json')
    ]
    
    # Also look for backup files
    backup_dir = CHECKPOINT_DIR / 'backups'
    if backup_dir.exists():
        backup_files = list(backup_dir.glob(f'checkpoint_E{energy}_D{channel_diameter}_*.pkl'))
        if backup_files:
            # Sort by modification time, most recent first
            backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            # Add the most recent backup file to the list
            files_to_try.append((backup_files[0], 'pickle'))
    
    # Try loading from each file until one succeeds
    for file_path, file_type in files_to_try:
        if file_path.exists():
            try:
                if file_type == 'pickle':
                    with open(file_path, 'rb') as f:
                        results = pickle.load(f)
                    logger.info(f"Loaded checkpoint from {file_path}")
                else:  # json
                    with open(file_path, 'r') as f:
                        json_results = json.load(f)
                    
                    # Convert JSON data to proper Python objects
                    results = {}
                    for key, value in json_results.items():
                        if isinstance(value, dict):
                            results[key] = {}
                            for subkey, subvalue in value.items():
                                if isinstance(subvalue, dict):
                                    results[key][subkey] = {}
                                    for subsubkey, subsubvalue in subvalue.items():
                                        if isinstance(subsubvalue, dict):
                                            results[key][subkey][subsubkey] = {}
                                            for k, v in subsubvalue.items():
                                                if isinstance(v, list):
                                                    results[key][subkey][subsubkey][k] = np.array(v)
                                                else:
                                                    results[key][subkey][subsubkey][k] = v
                                        elif isinstance(subsubvalue, list):
                                            results[key][subkey][subsubkey] = np.array(subsubvalue)
                                        else:
                                            results[key][subkey][subsubkey] = subsubvalue
                                elif isinstance(subvalue, list):
                                    results[key][subkey] = np.array(subvalue)
                                else:
                                    results[key][subkey] = subvalue
                        elif isinstance(value, list):
                            results[key] = np.array(value)
                        else:
                            results[key] = value
                       
                    logger.info(f"Loaded checkpoint from JSON file {file_path}")
                
                # Verify checkpoint integrity
                if (str(energy) in results and 
                    str(channel_diameter) in results[str(energy)] and
                    'detector_results' in results[str(energy)][str(channel_diameter)]):
                    
                    # Save to main checkpoint file if loaded from a backup or temp file
                    if file_path != checkpoint_file and file_type == 'pickle':
                        try:
                            with open(checkpoint_file, 'wb') as f:
                                pickle.dump(results, f)
                            logger.info(f"Restored primary checkpoint from {file_path}")
                        except Exception as e:
                            logger.warning(f"Failed to restore primary checkpoint: {e}")
                    
                    return results
                else:
                    logger.warning(f"Checkpoint structure invalid for E={energy}, D={channel_diameter} in {file_path}")
            except Exception as e:
                logger.warning(f"Error loading checkpoint from {file_path}: {e}")
    
    logger.info(f"No valid checkpoint found for energy {energy} MeV and channel diameter {channel_diameter} cm")
    return None

def visualize_radiation_map(mesh_data, mesh_dimensions, mesh_bounds, energy, channel_diameter, filename):
    """
    Visualize 2D radiation map showing the full 360-degree radiation pattern.

    Args:
        mesh_data (numpy.ndarray): Mesh tally data
        mesh_dimensions (list): Mesh dimensions [nx, ny, nz]
        mesh_bounds (list): Mesh bounds [xmin, xmax, ymin, ymax]
        energy (float): Gamma-ray energy in MeV
        channel_diameter (float): Diameter of the air channel in cm
        filename (str): Output filename
    """
    logger.info(f"Creating radiation map visualization for energy {energy} MeV and channel diameter {channel_diameter} cm...")

    # Reshape mesh data
    nx, ny, nz = mesh_dimensions
    xmin, xmax, ymin, ymax = mesh_bounds

    try:
        # Extract central slice (z=0)
        central_slice = mesh_data.reshape(nx, ny, nz)[:, :, nz//2]

        # Create coordinate meshes
        x = np.linspace(xmin, xmax, nx)
        y = np.linspace(ymin, ymax, ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Mirror the data for the bottom half
        X_full = X.copy()
        Y_top = Y.copy()
        Y_bottom = -Y.copy()  # Mirror across x-axis
        data_top = central_slice.copy()
        data_bottom = central_slice.copy()  # Same data, mirrored location

        # Apply log scale for better visualization with error handling
        with np.errstate(divide='ignore', invalid='ignore'):
            log_data_top = np.log10(data_top)
            log_data_bottom = np.log10(data_bottom)

        # Replace -inf with minimum finite value
        # Handle case when all values are invalid
        finite_mask_top = np.isfinite(log_data_top)
        finite_mask_bottom = np.isfinite(log_data_bottom)
        
        if np.any(finite_mask_top):
            min_val_top = np.min(log_data_top[finite_mask_top])
            log_data_top[~finite_mask_top] = min_val_top - 1
        else:
            # If no finite values, set a default minimum
            min_val_top = -10
            log_data_top.fill(min_val_top)
            
        if np.any(finite_mask_bottom):
            min_val_bottom = np.min(log_data_bottom[finite_mask_bottom])
            log_data_bottom[~finite_mask_bottom] = min_val_bottom - 1
        else:
            # If no finite values, set a default minimum
            min_val_bottom = -10
            log_data_bottom.fill(min_val_bottom)
    
        min_val = min(min_val_top, min_val_bottom)

        # Create figure with improved size
        plt.figure(figsize=(14, 16))  # Taller figure for full view

        # Plot radiation map with improved colormap
        cmap = plt.cm.viridis
        norm = colors.Normalize(vmin=min_val, vmax=max(np.max(log_data_top), np.max(log_data_bottom)))
        plt.pcolormesh(X_full, Y_top, log_data_top, cmap=cmap, norm=norm, shading='auto')
        plt.pcolormesh(X_full, Y_bottom, log_data_bottom, cmap=cmap, norm=norm, shading='auto')

        # Add colorbar with better formatting
        cbar = plt.colorbar()
        cbar.set_label('Log10(Flux) [particles/cm²]', fontsize=12)

        # Add shield outline with improved styling
        plt.axvline(x=0, color='white', linestyle='-', linewidth=2, label='Entrance')
        plt.axvline(x=CONCRETE_THICKNESS, color='white', linestyle='-', linewidth=2, label='Exit')

        # Add source point with improved styling
        plt.plot(-SOURCE_DISTANCE, 0, 'yo', markersize=10, markeredgecolor='black', label='Source')

        # Add channel with improved styling
        channel_radius = channel_diameter / 2.0
        plt.plot([0, CONCRETE_THICKNESS], [0, 0], 'w--', linewidth=1.5)
        
        # Add channel position with more prominent marking
        circle = plt.Circle((CONCRETE_THICKNESS, 0), channel_radius, fill=False, 
                          edgecolor='yellow', linewidth=2, label='Channel')
        plt.gca().add_patch(circle)

        # Add detector positions with improved styling
        for distance in [30, 50, 100, 150]:
            circle = plt.Circle((CONCRETE_THICKNESS, 0), distance, fill=False, 
                              edgecolor='gray', linestyle='--', linewidth=1)
            plt.gca().add_patch(circle)
            plt.text(CONCRETE_THICKNESS + distance, 0, f'{distance} cm', color='white', 
                   ha='left', va='center', fontsize=10, 
                   bbox=dict(facecolor='black', alpha=0.5, boxstyle='round'))

        # Add angle indicators for both top and bottom halves
        angles = [-45, -30, -15, 0, 15, 30, 45]
        for angle in angles:
            angle_rad = math.radians(angle)
            dx = 100 * math.cos(angle_rad)
            dy = 100 * math.sin(angle_rad)
            plt.plot([CONCRETE_THICKNESS, CONCRETE_THICKNESS + dx], [0, dy], 
                    'w--', linewidth=1, alpha=0.7)
            plt.text(CONCRETE_THICKNESS + dx + 5, dy, f'{angle} degrees', color='white', 
                   ha='left', va='center', fontsize=10,
                   bbox=dict(facecolor='black', alpha=0.5, boxstyle='round'))

        # Add labels and title with improved styling
        plt.xlabel('X [cm]', fontsize=12)
        plt.ylabel('Y [cm]', fontsize=12)
        plt.title(f'Full 360 degrees Radiation Map (E={energy} MeV, Channel Ø={channel_diameter} cm)',
                 fontsize=14, fontweight='bold')

        # Set aspect ratio to equal
        plt.axis('equal')
        
        # Set limits to focus on area of interest with wider view
        plt.xlim(-SOURCE_DISTANCE - 10, CONCRETE_THICKNESS + 120)
        plt.ylim(-80, 80)  # Extended to show bottom half

        # Add grid for better reference
        plt.grid(True, linestyle='--', alpha=0.3, color='gray')

        # Add legend with improved styling
        plt.legend(loc='upper right', framealpha=0.7, fontsize=10)

        # Save figure with high resolution
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also create a polar version of the radiation map
        create_polar_map(mesh_data, mesh_dimensions, mesh_bounds, channel_diameter, energy, 
                       str(filename).replace('.png', '_polar.png'))
    except Exception as e:
        logger.error(f"Error creating radiation map visualization: {e}")
        logger.error(traceback.format_exc())
        
        # Create a simple placeholder visualization with just the geometry
        plt.figure(figsize=(12, 10))
        plt.plot(-SOURCE_DISTANCE, 0, 'ro', markersize=10, label='Source')
        plt.axvline(x=0, color='gray', linestyle='-', linewidth=4, label='Shield Front')
        plt.axvline(x=CONCRETE_THICKNESS, color='gray', linestyle='-', linewidth=4, label='Shield Back')
        
        # Draw channel
        channel_radius = channel_diameter / 2.0
        plt.plot([0, CONCRETE_THICKNESS], [0, 0], 'b-', linewidth=2, label='Channel')
        
        # Add message about insufficient data
        plt.text(0, -50, f"Insufficient tally data for visualization\nTry running with more particles", 
                ha='center', va='center', fontsize=14, 
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
        
        plt.xlabel('X [cm]', fontsize=12)
        plt.ylabel('Y [cm]', fontsize=12)
        plt.title(f'Radiation Map (E={energy} MeV, Channel Ø={channel_diameter} cm) - No Data',
                 fontsize=14, fontweight='bold')
        plt.axis('equal')
        plt.legend()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

def create_polar_map(mesh_data, mesh_dimensions, mesh_bounds, channel_diameter, energy, filename):
    """
    Create a polar visualization of the standard radiation map.
    
    Args:
        mesh_data (numpy.ndarray): Mesh tally data
        mesh_dimensions (list): Mesh dimensions [nx, ny, nz]
        mesh_bounds (list): Mesh bounds [xmin, xmax, ymin, ymax]
        channel_diameter (float): Diameter of the air channel in cm
        energy (float): Gamma-ray energy in MeV
        filename (str): Output filename
    """
    logger.info(f"Creating polar radiation map for energy {energy} MeV and channel diameter {channel_diameter} cm...")
    
    # Reshape mesh data
    nx, ny, nz = mesh_dimensions
    xmin, xmax, ymin, ymax = mesh_bounds

    # Extract central slice (z=0)
    central_slice = mesh_data.reshape(nx, ny, nz)[:, :, nz//2]
    
    # Create coordinate meshes
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Calculate r and theta for each point (centered at exit face of shield)
    r = np.sqrt((X - CONCRETE_THICKNESS)**2 + Y**2)
    theta = np.arctan2(Y, X - CONCRETE_THICKNESS)
    
    # Prepare data for interpolation
    points = np.column_stack((r.flatten(), theta.flatten()))
    values = central_slice.flatten()
    
    # Create a regular grid in polar coordinates
    r_grid = np.linspace(0, 200, 200)  # Distances from 0 to 200 cm
    theta_grid = np.linspace(-np.pi, np.pi, 360)  # Full 360 degrees
    R, THETA = np.meshgrid(r_grid, theta_grid)
    
    # Interpolate the data to the regular polar grid
    from scipy.interpolate import griddata
    polar_data = griddata(points, values, (R.flatten(), THETA.flatten()), method='cubic', fill_value=1e-10)
    polar_data = polar_data.reshape(R.shape)
    
    # Apply log scale
    with np.errstate(divide='ignore', invalid='ignore'):
        log_polar = np.log10(polar_data)
    
    # Replace -inf with minimum finite value
    min_val = np.min(log_polar[np.isfinite(log_polar)])
    log_polar[~np.isfinite(log_polar)] = min_val
    
    # Create the polar plot
    plt.figure(figsize=(12, 12))
    ax = plt.subplot(111, polar=True)
    
    # Plot the heatmap
    cax = ax.pcolormesh(THETA, R, log_polar, cmap='viridis', shading='auto')
    
    # Add colorbar
    cbar = plt.colorbar(cax, ax=ax, pad=0.1)
    cbar.set_label('Log10(Flux) [particles/cm²]', fontsize=12)
    
    # Add radial grid lines
    distances = [10, 30, 50, 100, 150, 200]
    for d in distances:
        if d <= max(r_grid):
            ax.plot(theta_grid, [d] * len(theta_grid), 'k--', linewidth=0.5, alpha=0.3)
            ax.text(0, d, f'{d} cm', ha='center', va='bottom', fontsize=8, 
                  bbox=dict(facecolor='white', alpha=0.6))
    
    # Add angle lines
    angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    for angle in angles:
        angle_rad = math.radians(angle)
        ax.plot([angle_rad, angle_rad], [0, max(r_grid)], 'k--', linewidth=0.5, alpha=0.3)
        if angle <= 180:
            ax.text(angle_rad, max(r_grid) * 1.05, f'{angle} degrees', 
                  ha='center', va='center', fontsize=8,
                  bbox=dict(facecolor='white', alpha=0.6))
    
    # Set title
    plt.title(f'Polar Radiation Map (E={energy} MeV, Channel Ø={channel_diameter} cm)',
             fontsize=14, fontweight='bold')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_enhanced_radiation_map(mesh_data, mesh_dimensions, mesh_bounds, channel_diameter, energy, filename):
    """
    Create enhanced visualization of radiation distribution outside the wall with optimized
    visualization for this specific shielding problem.

    Args:
        mesh_data (numpy.ndarray): Fine mesh tally data
        mesh_dimensions (list): Mesh dimensions [nx, ny, nz]
        mesh_bounds (list): Mesh bounds [xmin, xmax, ymin, ymax]
        channel_diameter (float): Diameter of the air channel in cm
        energy (float): Gamma-ray energy in MeV
        filename (str): Output filename
    """
    logger.info(f"Creating enhanced close-up radiation map for energy {energy} MeV and channel diameter {channel_diameter} cm...")

    # Reshape mesh data
    nx, ny, nz = mesh_dimensions
    xmin, xmax, ymin, ymax = mesh_bounds

    # Extract central slice (z=0)
    central_slice = mesh_data.reshape(nx, ny, nz)[:, :, nz//2]

    # Create coordinate meshes
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Apply adaptive smoothing based on channel diameter
    # Smaller channels need more smoothing to enhance visibility
    sigma = max(0.5, channel_diameter / 2)  # Adaptive smoothing factor
    logger.info(f"Using adaptive smoothing sigma={sigma} for channel diameter {channel_diameter} cm")
    smoothed_data = gaussian_filter(central_slice, sigma=sigma)

    # Create a symmetrical data set for full 360-degree view
    # Mirror the data across the x-axis for symmetry
    X_full = X.copy()
    Y_top = Y.copy()
    Y_bottom = -Y.copy()  # Mirror across x-axis
    data_top = smoothed_data.copy()
    data_bottom = smoothed_data.copy()  # Same data, mirrored location

    # Apply log scale for better visualization
    with np.errstate(divide='ignore', invalid='ignore'):
        log_data_top = np.log10(data_top)
        log_data_bottom = np.log10(data_bottom)

    # Replace -inf with minimum finite value
    min_val_top = np.min(log_data_top[np.isfinite(log_data_top)])
    log_data_top[~np.isfinite(log_data_top)] = min_val_top - 1
    
    min_val_bottom = np.min(log_data_bottom[np.isfinite(log_data_bottom)])
    log_data_bottom[~np.isfinite(log_data_bottom)] = min_val_bottom - 1
    
    min_val = min(min_val_top, min_val_bottom)

    # Create figure with optimized size
    plt.figure(figsize=(14, 16))  # Taller figure to accommodate full view

    # Use a scientific colormap that's perceptually uniform
    cmap = plt.cm.viridis
    
    # Create a custom normalization that emphasizes important flux regions
    # Find key values for color scaling
    with np.errstate(divide='ignore', invalid='ignore'):
        vmin = min_val
        vmax = max(np.max(log_data_top), np.max(log_data_bottom))
        vmid = (vmin + vmax) / 2
        # Create a custom norm that emphasizes mid-range values
        norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=vmid, vmax=vmax)
    
    # Plot radiation map with enhanced color scale
    plt.pcolormesh(X_full, Y_top, log_data_top, cmap=cmap, norm=norm, shading='auto')
    plt.pcolormesh(X_full, Y_bottom, log_data_bottom, cmap=cmap, norm=norm, shading='auto')

    # Add detailed colorbar with scientific formatting
    cbar = plt.colorbar(extend='both')
    cbar.set_label('Log₁₀(Flux) [particles/cm²/source particle]', fontsize=12)
    
    # Add tick labels with scientific notation
    cbar.ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
    cbar.ax.tick_params(labelsize=10)

    # Add shield outline with improved styling
    plt.axvline(x=CONCRETE_THICKNESS, color='white', linestyle='-', linewidth=2, label='Shield')

    # Add channel position with highlighting
    channel_radius = channel_diameter / 2.0
    circle = plt.Circle((CONCRETE_THICKNESS, 0), channel_radius, fill=True, 
                      edgecolor='yellow', facecolor='yellow', alpha=0.7, linewidth=2, label='Channel')
    plt.gca().add_patch(circle)

    # Add distance markers with clear labels
    distances = [10, 20, 30, 50, 100]
    for d in distances:
        circle = plt.Circle((CONCRETE_THICKNESS, 0), d, fill=False, 
                          edgecolor='white', linestyle='--', linewidth=1)
        plt.gca().add_patch(circle)
        # Add label on the positive x-axis
        plt.text(CONCRETE_THICKNESS + d, 0, f'{d} cm', color='white', 
               ha='left', va='center', fontsize=10, 
               bbox=dict(facecolor='black', alpha=0.7, boxstyle='round'))

    # Add angle indicators with improved clarity for both top and bottom halves
    angles = [-45, -30, -15, 0, 15, 30, 45]
    for angle in angles:
        angle_rad = math.radians(angle)
        dx = 75 * math.cos(angle_rad)
        dy = 75 * math.sin(angle_rad)
        # Draw angle lines
        plt.plot([CONCRETE_THICKNESS, CONCRETE_THICKNESS + dx], [0, dy], 
                'w--', linewidth=1, alpha=0.7)
        # Add angle labels
        plt.text(CONCRETE_THICKNESS + dx + 2, dy, f'{angle} degrees', 
               color='white', ha='left', va='center', fontsize=10,
               bbox=dict(facecolor='black', alpha=0.7, boxstyle='round'))

    # Highlight high-dose regions for better visibility
    high_dose_threshold_top = np.percentile(log_data_top, 95)
    high_dose_mask_top = log_data_top > high_dose_threshold_top
    plt.contour(X_full, Y_top, high_dose_mask_top, levels=[0.5], 
              colors='red', linewidths=2, linestyles='solid')
              
    high_dose_threshold_bottom = np.percentile(log_data_bottom, 95)
    high_dose_mask_bottom = log_data_bottom > high_dose_threshold_bottom
    plt.contour(X_full, Y_bottom, high_dose_mask_bottom, levels=[0.5], 
              colors='red', linewidths=2, linestyles='solid')
              
    # Add a text label for high dose regions
    plt.annotate('High dose region (95th percentile)', 
               xy=(CONCRETE_THICKNESS + 60, 40),
               xytext=(CONCRETE_THICKNESS + 60, 60),
               color='red', fontsize=10, fontweight='bold',
               bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'),
               arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

    # Add a detailed title with scientific information
    plt.title(f'Enhanced Radiation Distribution Outside Shield Wall\n'
             f'Energy: {energy} MeV, Channel Diameter: {channel_diameter} cm',
            fontsize=14, fontweight='bold')

    # Add axis labels with units
    plt.xlabel('Distance from Source (cm)', fontsize=12)
    plt.ylabel('Lateral Distance (cm)', fontsize=12)

    # Set aspect ratio to equal for proper spatial representation
    plt.axis('equal')

    # Set limits to focus on area immediately outside the wall
    plt.xlim(CONCRETE_THICKNESS - 5, CONCRETE_THICKNESS + 100)
    plt.ylim(-80, 80)  # Extended to show bottom half

    # Add grid for better reference
    plt.grid(True, linestyle='--', alpha=0.3, color='gray')

    # Add legend with improved positioning and styling
    plt.legend(loc='upper right', framealpha=0.7, fontsize=10)

    # Save figure with high resolution
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also create a polar version of the radiation map
    create_polar_radiation_map(mesh_data, mesh_dimensions, mesh_bounds, channel_diameter, energy, 
                             str(filename).replace('.png', '_polar.png'))

def create_polar_radiation_map(mesh_data, mesh_dimensions, mesh_bounds, channel_diameter, energy, filename):
    """
    Create a polar visualization of the radiation distribution.
    
    Args:
        mesh_data (numpy.ndarray): Fine mesh tally data
        mesh_dimensions (list): Mesh dimensions [nx, ny, nz]
        mesh_bounds (list): Mesh bounds [xmin, xmax, ymin, ymax]
        channel_diameter (float): Diameter of the air channel in cm
        energy (float): Gamma-ray energy in MeV
        filename (str): Output filename
    """
    logger.info(f"Creating polar radiation map for energy {energy} MeV and channel diameter {channel_diameter} cm...")
    
    # Reshape mesh data
    nx, ny, nz = mesh_dimensions
    xmin, xmax, ymin, ymax = mesh_bounds

    # Extract central slice (z=0)
    central_slice = mesh_data.reshape(nx, ny, nz)[:, :, nz//2]
    
    # Apply adaptive smoothing
    sigma = max(1, 3 * channel_diameter)
    smoothed_data = gaussian_filter(central_slice, sigma=sigma)
    
    # Convert to polar coordinates
    # First, create a regular grid in Cartesian coordinates
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Calculate r and theta for each point
    r = np.sqrt((X - CONCRETE_THICKNESS)**2 + Y**2)
    theta = np.arctan2(Y, X - CONCRETE_THICKNESS)
    
    # Prepare data for interpolation
    points = np.column_stack((r.flatten(), theta.flatten()))
    values = smoothed_data.flatten()
    
    # Create a regular grid in polar coordinates
    r_grid = np.linspace(0, 100, 100)  # Distances from 0 to 100 cm
    theta_grid = np.linspace(-np.pi, np.pi, 180)  # Full 360 degrees
    R, THETA = np.meshgrid(r_grid, theta_grid)
    
    # Interpolate the data to the regular polar grid
    from scipy.interpolate import griddata
    polar_data = griddata(points, values, (R.flatten(), THETA.flatten()), method='cubic', fill_value=1e-10)
    polar_data = polar_data.reshape(R.shape)
    
    # Apply log scale
    with np.errstate(divide='ignore', invalid='ignore'):
        log_polar = np.log10(polar_data)
    
    # Replace -inf with minimum finite value
    min_val = np.min(log_polar[np.isfinite(log_polar)])
    log_polar[~np.isfinite(log_polar)] = min_val
    
    # Create the polar plot
    plt.figure(figsize=(12, 12))
    ax = plt.subplot(111, polar=True)
    
    # Plot the heatmap
    cax = ax.pcolormesh(THETA, R, log_polar, cmap='viridis', shading='auto')
    
    # Add colorbar
    cbar = plt.colorbar(cax, ax=ax, pad=0.1)
    cbar.set_label('Log10(Flux) [particles/cm²]')
    
    # Add radial grid lines
    distances = [10, 20, 30, 50, 100]
    for d in distances:
        if d <= max(r_grid):
            ax.plot(theta_grid, [d] * len(theta_grid), 'k--', linewidth=0.5, alpha=0.3)
            ax.text(0, d, f'{d} cm', ha='center', va='bottom', fontsize=8, 
                  bbox=dict(facecolor='white', alpha=0.6))
    
    # Add angle lines
    angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    for angle in angles:
        angle_rad = math.radians(angle)
        ax.plot([angle_rad, angle_rad], [0, max(r_grid)], 'k--', linewidth=0.5, alpha=0.3)
        if angle <= 180:
            ax.text(angle_rad, max(r_grid) * 1.05, f'{angle} degrees', 
                  ha='center', va='center', fontsize=8,
                  bbox=dict(facecolor='white', alpha=0.6))
    
    # Set title
    plt.title(f'Polar Radiation Distribution (E={energy} MeV, Channel Ø={channel_diameter} cm)',
             fontsize=14, fontweight='bold')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_dose_vs_angle(results, energy, filename):
    """
    Create visualization of dose rate vs. angle for different distances.

    Args:
        results (dict): Results dictionary
        energy (float): Gamma-ray energy in MeV
        filename (str): Output filename
    """
    logger.info(f"Creating dose vs angle visualization for energy {energy} MeV...")

    # Check if we have results for this energy
    energy_str = str(energy)
    if energy_str not in results or not results[energy_str]:
        logger.warning(f"No results available for energy {energy} MeV")
        return

    # Create figure
    plt.figure(figsize=(14, 10))

    # Plot dose vs angle for each distance and channel diameter
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*']
    colors = plt.cm.viridis(np.linspace(0, 1, len(CHANNEL_DIAMETERS)))

    for i, diameter in enumerate(CHANNEL_DIAMETERS):
        diameter_str = str(diameter)

        # Check if we have results for this diameter
        if diameter_str not in results[energy_str]:
            logger.warning(f"No results available for diameter {diameter} cm")
            continue

        for j, distance in enumerate(DETECTOR_DISTANCES):
            angles = []
            doses = []

            for angle in DETECTOR_ANGLES:
                key = (distance, angle)
                key_str = str(key)

                # Check if we have results for this detector position
                if (key_str in results[energy_str][diameter_str]['detector_results'] and 
                    'dose' in results[energy_str][diameter_str]['detector_results'][key_str]):
                    dose = results[energy_str][diameter_str]['detector_results'][key_str]['dose']
                    angles.append(angle)
                    doses.append(dose)

            if angles and doses:  # Only plot if we have data
                marker_idx = j % len(markers)
                plt.semilogy(angles, doses, marker=markers[marker_idx], linestyle='-', 
                             color=colors[i], label=f'D={diameter} cm, r={distance} cm')

    # Add labels and title
    plt.xlabel('Detector Angle [degrees]')
    plt.ylabel('Dose Rate [rem/hr]')
    plt.title(f'Dose Rate vs. Angle (E={energy} MeV)')

    # Add grid
    plt.grid(True, linestyle='--', alpha=0.5)

    # Add legend (if we have any data)
    if plt.gca().get_legend_handles_labels()[0]:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Save figure
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_dose_heatmap(results, energy, channel_diameter, filename):
    """
    Create 2D dose distribution heat map showing the full 360-degree radiation pattern.

    Args:
        results (dict): Results dictionary
        energy (float): Gamma-ray energy in MeV
        channel_diameter (float): Diameter of the air channel in cm
        filename (str): Output filename
    """
    logger.info(f"Creating dose heatmap for energy {energy} MeV and channel diameter {channel_diameter} cm...")

    try:
        # Extract dose data
        x_coords = []
        y_coords = []
        doses = []

        # First collect all detector positions with dose data
        positions_with_data = {}
        for key_str, data in results[str(energy)][str(channel_diameter)]['detector_results'].items():
            if 'dose' in data:
                key = eval(key_str)  # Convert string tuple back to tuple
                distance, angle = key
                positions_with_data[(distance, angle)] = data['dose']

        # Check if we have any data
        if not positions_with_data:
            raise ValueError("No detector dose data available for visualization")

        # Now generate symmetric data including both positive and negative angles
        for distance in DETECTOR_DISTANCES:
            for angle in DETECTOR_ANGLES:
                # Check if we have data for this position
                if (distance, angle) in positions_with_data:
                    dose = positions_with_data[(distance, angle)]
                    
                    # Calculate detector position for positive angle
                    angle_rad = math.radians(angle)
                    x = CONCRETE_THICKNESS + distance * math.cos(angle_rad)
                    y = distance * math.sin(angle_rad)

                    # Add positive angle point
                    x_coords.append(x)
                    y_coords.append(y)
                    doses.append(dose)
        
                    # Add symmetric point for negative angle (except for 0 degrees)
                    if angle > 0:
                        x_coords.append(x)
                        y_coords.append(-y)
                        doses.append(dose)
                
                # If we don't have data for this position but have for the symmetric one
                elif angle > 0 and (distance, -angle) in positions_with_data:
                    dose = positions_with_data[(distance, -angle)]
                    
                    # Calculate detector position for this angle
                    angle_rad = math.radians(angle)
                    x = CONCRETE_THICKNESS + distance * math.cos(angle_rad)
                    y = distance * math.sin(angle_rad)
                    
                    # Add positive angle point
                    x_coords.append(x)
                    y_coords.append(y)
                    doses.append(dose)
                    
                    # Add symmetric point for negative angle
                    x_coords.append(x)
                    y_coords.append(-y)
                    doses.append(dose)

        # Check if we have enough data points for interpolation (need at least 4)
        if len(x_coords) < 4:
            raise ValueError(f"Insufficient detector data points ({len(x_coords)}) for interpolation")

        # Create grid for interpolation with finer resolution
        xi = np.linspace(CONCRETE_THICKNESS, CONCRETE_THICKNESS + 160, 400)
        yi = np.linspace(-120, 120, 480)  # Full -120 to 120 range for both top and bottom
        Xi, Yi = np.meshgrid(xi, yi)

        # Interpolate dose data using cubic interpolation for smoother visualization
        try:
            Zi = griddata((x_coords, y_coords), doses, (Xi, Yi), method='cubic', fill_value=np.min(doses)/100)
        except Exception as e:
            logger.warning(f"Cubic interpolation failed: {e}. Falling back to linear interpolation.")
            Zi = griddata((x_coords, y_coords), doses, (Xi, Yi), method='linear', fill_value=np.min(doses)/100)

        # Apply smoothing to enhance visualization
        sigma = max(1.0, channel_diameter * 2)  # Adaptive smoothing
        Zi_smooth = gaussian_filter(Zi, sigma=sigma)

        # Apply log scale for better visualization
        with np.errstate(divide='ignore', invalid='ignore'):
            log_Zi = np.log10(Zi_smooth)

        # Replace -inf with minimum finite value
        finite_mask = np.isfinite(log_Zi)
        if np.any(finite_mask):
            min_val = np.min(log_Zi[finite_mask])
            log_Zi[~finite_mask] = min_val
        else:
            # If no finite values, set a default minimum
            min_val = -10
            log_Zi.fill(min_val)

        # Create figure with improved styling
        plt.figure(figsize=(12, 14))

        # Plot dose heatmap with improved colormap
        cmap = plt.cm.viridis
        
        # Create a custom colormap normalization
        vmin = min_val
        vmax = np.max(log_Zi)
        vmid = (vmin + vmax) / 2
        norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=vmid, vmax=vmax)
        
        im = plt.pcolormesh(Xi, Yi, log_Zi, cmap=cmap, norm=norm, shading='auto')

        # Add colorbar with better formatting
        cbar = plt.colorbar(im, extend='both')
        cbar.set_label('Log₁₀(Dose Rate) [rem/hr]', fontsize=12)

        # Add shield outline
        plt.axvline(x=CONCRETE_THICKNESS, color='white', linestyle='-', linewidth=2, label='Shield')

        # Add channel position
        channel_radius = channel_diameter / 2.0
        circle = Circle((CONCRETE_THICKNESS, 0), channel_radius, fill=True, 
                      edgecolor='yellow', facecolor='yellow', alpha=0.7, linewidth=2, label='Channel')
        plt.gca().add_patch(circle)

        # Add distance markers as circles with better visibility
        distances = [30, 50, 100, 150]
        for d in distances:
            circle = Circle((CONCRETE_THICKNESS, 0), d, fill=False, 
                          edgecolor='white', linestyle='--', linewidth=1)
            plt.gca().add_patch(circle)
            # Only add text label on positive x-axis to avoid clutter
            plt.text(CONCRETE_THICKNESS + d, 0, f'{d} cm', color='white', 
                   ha='left', va='center', fontsize=10, 
                   bbox=dict(facecolor='black', alpha=0.7, boxstyle='round'))

        # Add angle indicators for both top and bottom halves
        all_angles = [-45, -30, -15, 0, 15, 30, 45]
        for angle in all_angles:
            angle_rad = math.radians(angle)
            dx = 120 * math.cos(angle_rad)
            dy = 120 * math.sin(angle_rad)
            plt.plot([CONCRETE_THICKNESS, CONCRETE_THICKNESS + dx], [0, dy], 
                    'w--', linewidth=1, alpha=0.7)
            plt.text(CONCRETE_THICKNESS + dx + 5, dy, f'{angle} degrees', 
                   color='white', ha='left', va='center', fontsize=10,
                   bbox=dict(facecolor='black', alpha=0.7, boxstyle='round'))

        # Add detector points with improved styling
        scatter = plt.scatter(x_coords, y_coords, c='white', s=25, edgecolor='black', alpha=0.7, label='Detector Points')

        # Highlight high-dose regions
        high_dose_threshold = np.percentile(log_Zi, 90)
        high_dose_contour = plt.contour(Xi, Yi, log_Zi, levels=[high_dose_threshold], 
                                      colors='red', linewidths=2, linestyles='solid')
        plt.clabel(high_dose_contour, inline=True, fontsize=8, fmt='High dose region')

        # Add labels and title
        plt.xlabel('X [cm]', fontsize=12)
        plt.ylabel('Y [cm]', fontsize=12)
        plt.title(f'Full Radiation Dose Distribution (E={energy} MeV, Channel Ø={channel_diameter} cm)', 
                fontsize=14, fontweight='bold')

        # Set aspect ratio to equal
        plt.axis('equal')

        # Set limits to focus on area of interest including both top and bottom halves
        plt.xlim(CONCRETE_THICKNESS - 5, CONCRETE_THICKNESS + 160)
        plt.ylim(-120, 120)

        # Add grid lines for better reference
        plt.grid(True, linestyle='--', alpha=0.3)

        # Add legend with improved positioning
        plt.legend(loc='upper right', framealpha=0.7)

        # Save figure with improved quality
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Dose heatmap saved to {filename}")
    
        # Create a polar heatmap version
        create_polar_heatmap(results, energy, channel_diameter, str(filename).replace('.png', '_polar.png'))
        
    except Exception as e:
        logger.error(f"Error creating dose heatmap: {e}")
        logger.error(traceback.format_exc())
        
        # Create a simple placeholder visualization
        plt.figure(figsize=(10, 8))
        
        # Draw geometry
        plt.axvline(x=CONCRETE_THICKNESS, color='gray', linestyle='-', linewidth=2)
        
        # Draw channel
        channel_radius = channel_diameter / 2.0
        circle = Circle((CONCRETE_THICKNESS, 0), channel_radius, fill=True, 
                      color='yellow', alpha=0.7, linewidth=2)
        plt.gca().add_patch(circle)
        
        # Add distance circles
        for d in [30, 50, 100, 150]:
            circle = Circle((CONCRETE_THICKNESS, 0), d, fill=False, 
                          edgecolor='black', linestyle='--', linewidth=1)
            plt.gca().add_patch(circle)
            
        # Add message about insufficient data
        plt.text(CONCRETE_THICKNESS + 80, 0, 
                "Insufficient dose data for visualization\nTry running with more particles", 
                ha='center', fontsize=14, 
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
        
        plt.xlabel('X [cm]', fontsize=12)
        plt.ylabel('Y [cm]', fontsize=12)
        plt.title(f'Dose Distribution (E={energy} MeV, Channel Ø={channel_diameter} cm) - No Data',
                fontsize=14, fontweight='bold')
        plt.axis('equal')
        plt.xlim(CONCRETE_THICKNESS - 5, CONCRETE_THICKNESS + 160)
        plt.ylim(-120, 120)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

def create_polar_heatmap(results, energy, channel_diameter, filename):
    """
    Create a polar heatmap showing the full 360-degree radiation dose distribution.
    
    Args:
        results (dict): Results dictionary
        energy (float): Gamma-ray energy in MeV
        channel_diameter (float): Diameter of the air channel in cm
        filename (str): Output filename
    """
    logger.info(f"Creating polar dose heatmap for energy {energy} MeV and channel diameter {channel_diameter} cm...")
    
    # Extract dose data and convert to polar coordinates
    r_data = []
    theta_data = []
    dose_data = []
    
    # First collect all detector positions with dose data
    positions_with_data = {}
    for key_str, data in results[str(energy)][str(channel_diameter)]['detector_results'].items():
        if 'dose' in data:
            key = eval(key_str)  # Convert string tuple back to tuple
            distance, angle = key
            positions_with_data[(distance, angle)] = data['dose']
    
    # Get all unique distances and angles
    distances = sorted(list(set([d for d, a in positions_with_data.keys()])))
    angles = sorted(list(set([a for d, a in positions_with_data.keys()])))
    
    # Generate full 360-degree data including both positive and negative angles
    for distance in distances:
        for angle_deg in range(-180, 181, 5):  # Full 360 degrees with 5-degree increments
            # Find the closest measured angle
            closest_angle = min(angles, key=lambda a: abs(a - abs(angle_deg)))
            
            # Determine the sign based on the original angle
            sign = -1 if angle_deg < 0 else 1
            lookup_angle = sign * closest_angle
            
            # Normalize lookup angle to be within measured angles
            if lookup_angle < min(angles):
                lookup_angle = abs(lookup_angle)
            elif lookup_angle > max(angles):
                lookup_angle = max(angles)
            
            # Get dose value if available, otherwise use symmetric value
            if (distance, lookup_angle) in positions_with_data:
                dose = positions_with_data[(distance, lookup_angle)]
            elif (distance, -lookup_angle) in positions_with_data:
                dose = positions_with_data[(distance, -lookup_angle)]
            else:
                # Skip if no data available
                continue
            
            # Convert to polar coordinates
            theta = math.radians(angle_deg)
            r_data.append(distance)
            theta_data.append(theta)
            dose_data.append(dose)
    
    # Convert to numpy arrays
    r_data = np.array(r_data)
    theta_data = np.array(theta_data)
    dose_data = np.array(dose_data)
    
    # Create a regular polar grid
    min_distance = min(distances) if distances else 30
    max_distance = max(distances) if distances else 150
    
    r_grid = np.linspace(min_distance, max_distance, 100)
    theta_grid = np.linspace(-np.pi, np.pi, 360)  # Full 360 degrees with 1-degree resolution
    
    # Create meshgrid for polar coordinates
    R, THETA = np.meshgrid(r_grid, theta_grid)
    
    # Interpolate the scattered data onto the grid with better handling for sparse data
    try:
        # Use cubic interpolation for smoother results
        LOG_DOSE = griddata((r_data, theta_data), np.log10(dose_data), (R, THETA), method='cubic')
        
        # Fill any remaining NaN values using nearest-neighbor interpolation
        nan_mask = np.isnan(LOG_DOSE)
        if np.any(nan_mask):
            nn_interp = griddata((r_data, theta_data), np.log10(dose_data), (R, THETA), method='nearest')
            LOG_DOSE[nan_mask] = nn_interp[nan_mask]
    except Exception as e:
        logger.warning(f"Cubic interpolation failed: {e}. Using linear interpolation.")
        LOG_DOSE = griddata((r_data, theta_data), np.log10(dose_data), (R, THETA), method='linear')
        
        # Fill any remaining NaN values
        nan_mask = np.isnan(LOG_DOSE)
        if np.any(nan_mask):
            nn_interp = griddata((r_data, theta_data), np.log10(dose_data), (R, THETA), method='nearest')
            LOG_DOSE[nan_mask] = nn_interp[nan_mask]
    
    # Apply smoothing for better visualization
    sigma = max(1.0, 3.0 / channel_diameter)  # More smoothing for smaller channels
    LOG_DOSE = gaussian_filter(LOG_DOSE, sigma=[sigma, sigma])
    
    # Create the polar plot with improved styling
    plt.figure(figsize=(12, 12))
    ax = plt.subplot(111, polar=True)
    
    # Use a better colormap with custom normalization
    cmap = plt.cm.viridis
    
    # Find min and max values for colorbar
    vmin = np.min(LOG_DOSE)
    vmax = np.max(LOG_DOSE)
    vmid = (vmin + vmax) / 2
    
    # Create a two-slope normalization for better contrast
    norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=vmid, vmax=vmax)
    
    # Plot the heatmap
    cax = ax.pcolormesh(THETA, R, LOG_DOSE, cmap=cmap, norm=norm, shading='auto')
    
    # Customize polar plot appearance
    ax.set_theta_zero_location("E")  # 0 degrees at the right (East)
    ax.set_theta_direction(-1)  # Clockwise
    
    # Add radial grid lines for each distance with improved styling
    for d in [30, 40, 60, 80, 100, 150]:
        if d >= min_distance and d <= max_distance:
            # Draw the distance circle
            ax.plot(theta_grid, [d] * len(theta_grid), 
                   color='white', linestyle='--', linewidth=0.7, alpha=0.5)
            
            # Add labels at cardinal directions for better readability
            for angle in [0, 90, 180, 270]:
                theta = math.radians(angle)
                ha = 'center'
                if angle == 0:
                    ha = 'left'
                elif angle == 180:
                    ha = 'right'
                
                ax.text(theta, d, f'{d} cm', 
                       color='white', ha=ha, va='center', fontsize=8,
                       bbox=dict(facecolor='black', alpha=0.7, boxstyle='round'))
    
    # Add angle lines with clear labels
    for angle in [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]:
        theta = math.radians(angle)
        ax.plot([theta, theta], [min_distance, max_distance], 
               color='white', linestyle='--', linewidth=0.7, alpha=0.5)
        
        # Add angle labels at the outer edge
        if angle % 45 == 0:  # Only show labels for major angles
            # Position label just outside the outermost circle
            r_pos = max_distance * 1.05
            
            ax.text(theta, r_pos, f'{angle} degrees', 
                   color='black', ha='center', va='center', fontsize=9,
                   bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
    
    # Add detector points
    for r, theta, dose in zip(r_data, theta_data, dose_data):
        ax.scatter(theta, r, c='white', s=10, edgecolor='black', alpha=0.5)
    
    # Add a colorbar with better formatting
    cbar = plt.colorbar(cax, ax=ax, pad=0.08)
    cbar.set_label('Log₁₀(Dose Rate) [rem/hr]', fontsize=12)
    
    # Add a title with improved scientific information
    plt.title(f'Polar Radiation Dose Distribution\nE={energy} MeV, Channel Ø={channel_diameter} cm',
             fontsize=14, fontweight='bold')
    
    # Add channel annotation
    ax.annotate('Channel Exit', 
               xy=(0, 0),  # At origin
               xytext=(0, min_distance/2),  # Text position
               color='yellow',
               fontweight='bold',
               ha='center',
               va='center',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7))
    
    # Add high dose region contour
    high_dose_threshold = np.percentile(LOG_DOSE, 90)
    ax.contour(THETA, R, LOG_DOSE, levels=[high_dose_threshold], 
              colors='red', linewidths=2)
    
    # Add annotation for high dose region
    high_dose_angle = 0  # Assume high dose is along 0 degrees
    high_dose_distance = distances[len(distances)//2]  # Middle distance
    ax.annotate('High Dose Region', 
               xy=(high_dose_angle, high_dose_distance),
               xytext=(math.radians(30), high_dose_distance),
               color='red',
               arrowprops=dict(arrowstyle='->',
                             connectionstyle='arc3',
                             color='red'),
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))
    
    # Save the figure with high resolution
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Polar dose heatmap saved to {filename}")

def visualize_flux_spectra(results, energy, channel_diameter, filename):
    """
    Visualize gamma-ray flux spectra exiting the shield.

    Args:
        results (dict): Results dictionary
        energy (float): Gamma-ray energy in MeV
        channel_diameter (float): Diameter of the air channel in cm
        filename (str): Output filename
    """
    logger.info(f"Creating flux spectra visualization for energy {energy} MeV and channel diameter {channel_diameter} cm...")

    # Extract energy bins and flux data
    energy_bins = results[str(energy)][str(channel_diameter)]['energy_bins']
    flux_spectrum = results[str(energy)][str(channel_diameter)]['flux_spectrum']

    # Create figure
    plt.figure(figsize=(10, 6))

    # Plot flux spectrum
    plt.step(energy_bins[:-1], flux_spectrum, where='post', linewidth=2)

    # Add labels and title
    plt.xlabel('Energy [MeV]')
    plt.ylabel('Flux [particles/cm²]')
    plt.title(f'Gamma-Ray Flux Spectrum (E={energy} MeV, Channel Ø={channel_diameter} cm)')

    # Set log scales
    plt.xscale('log')
    plt.yscale('log')

    # Add grid
    plt.grid(True, which='both', linestyle='--', alpha=0.5)

    # Add source energy marker
    plt.axvline(x=energy, color='red', linestyle='--', label=f'Source Energy ({energy} MeV)')

    # Add legend
    plt.legend()

    # Save figure
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def analyze_critical_parameters(results):
    """
    Analyze critical parameters from simulation results.

    Args:
        results (dict): Results dictionary

    Returns:
        dict: Critical parameters analysis
    """
    logger.info("Analyzing critical parameters...")

    # Initialize critical parameters
    critical_analysis = {
        'dose_threshold': 0.1,  # rem/hr
        'max_dose': 0.0,
        'critical_energy': 0.0,
        'critical_diameter': 0.0,
        'critical_combinations': [],
        'dose_vs_energy': {},   # Add this for compatibility with visualize_critical_parameters
        'dose_vs_diameter': {}  # Add this for compatibility with visualize_critical_parameters
    }

    # Analyze dose rates for all parameter combinations
    for energy_str, energy_data in results.items():
        if not energy_data:  # Skip if no data for this energy
            continue

        energy = float(energy_str)
        critical_analysis['dose_vs_energy'][energy_str] = {}
        
        if energy_str not in critical_analysis['dose_vs_diameter']:
            critical_analysis['dose_vs_diameter'][energy_str] = {}

        for diameter_str, diameter_data in energy_data.items():
            if 'detector_results' not in diameter_data:  # Skip if no detector results
                continue

            diameter = float(diameter_str)
            critical_analysis['dose_vs_energy'][energy_str][diameter_str] = 0.0

            # Find maximum dose for this energy-diameter combination
            max_dose_for_combo = 0.0

            for key_str, data in diameter_data['detector_results'].items():
                if 'dose' in data:
                    dose = data['dose']
                    max_dose_for_combo = max(max_dose_for_combo, dose)
            
            # Save max dose for this combination
            critical_analysis['dose_vs_energy'][energy_str][diameter_str] = max_dose_for_combo
            critical_analysis['dose_vs_diameter'][energy_str][diameter_str] = max_dose_for_combo

            # Update critical parameters if this is the highest dose so far
            if max_dose_for_combo > critical_analysis['max_dose']:
                critical_analysis['max_dose'] = max_dose_for_combo
                critical_analysis['critical_energy'] = energy
                critical_analysis['critical_diameter'] = diameter

            # Add to critical combinations if above threshold
            if max_dose_for_combo > critical_analysis['dose_threshold']:
                critical_analysis['critical_combinations'].append({
                    'energy': energy,
                    'diameter': diameter,
                    'dose': max_dose_for_combo
                })

    # Sort critical combinations by dose rate (descending)
    critical_analysis['critical_combinations'].sort(key=lambda x: x['dose'], reverse=True)

    return critical_analysis

def visualize_critical_parameters(critical_analysis, filename_energy, filename_diameter):
    """
    Visualize critical parameters analysis.

    Args:
        critical_analysis (dict): Critical parameters analysis
        filename_energy (str): Output filename for energy analysis
        filename_diameter (str): Output filename for diameter analysis
    """
    logger.info("Creating critical parameters visualizations...")

    # Visualize dose vs energy
    plt.figure(figsize=(10, 6))

    for diameter in CHANNEL_DIAMETERS:
        diameter_str = str(diameter)
        energies = []
        doses = []

        for energy_str, diameter_data in critical_analysis['dose_vs_energy'].items():
            if diameter_str in diameter_data:
                energies.append(float(energy_str))
                doses.append(diameter_data[diameter_str])

        if energies and doses:
            plt.loglog(energies, doses, marker='o', label=f'Ø={diameter} cm')

    # Add threshold line
    plt.axhline(y=critical_analysis['dose_threshold'], color='red', linestyle='--', 
                label=f'Threshold ({critical_analysis["dose_threshold"]} rem/hr)')

    # Add labels and title
    plt.xlabel('Gamma-Ray Energy [MeV]')
    plt.ylabel('Dose Rate at 100 cm [rem/hr]')
    plt.title('Dose Rate vs. Gamma-Ray Energy')

    # Add grid
    plt.grid(True, which='both', linestyle='--', alpha=0.5)

    # Add legend
    plt.legend()

    # Save figure
    plt.tight_layout()
    plt.savefig(filename_energy, dpi=300, bbox_inches='tight')
    plt.close()

    # Visualize dose vs diameter
    plt.figure(figsize=(10, 6))

    for energy in GAMMA_ENERGIES:
        energy_str = str(energy)
        if energy_str in critical_analysis['dose_vs_diameter']:
            diameters = []
            doses = []

            for diameter_str, dose in critical_analysis['dose_vs_diameter'][energy_str].items():
                diameters.append(float(diameter_str))
                doses.append(dose)

            if diameters and doses:
                plt.loglog(diameters, doses, marker='o', label=f'E={energy} MeV')

    # Add threshold line
    plt.axhline(y=critical_analysis['dose_threshold'], color='red', linestyle='--', 
                label=f'Threshold ({critical_analysis["dose_threshold"]} rem/hr)')

    # Add labels and title
    plt.xlabel('Channel Diameter [cm]')
    plt.ylabel('Dose Rate at 100 cm [rem/hr]')
    plt.title('Dose Rate vs. Channel Diameter')

    # Add grid
    plt.grid(True, which='both', linestyle='--', alpha=0.5)

    # Add legend
    plt.legend()

    # Save figure
    plt.tight_layout()
    plt.savefig(filename_diameter, dpi=300, bbox_inches='tight')
    plt.close()


def create_summary_report(results, critical_analysis, filename):
    """
    Create a summary report of the simulation results.

    Args:
        results (dict): Results dictionary
        critical_analysis (dict): Critical parameters analysis
        filename (str): Output filename
    """
    logger.info("Creating summary report...")

    with open(filename, 'w') as f:
        f.write("# Gamma-Ray Shielding Simulation Summary Report\n\n")

        f.write("## Simulation Parameters\n\n")
        f.write(f"- Concrete Wall Thickness: {CONCRETE_THICKNESS/30.48:.1f} ft ({CONCRETE_THICKNESS:.1f} cm)\n")
        f.write(f"- Source Distance: {SOURCE_DISTANCE/30.48:.1f} ft ({SOURCE_DISTANCE:.1f} cm)\n")
        f.write(f"- Channel Diameters: {', '.join([f'{d:.2f} cm' for d in CHANNEL_DIAMETERS])}\n")
        f.write(f"- Gamma-Ray Energies: {', '.join([f'{e:.1f} MeV' for e in GAMMA_ENERGIES])}\n")
        f.write(f"- Detector Distances: {', '.join([f'{d} cm' for d in DETECTOR_DISTANCES])}\n")
        f.write(f"- Detector Angles: {', '.join([f'{a} degrees' for a in DETECTOR_ANGLES])}\n\n")

        f.write("## Critical Parameters Analysis\n\n")
        f.write(f"- Dose Threshold: {critical_analysis['dose_threshold']:.2f} rem/hr\n")
        f.write(f"- Maximum Dose: {critical_analysis['max_dose']:.4f} rem/hr\n")
        f.write(f"- Critical Energy: {critical_analysis['critical_energy']:.1f} MeV\n")
        f.write(f"- Critical Channel Diameter: {critical_analysis['critical_diameter']:.2f} cm\n\n")

        f.write("### Critical Energy-Diameter Combinations\n\n")
        f.write("| Energy (MeV) | Diameter (cm) | Dose Rate (rem/hr) |\n")
        f.write("|-------------|---------------|--------------------|\n")

        for combo in critical_analysis['critical_combinations']:
            f.write(f"| {combo['energy']:.1f} | {combo['diameter']:.2f} | {combo['dose']:.4f} |\n")

        f.write("\n## Dose Rate Summary\n\n")

        for energy in GAMMA_ENERGIES:
            energy_str = str(energy)
            f.write(f"### Energy: {energy} MeV\n\n")

            # Skip energy if not in results
            if energy_str not in results:
                f.write(f"No data available for energy {energy} MeV\n\n")
                continue

            for diameter in CHANNEL_DIAMETERS:
                diameter_str = str(diameter)
                f.write(f"#### Channel Diameter: {diameter} cm\n\n")
                
                # Skip diameter if not in results for this energy
                if diameter_str not in results[energy_str]:
                    f.write(f"No data available for diameter {diameter} cm\n\n")
                    continue
                
                # Skip if no detector results
                if 'detector_results' not in results[energy_str][diameter_str]:
                    f.write(f"No detector results available for energy {energy} MeV and diameter {diameter} cm\n\n")
                    continue
                
                f.write("| Distance (cm) | Angle ( degrees) | Dose Rate (rem/hr) |\n")
                f.write("|---------------|----------|--------------------|\n")

                for distance in DETECTOR_DISTANCES:
                    for angle in DETECTOR_ANGLES:
                        key = (distance, angle)
                        key_str = str(key)

                        if key_str in results[energy_str][diameter_str]['detector_results']:
                            if 'dose' in results[energy_str][diameter_str]['detector_results'][key_str]:
                                dose = results[energy_str][diameter_str]['detector_results'][key_str]['dose']
                                f.write(f"| {distance} | {angle} | {dose:.4e} |\n")

                f.write("\n")

            f.write("\n")

def perform_error_analysis(results, energy, channel_diameter, filename):
    """
    Perform error analysis on simulation results.

    Args:
        results (dict): Results dictionary
        energy (float): Gamma-ray energy in MeV
        channel_diameter (float): Diameter of the air channel in cm
        filename (str): Output filename
    """
    logger.info(f"Performing error analysis for energy {energy} MeV and channel diameter {channel_diameter} cm...")

    # Extract detector results
    detector_results = results[str(energy)][str(channel_diameter)]['detector_results']

    # Calculate relative errors for each detector position
    errors = {}
    for key_str, data in detector_results.items():
        if 'dose_error' in data and data['dose'] > 0:
            rel_error = data['dose_error'] / data['dose']
            errors[key_str] = rel_error

    # Create figure
    plt.figure(figsize=(12, 8))

    # Group by distance
    for distance in DETECTOR_DISTANCES:
        angles = []
        rel_errors = []

        for angle in DETECTOR_ANGLES:
            key = (distance, angle)
            key_str = str(key)

            if key_str in errors:
                angles.append(angle)
                rel_errors.append(errors[key_str])

        if angles and rel_errors:
            plt.plot(angles, rel_errors, marker='o', label=f'Distance = {distance} cm')

    # Add labels and title
    plt.xlabel('Detector Angle [degrees]')
    plt.ylabel('Relative Error')
    plt.title(f'Relative Error Analysis (E={energy} MeV, Channel Ø={channel_diameter} cm)')

    # Add grid
    plt.grid(True, linestyle='--', alpha=0.5)

    # Add legend
    plt.legend()

    # Save figure
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    # Create error analysis report
    report_filename = filename.replace('.png', '.txt')
    with open(report_filename, 'w') as f:
        f.write(f"# Error Analysis Report (E={energy} MeV, Channel Ø={channel_diameter} cm)\n\n")

        f.write("## Statistical Errors\n\n")
        f.write("| Distance (cm) | Angle ( degrees) | Dose Rate (rem/hr) | Relative Error |\n")
        f.write("|---------------|----------|--------------------|-----------------|\n")

        for key_str, data in detector_results.items():
            if 'dose_error' in data and data['dose'] > 0:
                key = eval(key_str)
                distance, angle = key
                rel_error = data['dose_error'] / data['dose']
                f.write(f"| {distance} | {angle} | {data['dose']:.4e} | {rel_error:.4f} |\n")

        f.write("\n## Error Analysis Summary\n\n")

        # Calculate average and maximum relative errors
        if errors:
            avg_error = sum(errors.values()) / len(errors)
            max_error = max(errors.values())

            f.write(f"- Average Relative Error: {avg_error:.4f}\n")
            f.write(f"- Maximum Relative Error: {max_error:.4f}\n")

            # Assess simulation quality
            if max_error < 0.05:
                quality = "Excellent"
            elif max_error < 0.1:
                quality = "Good"
            elif max_error < 0.2:
                quality = "Acceptable"
            else:
                quality = "Poor - consider increasing particle count"

            f.write(f"- Simulation Quality: {quality}\n")

def run_simulation(energy, channel_diameter, particles=None):
    """
    Run simulation for a specific energy and channel diameter with improved error handling
    and checkpoint recovery to handle interruptions gracefully.

    Args:
        energy (float): Gamma-ray energy in MeV
        channel_diameter (float): Diameter of the air channel in cm
        particles (int, optional): Number of particles to simulate, overrides default calculation

    Returns:
        dict: Simulation results
    """
    logger.info(f"Starting simulation for energy {energy} MeV and channel diameter {channel_diameter} cm...")

    # Generate run ID
    run_id = f"E{energy}_D{channel_diameter}"
    status_file = CHECKPOINT_DIR / f'status_{run_id}.json'
    sim_lock_file = CHECKPOINT_DIR / f'lock_{run_id}.lock'
    
    # Use file locking to prevent multiple processes from running the same simulation
    try:
        if sim_lock_file.exists():
            # Check if lock file is stale (older than 2 hours)
            lock_age = time.time() - sim_lock_file.stat().st_mtime
            if lock_age < 7200:  # 2 hours in seconds
                logger.warning(f"Simulation for {run_id} appears to be in progress (lock file age: {lock_age:.2f}s). Skipping.")
                return None
            else:
                logger.warning(f"Found stale lock file for {run_id} (age: {lock_age:.2f}s). Removing and proceeding.")
                sim_lock_file.unlink()
        
        # Create lock file
        with open(sim_lock_file, 'w') as f:
            f.write(f"Started at: {time.time()}")
            
        # Update status file
        with open(status_file, 'w') as f:
            status = {
                'start_time': time.time(),
                'status': 'running',
                'energy': energy,
                'channel_diameter': channel_diameter,
                'pid': os.getpid()
            }
            json.dump(status, f, indent=2)
    except Exception as e:
        logger.warning(f"Error managing simulation locks: {e}")

    # First try to load checkpoint
    results = load_checkpoint(energy, channel_diameter)
    if results is not None:
        logger.info(f"Loaded checkpoint for energy {energy} MeV and channel diameter {channel_diameter} cm")
        # Update status file to indicate checkpoint was loaded
        try:
            with open(status_file, 'w') as f:
                status = {
                    'start_time': time.time(),
                    'completion_time': time.time(),
                    'status': 'completed_from_checkpoint',
                    'energy': energy,
                    'channel_diameter': channel_diameter
                }
                json.dump(status, f, indent=2)
            
            # Remove lock file
            if sim_lock_file.exists():
                sim_lock_file.unlink()
        except Exception as e:
            logger.warning(f"Failed to update status after loading checkpoint: {e}")
        
        return results
        
    # If no checkpoint, run the simulation
    try:
        start_time = time.time()
        
        # IMPORTANT: Create tallies first to get detector cells
        logger.info("Creating tallies first to get detector cells...")
        tallies, mesh_info, detector_cells = create_tallies(channel_diameter)
        
        # Then create geometry with detector cells
        logger.info("Creating geometry with detector cells...")
        geometry, surfaces = create_geometry(channel_diameter, detector_cells)
        
        # Create settings with the particles parameter if provided
        logger.info("Creating settings...")
        settings = create_settings(energy, channel_diameter, particles)
        
        # Create source
        logger.info("Creating source...")
        source = create_source(energy, channel_diameter)
        settings.source = source
        
        # Export settings to XML
        settings.export_to_xml()

        # Run OpenMC simulation
        logger.info(f"Running OpenMC for energy {energy} MeV and channel diameter {channel_diameter} cm...")
        try:
            openmc.run()
        except Exception as e:
            logger.error(f"OpenMC run failed: {e}")
            logger.error(traceback.format_exc())
            raise
        
        # Update lock file to indicate simulation has completed running
        try:
            with open(sim_lock_file, 'w') as f:
                f.write(f"Completed running at: {time.time()}")
        except Exception as e:
            logger.warning(f"Failed to update lock file: {e}")

        # Load statepoint file
        sp_files = list(Path('.').glob(f'statepoint.*_E{energy}_D{channel_diameter}.h5'))
        if not sp_files:
            raise FileNotFoundError("No statepoint file found")
        
        # Sort by modification time to get the most recent
        sp_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        sp_file = sp_files[0]
        
        logger.info(f"Processing results from {sp_file}")
        try:
            sp = openmc.StatePoint(sp_file)
        except Exception as e:
            logger.error(f"Could not load statepoint file: {e}")
            raise

        # Initialize results dictionary with improved structure
        results = {
            str(energy): {
                str(channel_diameter): {
                    'energy': energy,
                    'channel_diameter': channel_diameter,
                    'detector_results': {},
                    'mesh_data': None,
                    'fine_mesh_data': None,
                    'energy_bins': None,
                    'flux_spectrum': None,
                    'mesh_info': mesh_info,
                    'simulation_params': {
                        'num_particles': settings.particles,
                        'runtime': None,
                        'completion_time': time.time()
                    }
                }
            }
        }
        
        # Record simulation runtime
        runtime = time.time() - start_time
        results[str(energy)][str(channel_diameter)]['simulation_params']['runtime'] = runtime
        
        # Create intermediate checkpoint after each major processing step
        def save_intermediate_checkpoint():
            """Save checkpoint after each major processing step"""
            try:
                temp_checkpoint = CHECKPOINT_DIR / f'temp_checkpoint_E{energy}_D{channel_diameter}.pkl'
                with open(temp_checkpoint, 'wb') as f:
                    pickle.dump(results, f)
                logger.info(f"Saved intermediate checkpoint for E={energy}, D={channel_diameter}")
            except Exception as e:
                logger.warning(f"Failed to save intermediate checkpoint: {e}")

        # Extract mesh tally results
        try:
            # In OpenMC 0.14.0, we retrieve tallies by name
            mesh_tally = sp.get_tally(name='FullMeshTally')
            mesh_values = mesh_tally.get_values(scores=['flux'])
            results[str(energy)][str(channel_diameter)]['mesh_data'] = mesh_values.flatten()
            
            fine_mesh_tally = sp.get_tally(name='FineMeshTally')
            fine_mesh_values = fine_mesh_tally.get_values(scores=['flux'])
            results[str(energy)][str(channel_diameter)]['fine_mesh_data'] = fine_mesh_values.flatten()
            
            # Save intermediate checkpoint after mesh data extraction
            save_intermediate_checkpoint()
            
            # Create intermediate visualizations right after getting mesh data
            if not PLOTS_DIR.exists():
                PLOTS_DIR.mkdir(parents=True, exist_ok=True)
                
            # Save intermediate mesh visualization
            intermediate_filename = PLOTS_DIR / f'radiation_map_E{energy}_D{channel_diameter}_intermediate.png'
            try:
                visualize_radiation_map(
                    results[str(energy)][str(channel_diameter)]['mesh_data'],
                    mesh_info['full_mesh']['dimensions'],
                    mesh_info['full_mesh']['bounds'],
                    energy,
                    channel_diameter,
                    intermediate_filename
                )
                logger.info(f"Created intermediate radiation map: {intermediate_filename}")
            except Exception as e:
                logger.error(f"Failed to create intermediate visualization: {e}")
                
        except Exception as e:
            logger.error(f"Error extracting mesh data: {e}")
            # Continue even if mesh data extraction fails

        # Extract detector tally results
        detector_count = 0
        for distance in DETECTOR_DISTANCES:
            for angle in DETECTOR_ANGLES:
                tally_name = f'DetectorTally_D{distance}_A{angle}'
                try:
                    detector_tally = sp.get_tally(name=tally_name)

                    # Get flux values - handle multi-dimensional arrays properly
                    flux_values = detector_tally.get_values(scores=['flux'])
                    flux = flux_values.flatten()
                    flux_error = detector_tally.std_dev.flatten()

                    # Convert flux to dose
                    dose = 0.0
                    dose_error = 0.0

                    # Check if we have energy bins from energy filter
                    energy_filters = [f for f in detector_tally.filters if isinstance(f, openmc.EnergyFilter)]
                    if energy_filters and energy_filters[0].bins.size > 1:
                        energy_filter = energy_filters[0]
                        energy_bins = energy_filter.bins
                        
                        # Store energy bins if not already set
                        if results[str(energy)][str(channel_diameter)]['energy_bins'] is None:
                            results[str(energy)][str(channel_diameter)]['energy_bins'] = energy_bins
                        
                        # Integrate over energy bins - handle multi-dimensional arrays properly
                        for i in range(len(energy_bins) - 1):
                            e_avg = (energy_bins[i] + energy_bins[i+1]) / 2
                            dose_factor = interpolate_flux_to_dose(e_avg)
                            if i < flux.size:  # Ensure index is within bounds
                                dose += flux[i] * dose_factor
                                dose_error += (flux_error[i] * dose_factor)**2

                        dose_error = np.sqrt(dose_error)
                    else:
                        # Simplified calculation if we don't have energy bins
                        dose_factor = interpolate_flux_to_dose(energy)
                        dose = np.sum(flux) * dose_factor
                        dose_error = np.sqrt(np.sum(flux_error**2)) * dose_factor

                    # Store results
                    key = (distance, angle)
                    results[str(energy)][str(channel_diameter)]['detector_results'][str(key)] = {
                        'flux': float(np.sum(flux)),
                        'flux_error': float(np.sqrt(np.sum(flux_error**2))),
                        'dose': float(dose),
                        'dose_error': float(dose_error),
                        'position': {
                            'x': CONCRETE_THICKNESS + distance * math.cos(math.radians(angle)),
                            'y': distance * math.sin(math.radians(angle)),
                            'z': 0.0
                        }
                    }
                    detector_count += 1
                    
                    # Save intermediate checkpoint every 10 detectors to avoid data loss
                    if detector_count % 10 == 0:
                        save_intermediate_checkpoint()

                except Exception as e:
                    logger.warning(f"Could not extract tally {tally_name}: {e}")
                    logger.warning(f"Traceback: {traceback.format_exc()}")
                    
        # Save intermediate dose visualization
        intermediate_dose = PLOTS_DIR / f'dose_heatmap_E{energy}_D{channel_diameter}_intermediate.png'
        try:
            visualize_dose_heatmap(results, energy, channel_diameter, intermediate_dose)
            logger.info(f"Created intermediate dose heatmap: {intermediate_dose}")
        except Exception as e:
            logger.error(f"Failed to create intermediate dose heatmap: {e}")
                
        # Update status file to indicate completion
        try:
            with open(status_file, 'w') as f:
                status = {
                    'start_time': start_time,
                    'completion_time': time.time(),
                    'status': 'completed',
                    'energy': energy,
                    'channel_diameter': channel_diameter,
                    'runtime': runtime
                }
                json.dump(status, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to update status file: {e}")

        # Save checkpoint
        save_checkpoint(results, energy, channel_diameter)

        # Clean up statepoint files to save disk space
        # Keep only the most recent one
        for sp_file_path in sp_files[1:]:
            try:
                sp_file_path.unlink()
                logger.info(f"Removed old statepoint file: {sp_file_path}")
            except Exception as e:
                logger.warning(f"Could not remove statepoint file {sp_file_path}: {e}")
                
        # Remove lock file
        if sim_lock_file.exists():
            try:
                sim_lock_file.unlink()
            except Exception as e:
                logger.warning(f"Could not remove lock file: {e}")
                
        logger.info(f"Simulation completed for energy {energy} MeV and channel diameter {channel_diameter} cm")
        return results

    except Exception as e:
        logger.error(f"Simulation failed for energy {energy} MeV and channel diameter {channel_diameter} cm: {e}")
        logger.error(traceback.format_exc())
        
        # Update status file to indicate failure
        try:
            with open(status_file, 'w') as f:
                status = {
                    'start_time': time.time(),
                    'completion_time': time.time(),
                    'status': 'failed',
                    'energy': energy,
                    'channel_diameter': channel_diameter,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
                json.dump(status, f, indent=2)
        except Exception as e2:
            logger.warning(f"Failed to update status file after error: {e2}")
            
        # Remove lock file
        if sim_lock_file.exists():
            try:
                sim_lock_file.unlink()
            except Exception as e:
                logger.warning(f"Could not remove lock file after error: {e}")
        
        # Return None to indicate failure
        return None

def process_results(sp_filename, energy, channel_diameter, mesh_info):
    """
    Process simulation results.

    Args:
        sp_filename (str): Statepoint filename
        energy (float): Gamma-ray energy in MeV
        channel_diameter (float): Diameter of the air channel in cm
        mesh_info (dict): Mesh information

    Returns:
        dict: Processed results
    """
    # Initialize results dictionary
    results = {
        'detector_results': {},
        'mesh_data': None,
        'mesh_info': mesh_info,
        'fine_mesh_data': None,
        'energy_bins': None,
        'flux_spectrum': None
    }

    try:
        # Open statepoint file
        sp = openmc.StatePoint(sp_filename)

        # Process detector tallies
        for tally in sp.tallies.values():
            if tally.name.startswith('detector_'):
                # Extract detector position from tally name
                parts = tally.name.split('_')
                if len(parts) >= 3:
                    distance = int(parts[1])
                    angle = int(parts[2])
                    key = (distance, angle)
                    key_str = str(key)

                    # Get tally results
                    mean = tally.mean.flatten()[0]
                    std_dev = tally.std_dev.flatten()[0]
                    rel_error = std_dev / mean if mean > 0 else 0.0

                    # Convert to dose rate (rem/hr)
                    # Assuming a simple conversion factor for demonstration
                    dose_conversion_factor = 3.6e-6  # (rem/hr)/(particle/cm²)
                    dose = mean * dose_conversion_factor

                    # Store results
                    results['detector_results'][key_str] = {
                        'flux': float(mean),
                        'std_dev': float(std_dev),
                        'rel_error': float(rel_error),
                        'dose': float(dose)
                    }

        # Process mesh tallies
        if 'mesh' in sp.tallies:
            mesh_tally = sp.tallies['mesh']
            mesh_data = mesh_tally.get_pandas_dataframe()
            results['mesh_data'] = mesh_data.to_dict()

        # Process fine mesh tallies
        if 'fine_mesh' in sp.tallies:
            fine_mesh_tally = sp.tallies['fine_mesh']
            fine_mesh_data = fine_mesh_tally.get_pandas_dataframe()
            results['fine_mesh_data'] = fine_mesh_data.to_dict()

        # Process energy spectrum
        if 'energy_spectrum' in sp.tallies:
            energy_tally = sp.tallies['energy_spectrum']
            energy_data = energy_tally.get_pandas_dataframe()

            # Extract energy bins and flux spectrum
            energy_bins = energy_data.index.get_level_values('energy low').unique().tolist()
            energy_bins.append(energy_data.index.get_level_values('energy high').max())

            flux_spectrum = energy_data['mean'].tolist()

            results['energy_bins'] = energy_bins
            results['flux_spectrum'] = flux_spectrum

        return results

    except Exception as e:
        logger.error(f"Error processing results: {e}")
        return results

def run_simulation_wrapper(params):
    """
    Wrapper for run_simulation to be used with multiprocessing.map.
    
    Args:
        params (tuple): (energy, channel_diameter) or (energy, channel_diameter, particles)
    
    Returns:
        dict: Simulation results
    """
    if len(params) == 3:
        energy, channel_diameter, particles = params
        return run_simulation(energy, channel_diameter, particles)
    else:
        energy, channel_diameter = params
        return run_simulation(energy, channel_diameter)
def merge_results(results_list):
    """
    Merge results from multiple simulations.
    
    Args:
        results_list (list): List of results dictionaries
    
    Returns:
        dict: Merged results
    """
    merged = {}
    for results in results_list:
        for energy_str, energy_data in results.items():
            if energy_str not in merged:
                merged[energy_str] = {}
            for diameter_str, diameter_data in energy_data.items():
                merged[energy_str][diameter_str] = diameter_data
    return merged

def print_progress(completed, total, run_ids=None):
    """
    Print progress of simulations.
    
    Args:
        completed (int): Number of completed simulations
        total (int): Total number of simulations
        run_ids (list, optional): List of completed run IDs
    """
    percent = completed * 100 / total
    bar_length = 40
    filled_length = int(bar_length * completed / total)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    
    print(f'\r[{bar}] {percent:.1f}% ({completed}/{total}) Complete', end='')
    
    if run_ids and len(run_ids) > 0:
        last_runs = ', '.join(run_ids[-min(3, len(run_ids)):])
        print(f' | Recent: {last_runs}', end='')
        
    if completed == total:
        print()  # Add newline at the end

def main():
    """
    Main entry point for the simulation.
    """
    # Make args available globally
    global args
    
    print("Starting simulation main function")
    sys.stdout.flush()
    
    parser = argparse.ArgumentParser(description="OpenMC simulation for gamma-ray shielding study")
    parser.add_argument("--parallel", type=int, default=mp.cpu_count(), 
                      help=f'Number of parallel processes (default: {mp.cpu_count()} - all CPUs)')
    parser.add_argument("--energy", type=float, help='Specific energy to simulate (MeV)')
    parser.add_argument("--diameter", type=float, help='Specific channel diameter to simulate (cm)')
    parser.add_argument("--particles", type=int, help='Number of particles to simulate (overrides default calculation)')
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze existing results without running simulations")
    parser.add_argument("--force-rerun", action="store_true", help="Force rerun of simulations even if checkpoints exist")
    parser.add_argument("--skip-failed", action="store_true", help="Skip failed simulations instead of retrying")
    parser.add_argument("--detailed-plots", action="store_true", help="Generate detailed scientific plots")
    parser.add_argument("--comprehensive-heatmaps", action="store_true", help="Generate comprehensive heatmaps that show both top and bottom hemispheres")
    parser.add_argument("--json-file", type=str, default=str(DATA_DIR / 'simulation_results.json'), 
                      help='Path to JSON file for saving/loading results')
    parser.add_argument("--no-visualizations", action="store_true", help="Skip creating visualizations (useful for headless environments)")
    parser.add_argument("--report-only", action="store_true", help="Only generate report from existing results")
    args = parser.parse_args()

    # Adjust number of parallel processes to be reasonable
    if args.parallel <= 0:
        args.parallel = 1
    elif args.parallel > mp.cpu_count():
        logger.warning(f"Requested {args.parallel} processes, but system only has {mp.cpu_count()} CPUs")
        args.parallel = mp.cpu_count()
    
    logger.info(f"Using {args.parallel} parallel processes")

    # Create directories if they don't exist
    for directory in [RESULTS_DIR, CHECKPOINT_DIR, PLOTS_DIR, DATA_DIR]:
        try:
            directory.mkdir(exist_ok=True, parents=True)
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {e}")
            # Continue execution even if directory creation fails

    # Determine which parameter combinations to run
    if args.energy is not None and args.diameter is not None:
        # Run a single specific combination
        if args.particles is not None:
            param_combinations = [(args.energy, args.diameter, args.particles)]
        else:
            param_combinations = [(args.energy, args.diameter)]
    elif args.energy is not None:
        # Run all diameters for a specific energy
        if args.particles is not None:
            param_combinations = [(args.energy, d, args.particles) for d in CHANNEL_DIAMETERS]
        else:
            param_combinations = [(args.energy, d) for d in CHANNEL_DIAMETERS]
    elif args.diameter is not None:
        # Run all energies for a specific diameter
        if args.particles is not None:
            param_combinations = [(e, args.diameter, args.particles) for e in GAMMA_ENERGIES]
        else:
            param_combinations = [(e, args.diameter) for e in GAMMA_ENERGIES]
    else:
        # Run all combinations
        if args.particles is not None:
            param_combinations = [(e, d, args.particles) for e in GAMMA_ENERGIES for d in CHANNEL_DIAMETERS]
        else:
            param_combinations = [(e, d) for e in GAMMA_ENERGIES for d in CHANNEL_DIAMETERS]
            
    # Sort combinations by estimated computation time (smallest channel first)
    # This helps distribute work more evenly
    param_combinations.sort(key=lambda x: (x[0], 1/x[1]))

    all_results = {}
    failed_runs = []
    completed_runs = []

    if not args.analyze_only and not args.report_only:
        logger.info(f"Planning to run {len(param_combinations)} simulation(s)")
        print(f"Simulations to run: {len(param_combinations)}")
        for i, params in enumerate(param_combinations):
            if len(params) == 3:
                e, d, p = params
                print(f"  {i+1}. Energy: {e} MeV, Diameter: {d} cm, Particles: {p}")
            else:
                e, d = params
                print(f"  {i+1}. Energy: {e} MeV, Diameter: {d} cm")
        print()
        
        # Check if some simulations can be loaded from checkpoints
        skip_combinations = []
        if not args.force_rerun:
            for params in param_combinations:
                if len(params) == 3:
                    energy, diameter, _ = params  # Ignore particles for checkpoint
                else:
                    energy, diameter = params
                    
                results = load_checkpoint(energy, diameter)
                if results is not None:
                    logger.info(f"Found checkpoint for E={energy}, D={diameter}, skipping simulation")
                    all_results.update(results)
                    skip_combinations.append(params)  # Skip the full params tuple
                    completed_runs.append(f"E{energy}_D{diameter}")
                    
        # Remove combinations that can be loaded from checkpoints
        for combo in skip_combinations:
            param_combinations.remove(combo)
            
        if skip_combinations:
            logger.info(f"Skipping {len(skip_combinations)} simulations that have existing checkpoints")
            print(f"Skipping {len(skip_combinations)} simulations with existing checkpoints.")
            print(f"Running {len(param_combinations)} simulations.")
        
        if not param_combinations:
            logger.info("All simulations already have checkpoints, skipping to analysis")
            print("All simulations already have checkpoints. Proceeding to analysis.")
        else:
            # Run simulations in parallel
            print("Starting simulations...")
            if args.parallel > 1:
                # Create a multiprocessing pool with shared progress counter
                with mp.Pool(processes=args.parallel) as pool:
                    # Use a simple map for more predictable behavior
                    results_list = []
                    completed = 0
                    
                    for result in pool.imap_unordered(run_simulation_wrapper, param_combinations):
                        completed += 1
                        if result is not None:
                            energy, diameter = None, None
                            for energy_str in result:
                                for diameter_str in result[energy_str]:
                                    energy = float(energy_str)
                                    diameter = float(diameter_str)
                                    break
                                break
                            
                            if energy is not None and diameter is not None:
                                run_id = f"E{energy}_D{diameter}"
                                completed_runs.append(run_id)
                                results_list.append(result)
                            else:
                                failed_runs.append("Unknown")
                        else:
                            # Record the parameters of the failed run
                            failed_params = param_combinations[len(results_list) + len(failed_runs)]
                            if len(failed_params) == 3:
                                energy, diameter, _ = failed_params
                            else:
                                energy, diameter = failed_params
                            failed_runs.append(f"E{energy}_D{diameter}")
                        
                        # Update progress
                        print_progress(completed, len(param_combinations), completed_runs)
            else:
                # Run sequentially with progress updates
                results_list = []
                completed = 0
                
                for params in param_combinations:
                    if len(params) == 3:
                        energy, diameter, _ = params  # We don't need to use particles here as it's passed in the wrapper
                    else:
                        energy, diameter = params
                    run_id = f"E{energy}_D{diameter}"
                    print(f"\nRunning simulation {completed+1}/{len(param_combinations)}: {run_id}")
                    
                    result = run_simulation_wrapper(params)
                    completed += 1
                    
                    if result is not None:
                        results_list.append(result)
                        completed_runs.append(run_id)
                    else:
                        failed_runs.append(run_id)
                    
                    # Update progress
                    print_progress(completed, len(param_combinations), completed_runs)
            
            # Process failed runs
            if failed_runs and not args.skip_failed:
                logger.warning(f"{len(failed_runs)} simulations failed. Retrying serially.")
                print(f"\n{len(failed_runs)} simulations failed. Retrying one by one:")
                
                for run_id in failed_runs[:]:
                    parts = run_id.replace('E', '').replace('D', '').split('_')
                    if len(parts) == 2:
                        energy = float(parts[0])
                        diameter = float(parts[1])
                        
                        print(f"Retrying {run_id}...")
                        # Pass the particles parameter if it's available
                        if args.particles is not None:
                            result = run_simulation(energy, diameter, args.particles)
                        else:
                            result = run_simulation(energy, diameter)
                        
                        if result is not None:
                            results_list.append(result)
                            failed_runs.remove(run_id)
                            completed_runs.append(run_id)
                            print(f"Retry successful for {run_id}")
                        else:
                            print(f"Retry failed for {run_id}")
            
            # Report final status
            if failed_runs:
                logger.warning(f"{len(failed_runs)} simulations failed permanently: {', '.join(failed_runs)}")
                print(f"\n{len(failed_runs)} simulations failed permanently:")
                for run_id in failed_runs:
                    print(f"  - {run_id}")
            
            # Merge results
            all_results.update(merge_results(results_list))
            
            # Save complete results to JSON with improved handling
            json_file = Path(args.json_file)
            if save_results_to_json(all_results, json_file):
                print(f"Results successfully saved to {json_file}")
            else:
                print(f"Error saving results to {json_file}")
    else:
        # Load existing results for analysis or reporting
        if args.analyze_only:
            msg = "Analyze-only mode: Loading existing results"
        else:
            msg = "Report-only mode: Loading existing results"
            
        logger.info(msg)
        print(msg)
        
        try:
            json_file = Path(args.json_file)
            if not json_file.exists():
                logger.error(f"No existing results found at {json_file}. Run simulations first or specify a different JSON file.")
                print(f"ERROR: No existing results found at {json_file}. Run simulations first or specify a different JSON file.")
                return
                
            print(f"Loading results from {json_file}...")
            with open(json_file, 'r') as f:
                json_results = json.load(f)

                # Convert lists back to numpy arrays
                for energy_str, energy_data in json_results.items():
                    if energy_str == 'metadata':
                        continue
                        
                    if energy_str not in all_results:
                        all_results[energy_str] = {}

                    for diameter_str, diameter_data in energy_data.items():
                        all_results[energy_str][diameter_str] = {}

                        for key, value in diameter_data.items():
                            if isinstance(value, dict):
                                all_results[energy_str][diameter_str][key] = {}
                                for subkey, subvalue in value.items():
                                    if isinstance(subvalue, dict):
                                        all_results[energy_str][diameter_str][key][subkey] = {}
                                        for subsubkey, subsubvalue in subvalue.items():
                                            if isinstance(subsubvalue, list):
                                                all_results[energy_str][diameter_str][key][subkey][subsubkey] = np.array(subsubvalue)
                                            else:
                                                all_results[energy_str][diameter_str][key][subkey][subsubkey] = subsubvalue
                                    elif isinstance(subvalue, list):
                                        all_results[energy_str][diameter_str][key][subkey] = np.array(subvalue)
                                    else:
                                        all_results[energy_str][diameter_str][key][subkey] = subvalue
                            elif isinstance(value, list):
                                all_results[energy_str][diameter_str][key] = np.array(value)
                            else:
                                all_results[energy_str][diameter_str][key] = value

            logger.info("Successfully loaded existing results for analysis")
            print("Successfully loaded existing results. Proceeding with analysis.")
        except Exception as e:
            logger.error(f"Error loading results: {e}")
            logger.error(traceback.format_exc())
            print(f"ERROR: Could not load results: {e}")
            return

    # Skip visualizations if requested
    if args.no_visualizations:
        logger.info("Skipping visualizations as requested")
        print("\nSkipping visualizations as requested.")
        
        # Still generate report
        try:
            # Analyze critical parameters
            critical_analysis = analyze_critical_parameters(all_results)
            
            # Create summary report
            summary_file = RESULTS_DIR / 'summary_report.md'
            create_summary_report(all_results, critical_analysis, summary_file)
            print(f"\nSummary report generated: {summary_file}")
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            print(f"Error generating report: {e}")
            
        return

    # Generate visualizations
    print("\nGenerating visualizations and analysis...")
    
    # Create a progress counter for visualizations
    viz_tasks = len(GAMMA_ENERGIES) * 3 + len(CHANNEL_DIAMETERS) * len(GAMMA_ENERGIES) + 3
    
    # Add tasks for enhanced visualizations if requested
    if args.comprehensive_heatmaps:
        viz_tasks += len(GAMMA_ENERGIES) * 2  # Enhanced dose vs angle study, one per energy
        viz_tasks += len(GAMMA_ENERGIES) * len(CHANNEL_DIAMETERS)  # Full config visualization, one per combination
        viz_tasks += len(GAMMA_ENERGIES) * len(CHANNEL_DIAMETERS)  # Flux vs distance, one per combination
    
    # Add tasks for detailed plots if requested
    if args.detailed_plots:
        viz_tasks += len(GAMMA_ENERGIES) * len(CHANNEL_DIAMETERS)  # Add tasks for detailed plots
    
    viz_completed = 0

    # Create dose vs angle visualizations for each energy
    for energy in GAMMA_ENERGIES:
        energy_str = str(energy)
        if energy_str in all_results:
            viz_completed += 1
            print_progress(viz_completed, viz_tasks)
            
            try:
                visualize_dose_vs_angle(
                    all_results,
                    energy,
                    PLOTS_DIR / f'dose_vs_angle_E{energy}.png'
                )
            except Exception as e:
                logger.error(f"Error creating dose vs angle visualization for E={energy}: {e}")
            
            # Enhanced angle study if requested
            if args.comprehensive_heatmaps:
                try:
                    visualize_enhanced_dose_vs_angle_study(
                        all_results,
                        energy,
                        PLOTS_DIR / f'enhanced_dose_vs_angle_E{energy}.png'
                    )
                except Exception as e:
                    logger.error(f"Error creating enhanced dose angle study for E={energy}: {e}")
            
            # Generate standard radiation maps for each diameter
            for diameter in CHANNEL_DIAMETERS:
                diameter_str = str(diameter)
                if diameter_str in all_results[energy_str]:
                    viz_completed += 1
                    print_progress(viz_completed, viz_tasks)
                    
                    # Check if mesh data is available
                    if ('mesh_data' in all_results[energy_str][diameter_str] and
                        'mesh_info' in all_results[energy_str][diameter_str]):
                        try:
                            # Main radiation map
                            mesh_data = all_results[energy_str][diameter_str]['mesh_data']
                            mesh_info = all_results[energy_str][diameter_str]['mesh_info']
                            
                            visualize_radiation_map(
                                mesh_data,
                                mesh_info['full_mesh']['dimensions'],
                                mesh_info['full_mesh']['bounds'],
                                energy,
                                diameter,
                                PLOTS_DIR / f'radiation_map_E{energy}_D{diameter}.png'
                            )
                            
                            # Enhanced close-up map
                            if 'fine_mesh_data' in all_results[energy_str][diameter_str]:
                                fine_mesh_data = all_results[energy_str][diameter_str]['fine_mesh_data']
                                
                                visualize_enhanced_radiation_map(
                                    fine_mesh_data,
                                    mesh_info['fine_mesh']['dimensions'],
                                    mesh_info['fine_mesh']['bounds'],
                                    diameter,
                                    energy,
                                    PLOTS_DIR / f'enhanced_radiation_map_E{energy}_D{diameter}.png'
                                )
                                
                                # Full configuration visualization if requested
                                if args.comprehensive_heatmaps:
                                    try:
                                        viz_completed += 1
                                        print_progress(viz_completed, viz_tasks)
                                        visualize_full_configuration(
                                            mesh_data,
                                            mesh_info['full_mesh']['dimensions'],
                                            mesh_info['full_mesh']['bounds'],
                                            energy,
                                            diameter,
                                            PLOTS_DIR / f'full_configuration_E{energy}_D{diameter}.png'
                                        )
                                        logger.info(f"Created full configuration visualization for E={energy}, D={diameter}")
                                    except Exception as e:
                                        logger.error(f"Error creating full configuration visualization for E={energy}, D={diameter}: {e}")
                        except Exception as e:
                            logger.error(f"Error creating radiation maps for E={energy}, D={diameter}: {e}")
                    
                    # Flux spectra visualization
                    try:
                        if ('energy_bins' in all_results[energy_str][diameter_str] and
                            'flux_spectrum' in all_results[energy_str][diameter_str]):
                            visualize_flux_spectra(
                                all_results,
                                energy,
                                diameter,
                                PLOTS_DIR / f'flux_spectrum_E{energy}_D{diameter}.png'
                            )
                            
                            # Flux spectra vs distance if requested
                            if args.comprehensive_heatmaps:
                                try:
                                    viz_completed += 1
                                    print_progress(viz_completed, viz_tasks)
                                    visualize_flux_spectra_vs_distance(
                                        all_results,
                                        energy,
                                        diameter,
                                        PLOTS_DIR / f'flux_spectra_vs_distance_E{energy}_D{diameter}.png'
                                    )
                                    logger.info(f"Created flux spectra vs distance visualization for E={energy}, D={diameter}")
                                except Exception as e:
                                    logger.error(f"Error creating flux spectra vs distance for E={energy}, D={diameter}: {e}")
                    except Exception as e:
                        logger.error(f"Error creating flux spectrum for E={energy}, D={diameter}: {e}")
                    
                    # Dose heatmap visualization
                    try:
                        if 'detector_results' in all_results[energy_str][diameter_str]:
                            visualize_dose_heatmap(
                                all_results,
                                energy,
                                diameter,
                                PLOTS_DIR / f'dose_heatmap_E{energy}_D{diameter}.png'
                            )
                    except Exception as e:
                        logger.error(f"Error creating dose heatmap for E={energy}, D={diameter}: {e}")

    # Analyze critical parameters
    try:
        viz_completed += 1
        print_progress(viz_completed, viz_tasks)
        critical_analysis = analyze_critical_parameters(all_results)
    except Exception as e:
        logger.error(f"Error analyzing critical parameters: {e}")
        critical_analysis = None

    # Visualize critical parameters if analysis was successful
    if critical_analysis is not None:
        try:
            viz_completed += 1
            print_progress(viz_completed, viz_tasks)
            visualize_critical_parameters(
                critical_analysis,
                PLOTS_DIR / 'critical_energy_analysis.png',
                PLOTS_DIR / 'critical_diameter_analysis.png'
            )
        except Exception as e:
            logger.error(f"Error creating critical parameter visualizations: {e}")

    # Create summary report
    try:
        viz_completed += 1
        print_progress(viz_completed, viz_tasks)
        create_summary_report(
            all_results,
            critical_analysis if critical_analysis else {},
            RESULTS_DIR / 'summary_report.md'
        )
    except Exception as e:
        logger.error(f"Error creating summary report: {e}")
    
    # Generate detailed scientific plots if requested
    if args.detailed_plots:
        try:
            print("\nGenerating detailed scientific visualizations...")
            visualize_all_detailed_radiation_maps(all_results)
            viz_completed += len(GAMMA_ENERGIES) * len(CHANNEL_DIAMETERS)
            print_progress(viz_completed, viz_tasks)
        except Exception as e:
            logger.error(f"Error creating detailed scientific visualizations: {e}")
    else:
        # Always create detailed plot for 0.1 MeV and 0.5 cm channel diameter
        try:
            energy = 0.1
            diameter = 0.5
            energy_str = str(energy)
            diameter_str = str(diameter)
            
            if (energy_str in all_results and 
                diameter_str in all_results[energy_str] and
                'fine_mesh_data' in all_results[energy_str][diameter_str] and
                'mesh_info' in all_results[energy_str][diameter_str]):
                
                print("\nGenerating detailed scientific visualization for E=0.1 MeV, D=0.5 cm...")
                fine_mesh_data = all_results[energy_str][diameter_str]['fine_mesh_data']
                mesh_info = all_results[energy_str][diameter_str]['mesh_info']
                
                output_file = PLOTS_DIR / f'detailed_radiation_map_E{energy}_D{diameter}.png'
                visualize_detailed_radiation_map(
                    fine_mesh_data,
                    mesh_info['fine_mesh']['dimensions'],
                    mesh_info['fine_mesh']['bounds'],
                    energy,
                    diameter,
                    output_file
                )
                print(f"Created detailed scientific visualization: {output_file}")
        except Exception as e:
            logger.error(f"Error creating detailed scientific visualization for E=0.1, D=0.5: {e}")

    print("\nAnalysis completed successfully!")
    logger.info("Analysis completed successfully")

    # List all generated files
    print("\nGenerated files:")
    print(f"  Results data: {args.json_file}")
    print(f"  Summary report: {RESULTS_DIR / 'summary_report.md'}")
    print(f"  Visualizations: {PLOTS_DIR}")
    
    # Show most important files
    if critical_analysis is not None and 'critical_energy' in critical_analysis:
        critical_energy = critical_analysis['critical_energy']
        critical_diameter = critical_analysis['critical_diameter']
        print("\nKey visualization files for critical parameters:")
        print(f"  Critical energy analysis: {PLOTS_DIR / 'critical_energy_analysis.png'}")
        print(f"  Critical diameter analysis: {PLOTS_DIR / 'critical_diameter_analysis.png'}")
        print(f"  Enhanced radiation map for critical case: "
             f"{PLOTS_DIR / f'enhanced_radiation_map_E{critical_energy}_D{critical_diameter}.png'}")
        print(f"  Dose angle study for critical energy: "
             f"{PLOTS_DIR / f'dose_vs_angle_E{critical_energy}.png'}")
        
        # Add detailed plot to important files list if it exists
        detailed_file = PLOTS_DIR / f'detailed_radiation_map_E0.1_D0.5.png'
        if detailed_file.exists():
            print(f"  Detailed scientific visualization: {detailed_file}")
    
    # If detailed plots were generated, show them
    if args.detailed_plots:
        detailed_plots = list(PLOTS_DIR.glob('detailed_radiation_map_*.png'))
        if detailed_plots:
            print("\nDetailed scientific visualizations:")
            for plot in detailed_plots[:5]:  # Show max 5 examples
                print(f"  {plot}")
            if len(detailed_plots) > 5:
                print(f"  ... and {len(detailed_plots) - 5} more detailed visualizations")

    # If enhanced visualizations were generated, show them
    if args.comprehensive_heatmaps:
        enhanced_plots = list(PLOTS_DIR.glob('enhanced_dose_vs_angle_*.png')) + \
                       list(PLOTS_DIR.glob('full_configuration_*.png')) + \
                       list(PLOTS_DIR.glob('flux_spectra_vs_distance_*.png'))
        if enhanced_plots:
            print("\nEnhanced visualizations:")
            for plot in enhanced_plots[:5]:  # Show max 5 examples
                print(f"  {plot}")
            if len(enhanced_plots) > 5:
                print(f"  ... and {len(enhanced_plots) - 5} more enhanced visualizations")

def visualize_detailed_radiation_map(mesh_data, mesh_dimensions, mesh_bounds, energy, channel_diameter, filename):
    """
    Create a detailed scientific visualization of radiation distribution outside the wall
    with parameter information, distance markers, and enhanced color mapping.
    
    Args:
        mesh_data (numpy.ndarray): Fine mesh tally data
        mesh_dimensions (list): Mesh dimensions [nx, ny, nz]
        mesh_bounds (list): Mesh bounds [xmin, xmax, ymin, ymax]
        energy (float): Gamma-ray energy in MeV
        channel_diameter (float): Diameter of the air channel in cm
        filename (str): Output filename
    """
    logger.info(f"Creating detailed scientific radiation map for energy {energy} MeV and channel diameter {channel_diameter} cm...")

    # Reshape mesh data
    nx, ny, nz = mesh_dimensions
    xmin, xmax, ymin, ymax = mesh_bounds

    # Extract central slice (z=0)
    central_slice = mesh_data.reshape(nx, ny, nz)[:, :, nz//2]

    # Create coordinate meshes
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Apply adaptive smoothing based on channel diameter
    sigma = max(0.3, channel_diameter / 4)  # Finer smoothing for detailed view
    logger.info(f"Using fine adaptive smoothing sigma={sigma} for channel diameter {channel_diameter} cm")
    smoothed_data = gaussian_filter(central_slice, sigma=sigma)

    # Apply log scale for better visualization (use natural log instead of log10 for scientific visualization)
    with np.errstate(divide='ignore', invalid='ignore'):
        flux_data = smoothed_data.copy()
        # Set a minimum threshold to avoid negative infinity
        min_threshold = np.max(flux_data) * 1e-12
        flux_data[flux_data < min_threshold] = min_threshold

    # Create a figure with the right aspect ratio
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Focus on the area outside the wall - limit the x range
    x_min_plot = CONCRETE_THICKNESS - 10
    x_max_plot = CONCRETE_THICKNESS + 150
    y_min_plot = -50
    y_max_plot = 50
    
    # Get the indices corresponding to the plot range
    x_indices = np.where((X[:, 0] >= x_min_plot) & (X[:, 0] <= x_max_plot))[0]
    y_indices = np.where((Y[0, :] >= y_min_plot) & (Y[0, :] <= y_max_plot))[0]
    
    # Extract the data for the plot range
    X_plot = X[np.ix_(x_indices, y_indices)]
    Y_plot = Y[np.ix_(x_indices, y_indices)]
    flux_plot = flux_data[np.ix_(x_indices, y_indices)]
    
    # Define a scientific color scale that transitions from red to yellow to green to blue
    # Similar to the image provided
    colors_scientific = [
        (1.0, 0.0, 0.0),          # Red
        (1.0, 0.4, 0.0),          # Orange-Red
        (1.0, 0.7, 0.0),          # Orange
        (1.0, 1.0, 0.0),          # Yellow
        (0.7, 1.0, 0.0),          # Yellow-Green
        (0.0, 1.0, 0.0),          # Green
        (0.0, 0.8, 0.8),          # Teal
        (0.0, 0.4, 1.0),          # Light Blue
        (0.0, 0.0, 1.0),          # Blue
        (0.0, 0.0, 0.5),          # Dark Blue
        (0.0, 0.0, 0.2)           # Very Dark Blue
    ]
    
    cmap_scientific = colors.LinearSegmentedColormap.from_list('scientific', colors_scientific)
    
    # Create a logarithmic normalization
    min_val = np.min(flux_plot)
    max_val = np.max(flux_plot)
    
    # Use a logarithmic norm for better visualization
    norm = colors.LogNorm(vmin=min_val, vmax=max_val)
    
    # Plot the heatmap
    im = ax.pcolormesh(X_plot, Y_plot, flux_plot, cmap=cmap_scientific, norm=norm, shading='auto')
    
    # Add shield wall line
    ax.axvline(x=CONCRETE_THICKNESS, color='black', linestyle='-', linewidth=3, label='Wall Exit')
    
    # Add channel position
    channel_radius = channel_diameter / 2.0
    rect = plt.Rectangle((CONCRETE_THICKNESS-0.5, -channel_radius), 1, 2*channel_radius, 
                       facecolor='gray', edgecolor='black', alpha=0.7, label='Channel Exit')
    ax.add_patch(rect)
    
    # Add distance markers as curved dashed lines
    distances = [25, 50, 75, 100]
    for d in distances:
        # Create a circle centered at the channel exit
        theta = np.linspace(0, np.pi, 100)  # Upper half
        x_circle = CONCRETE_THICKNESS + d * np.cos(theta)
        y_circle = d * np.sin(theta)
        ax.plot(x_circle, y_circle, 'w--', linewidth=1, alpha=0.7)
        
        # Add text label at the top of the circle
        ax.text(CONCRETE_THICKNESS, d, f'{d} cm', 
               color='white', ha='right', va='bottom', fontsize=9,
               bbox=dict(facecolor='gray', alpha=0.7, boxstyle='round,pad=0.2'))
    
    # Add angle indicators
    angles = [15, 30, 45]
    for angle in angles:
        angle_rad = math.radians(angle)
        # Draw line at this angle, extending to the edge of the plot
        max_r = 150  # Maximum radial distance to draw the line
        dx = max_r * math.cos(angle_rad)
        dy = max_r * math.sin(angle_rad)
        
        ax.plot([CONCRETE_THICKNESS, CONCRETE_THICKNESS + dx], [0, dy], 
                'w-', linewidth=1, alpha=0.9)
        
        # Add text label in a small box at 45 degrees
        if angle == 45:
            box_x = CONCRETE_THICKNESS + 25 * math.cos(angle_rad)
            box_y = 25 * math.sin(angle_rad)
            ax.text(box_x, box_y, f'{angle}.0 degrees', 
                   color='white', ha='center', va='center', fontsize=10, weight='bold',
                   bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.3'))
    
    # Add a detector circle at 30cm, 45 degrees
    detector_distance = 30
    detector_angle = 45
    angle_rad = math.radians(detector_angle)
    detector_x = CONCRETE_THICKNESS + detector_distance * math.cos(angle_rad)
    detector_y = detector_distance * math.sin(angle_rad)
    
    detector_circle = plt.Circle((detector_x, detector_y), PHANTOM_DIAMETER/2, 
                             fill=False, edgecolor='red', linewidth=2, label='Detector')
    ax.add_patch(detector_circle)
    
    # Add detailed parameter information box
    info_text = (f"Source: {energy} MeV Gamma\n"
               f"Wall: 2.0 ft concrete\n"
               f"Channel: {channel_diameter} cm dia.\n"
               f"Detector: 30.0 cm from wall\n"
               f"Angle: 45.0 degrees")
    
    # Position the info box in the upper left
    info_box = ax.text(0.02, 0.98, info_text,
                     transform=ax.transAxes,
                     fontsize=10,
                     verticalalignment='top',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    # Add an annotation showing the channel diameter with an arrow
    channel_label_x = CONCRETE_THICKNESS + 15
    channel_label_y = channel_diameter * 2
    ax.annotate(f"{channel_diameter} cm", 
              xy=(CONCRETE_THICKNESS, channel_radius), 
              xytext=(channel_label_x, channel_label_y),
              arrowprops=dict(arrowstyle='->',
                            connectionstyle='arc3',
                            color='black'),
              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # Create a custom, scientific-style colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.02, extend='both')
    cbar.set_label('Radiation Flux (particles/cm²/s)', fontsize=12)
    
    # Use scientific notation for colorbar ticks
    cbar.ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
    cbar.ax.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
    
    # Set title
    plt.title(f'Radiation Distribution Outside Wall\n{energy} MeV Gamma, Channel Diameter: {channel_diameter} cm', 
             fontsize=14, fontweight='bold')
    
    # Set axis labels
    ax.set_xlabel('Distance (cm)', fontsize=12)
    ax.set_ylabel('Lateral Distance (cm)', fontsize=12)
    
    # Set axis limits
    ax.set_xlim(x_min_plot, x_max_plot)
    ax.set_ylim(y_min_plot, y_max_plot)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    
    # Add legend
    ax.legend(loc='upper right', framealpha=0.7)
    
    # Save figure with high resolution
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Detailed scientific radiation map saved to {filename}")

# Add this function to the main function's visualization section 
def visualize_all_detailed_radiation_maps(all_results):
    """Create detailed radiation maps for all combinations of energy and channel diameter."""
    logger.info("Creating detailed scientific radiation maps for all parameter combinations...")
    
    for energy in GAMMA_ENERGIES:
        energy_str = str(energy)
        if energy_str in all_results:
            for diameter in CHANNEL_DIAMETERS:
                diameter_str = str(diameter)
                if diameter_str in all_results[energy_str]:
                    # Check if mesh data is available
                    if ('fine_mesh_data' in all_results[energy_str][diameter_str] and
                        'mesh_info' in all_results[energy_str][diameter_str]):
                        try:
                            # Use fine mesh data for detailed visualization
                            fine_mesh_data = all_results[energy_str][diameter_str]['fine_mesh_data']
                            mesh_info = all_results[energy_str][diameter_str]['mesh_info']
                            
                            # Create the detailed scientific visualization
                            output_file = PLOTS_DIR / f'detailed_radiation_map_E{energy}_D{diameter}.png'
                            visualize_detailed_radiation_map(
                                fine_mesh_data,
                                mesh_info['fine_mesh']['dimensions'],
                                mesh_info['fine_mesh']['bounds'],
                                energy,
                                diameter,
                                output_file
                            )
                        except Exception as e:
                            logger.error(f"Error creating detailed radiation map for E={energy}, D={diameter}: {e}")
    
    logger.info("Completed creating detailed scientific radiation maps.")

def visualize_dose_angle_study(results, energy, filename):
    """
    Visualize dose rate as a function of angle for different distances.

    Args:
        results (dict): Results dictionary
        energy (float): Gamma-ray energy in MeV
        filename (str): Output filename
    """
    logger.info(f"Creating enhanced dose angle study for energy {energy} MeV...")

    # Create figure with improved styling
    plt.figure(figsize=(12, 8))

    # Plot dose vs angle for each distance and channel diameter
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*']
    linestyles = ['-', '--', '-.', ':']
    colors = plt.cm.viridis(np.linspace(0, 1, len(CHANNEL_DIAMETERS)))
    
    # Count for labeling
    label_count = 0
    
    # Collect all data points for consistent scaling
    all_doses = []

    # Plot for each channel diameter
    for i, diameter in enumerate(CHANNEL_DIAMETERS):
        diameter_str = str(diameter)
        
        # Check if we have results for this energy
        energy_str = str(energy)
        if energy_str not in results or diameter_str not in results[energy_str]:
            continue

        # Plot for each distance
        for j, distance in enumerate(DETECTOR_DISTANCES):
            angles = []
            doses = []

            # Collect dose data for all angles at this distance
            for angle in DETECTOR_ANGLES:
                key = (distance, angle)
                key_str = str(key)

                if (key_str in results[energy_str][diameter_str]['detector_results'] and 
                    'dose' in results[energy_str][diameter_str]['detector_results'][key_str]):
                    dose = results[energy_str][diameter_str]['detector_results'][key_str]['dose']
                    angles.append(angle)
                    doses.append(dose)
                    all_doses.append(dose)

            # Only plot if we have data
            if angles and doses:
                marker_idx = j % len(markers)
                ls_idx = j % len(linestyles)
                
                # Create label with distance and diameter info
                label = f'D={diameter} cm, r={distance} cm'
                
                # Plot data with semilog y scale for better visualization of wide range of values
                plt.semilogy(angles, doses, marker=markers[marker_idx], linestyle=linestyles[ls_idx], 
                            color=colors[i], label=label, linewidth=2, markersize=8)
                label_count += 1

    # Only add threshold and styling if we have plotted data
    if label_count > 0:
        # Add threshold line if we have any dose data
        if all_doses:
            threshold = 0.1  # rem/hr
            plt.axhline(y=threshold, color='red', linestyle='--', linewidth=2,
                       label=f'Threshold ({threshold} rem/hr)')

        # Add detailed labels
        plt.xlabel('Detector Angle (degrees)', fontsize=12)
        plt.ylabel('Dose Rate (rem/hr)', fontsize=12)
        plt.title(f'Dose Rate vs. Angle for {energy} MeV Gamma Rays', fontsize=14, fontweight='bold')

def save_results_to_json(all_results, filename):
    """
    Save simulation results to a properly formatted JSON file with enhanced 
    data handling for scientific analysis.
    
    Args:
        all_results (dict): Full results dictionary
        filename (str): Output filename for JSON results
        
    Returns:
        bool: Success status
    """
    logger.info(f"Saving comprehensive results to JSON file: {filename}")
    
    # Create parent directory if needed
    parent_dir = Path(filename).parent
    if not parent_dir.exists():
        try:
            parent_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create directory for JSON results: {e}")
            return False
    
    # Create a temporary file first for atomic write
    temp_filename = f"{filename}.tmp"
    
    try:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        
        # Track conversion progress for large datasets
        energy_count = 0
        total_energies = len(all_results)
        
        for energy_str, energy_data in all_results.items():
            energy_count += 1
            logger.info(f"Converting energy {energy_str} data to JSON ({energy_count}/{total_energies})")
            
            json_results[energy_str] = {}
            
            for diameter_str, diameter_data in energy_data.items():
                json_results[energy_str][diameter_str] = {}
                
                # Process each key in diameter data
                for key, value in diameter_data.items():
                    # Handle nested dictionaries
                    if isinstance(value, dict):
                        json_results[energy_str][diameter_str][key] = {}
                        
                        # Special handling for detector_results which can be very large
                        if key == 'detector_results':
                            detector_count = 0
                            total_detectors = len(value)
                            
                            for subkey, subvalue in value.items():
                                detector_count += 1
                                if detector_count % 20 == 0:
                                    logger.info(f"Processing detector {detector_count}/{total_detectors} for E={energy_str}, D={diameter_str}")
                                
                                json_results[energy_str][diameter_str][key][subkey] = {}
                                
                                for subsubkey, subsubvalue in subvalue.items():
                                    if isinstance(subsubvalue, np.ndarray):
                                        json_results[energy_str][diameter_str][key][subkey][subsubkey] = subsubvalue.tolist()
                                    elif isinstance(subsubvalue, (np.float32, np.float64)):
                                        json_results[energy_str][diameter_str][key][subkey][subsubkey] = float(subsubvalue)
                                    elif isinstance(subsubvalue, (np.int32, np.int64)):
                                        json_results[energy_str][diameter_str][key][subkey][subsubkey] = int(subsubvalue)
                                    else:
                                        json_results[energy_str][diameter_str][key][subkey][subsubkey] = subsubvalue
                        else:
                            # Other nested dictionaries
                            for subkey, subvalue in value.items():
                                if isinstance(subvalue, np.ndarray):
                                    json_results[energy_str][diameter_str][key][subkey] = subvalue.tolist()
                                elif isinstance(subvalue, (np.float32, np.float64)):
                                    json_results[energy_str][diameter_str][key][subkey] = float(subvalue)
                                elif isinstance(subvalue, (np.int32, np.int64)):
                                    json_results[energy_str][diameter_str][key][subkey] = int(subvalue)
                                else:
                                    json_results[energy_str][diameter_str][key][subkey] = subvalue
                    # Handle numpy arrays directly
                    elif isinstance(value, np.ndarray):
                        # For large mesh data, log the conversion
                        if key in ['mesh_data', 'fine_mesh_data'] and value.size > 1000000:
                            logger.info(f"Converting large {key} array of size {value.size} for E={energy_str}, D={diameter_str}")
                            
                        json_results[energy_str][diameter_str][key] = value.tolist()
                    # Handle numpy scalar types
                    elif isinstance(value, (np.float32, np.float64)):
                        json_results[energy_str][diameter_str][key] = float(value)
                    elif isinstance(value, (np.int32, np.int64)):
                        json_results[energy_str][diameter_str][key] = int(value)
                    # Handle other types
                    else:
                        json_results[energy_str][diameter_str][key] = value
        
        # Add metadata
        json_results['metadata'] = {
            'creation_time': time.time(),
            'creation_date': time.strftime("%Y-%m-%d %H:%M:%S"),
            'version': '1.0.0',
            'parameters': {
                'concrete_thickness': CONCRETE_THICKNESS,
                'source_distance': SOURCE_DISTANCE,
                'channel_diameters': CHANNEL_DIAMETERS,
                'gamma_energies': GAMMA_ENERGIES,
                'detector_distances': DETECTOR_DISTANCES,
                'detector_angles': DETECTOR_ANGLES
            }
        }
        
        # Write to temporary file first
        logger.info(f"Writing JSON data to temporary file: {temp_filename}")
        with open(temp_filename, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # Move temporary file to final destination
        logger.info(f"Moving temporary file to final destination: {filename}")
        shutil.move(temp_filename, filename)
        
        # Report success
        file_size_mb = Path(filename).stat().st_size / (1024 * 1024)
        logger.info(f"Successfully saved results to {filename} ({file_size_mb:.2f} MB)")
        return True
        
    except Exception as e:
        logger.error(f"Error saving results to JSON: {e}")
        logger.error(traceback.format_exc())
        
        # Try to clean up temporary file
        try:
            if Path(temp_filename).exists():
                Path(temp_filename).unlink()
        except Exception:
            pass
            
        return False

def visualize_comprehensive_radiation_heatmap(mesh_data, mesh_dimensions, mesh_bounds, energy, channel_diameter, filename):
    """
    Create a comprehensive radiation heatmap visualization that displays both the top and bottom hemispheres
    with detailed scientific information, projections, and advanced color mapping.
    
    Args:
        mesh_data (numpy.ndarray): Mesh tally data
        mesh_dimensions (list): Mesh dimensions [nx, ny, nz]
        mesh_bounds (list): Mesh bounds [xmin, xmax, ymin, ymax]
        energy (float): Gamma-ray energy in MeV
        channel_diameter (float): Diameter of the air channel in cm
        filename (str): Output filename
    """
    logger.info(f"Creating comprehensive radiation heatmap for energy {energy} MeV and channel diameter {channel_diameter} cm...")

    # Reshape mesh data
    nx, ny, nz = mesh_dimensions
    xmin, xmax, ymin, ymax = mesh_bounds

    # Extract central slice (z=0)
    central_slice = mesh_data.reshape(nx, ny, nz)[:, :, nz//2]

    # Create coordinate meshes with higher resolution
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Apply adaptive smoothing based on channel diameter
    # Use finer smoothing for smaller channels
    sigma = max(0.25, channel_diameter / 5)  
    logger.info(f"Using adaptive smoothing sigma={sigma} for comprehensive heatmap")
    smoothed_data = gaussian_filter(central_slice, sigma=sigma)

    # Create a high-resolution figure for detailed visualization
    fig, ax = plt.subplots(figsize=(14, 10))

    # Focus on the region outside the shield
    x_min_plot = CONCRETE_THICKNESS - 5
    x_max_plot = CONCRETE_THICKNESS + 200
    y_min_plot = -100  # Include bottom hemisphere
    y_max_plot = 100   # Include top hemisphere
    
    # Get the indices corresponding to the plot range
    x_indices = np.where((X[:, 0] >= x_min_plot) & (X[:, 0] <= x_max_plot))[0]
    y_indices = np.where((Y[0, :] >= y_min_plot) & (Y[0, :] <= y_max_plot))[0]
    
    # Extract the data for the plot range
    X_plot = X[np.ix_(x_indices, y_indices)]
    Y_plot = Y[np.ix_(x_indices, y_indices)]
    flux_plot = smoothed_data[np.ix_(x_indices, y_indices)]
    
    # Apply log transform with proper handling of zeros and negatives
    with np.errstate(divide='ignore', invalid='ignore'):
        # Set minimum threshold
        min_positive = np.min(flux_plot[flux_plot > 0])
        threshold = min_positive * 0.01
        log_data = flux_plot.copy()
        log_data[log_data <= 0] = threshold
        log_data = np.log10(log_data)
    
    # Define a scientific color scale that transitions from blue (low) to red (high)
    # This is reversed from the typical scientific visualization to emphasize high dose regions
    colors_scientific = [
        (0.0, 0.0, 0.4),      # Very dark blue (lowest)
        (0.0, 0.0, 0.8),      # Dark blue
        (0.0, 0.4, 1.0),      # Medium blue
        (0.0, 0.8, 1.0),      # Light blue
        (0.0, 1.0, 0.8),      # Cyan
        (0.0, 1.0, 0.0),      # Green
        (0.8, 1.0, 0.0),      # Light green
        (1.0, 1.0, 0.0),      # Yellow
        (1.0, 0.8, 0.0),      # Light orange
        (1.0, 0.4, 0.0),      # Orange
        (1.0, 0.0, 0.0)       # Red (highest)
    ]
    
    custom_cmap = colors.LinearSegmentedColormap.from_list('scientific_reversed', colors_scientific)
    
    # Create a custom norm that emphasizes mid-range values
    # This makes the visualization more informative by reducing the dominance of extreme values
    vmin = np.min(log_data)
    vmax = np.max(log_data)
    vmid = (vmin + vmax) / 2
    
    # Use TwoSlopeNorm for better visualization of mid-range values
    norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=vmid, vmax=vmax)
    
    # Plot the heatmap with custom colormap and normalization
    im = ax.pcolormesh(X_plot, Y_plot, log_data, cmap=custom_cmap, norm=norm, shading='auto')
    
    # Add concrete shield with proper dimensions
    shield_rect = plt.Rectangle((0, y_min_plot), CONCRETE_THICKNESS, y_max_plot-y_min_plot, 
                              facecolor='gray', alpha=0.5, edgecolor='black', linewidth=2,
                              label='Concrete Shield')
    ax.add_patch(shield_rect)
    
    # Add channel opening
    channel_radius = channel_diameter / 2.0
    channel_rect = plt.Rectangle((0, -channel_radius), CONCRETE_THICKNESS, channel_diameter, 
                               facecolor='white', edgecolor='black', linewidth=1.5,
                               label='Air Channel')
    ax.add_patch(channel_rect)
    
    # Add wall exit line with label
    ax.axvline(x=CONCRETE_THICKNESS, color='black', linestyle='-', linewidth=2)
    ax.text(CONCRETE_THICKNESS+2, y_max_plot-10, 'Wall Exit', 
           rotation=90, fontsize=10, va='top', ha='left',
           bbox={'facecolor': 'white', 'alpha': 0.7, 'boxstyle': 'round,pad=0.2'})
    
    # Add distance markers as semicircles
    distances = [30, 60, 100, 150]
    for d in distances:
        # Full circles to include both top and bottom hemispheres
        theta = np.linspace(0, 2*np.pi, 200)
        x_circle = CONCRETE_THICKNESS + d * np.cos(theta)
        y_circle = d * np.sin(theta)
        ax.plot(x_circle, y_circle, 'w--', linewidth=0.8, alpha=0.6)
        
        # Add distance labels at top and bottom
        ax.text(CONCRETE_THICKNESS, d+5, f'{d} cm', 
              color='white', ha='right', va='bottom', fontsize=9,
              bbox={'facecolor': 'black', 'alpha': 0.6, 'boxstyle': 'round,pad=0.2'})
        ax.text(CONCRETE_THICKNESS, -d-5, f'{d} cm', 
              color='white', ha='right', va='top', fontsize=9,
              bbox={'facecolor': 'black', 'alpha': 0.6, 'boxstyle': 'round,pad=0.2'})
    
    # Add angle indicators
    for angle in [15, 30, 45, 60, 75]:
        # Upper hemisphere
        angle_rad = math.radians(angle)
        dx = 200 * math.cos(angle_rad)
        dy = 200 * math.sin(angle_rad)
        
        ax.plot([CONCRETE_THICKNESS, CONCRETE_THICKNESS + dx], [0, dy], 
               'w-', linewidth=0.8, alpha=0.6)
        
        # Lower hemisphere
        angle_rad = math.radians(-angle)
        dx = 200 * math.cos(angle_rad)
        dy = 200 * math.sin(angle_rad)
        
        ax.plot([CONCRETE_THICKNESS, CONCRETE_THICKNESS + dx], [0, dy], 
               'w-', linewidth=0.8, alpha=0.6)
        
        # Add angle labels - Using direct dict instead of dict() for cleaner syntax
        radius = 75
        label_x = CONCRETE_THICKNESS + radius * math.cos(math.radians(angle))
        label_y = radius * math.sin(math.radians(angle))
        ax.text(label_x, label_y, f'{angle} degrees', 
              color='white', ha='center', va='center', fontsize=9,
              bbox={'facecolor': 'black', 'alpha': 0.6, 'boxstyle': 'round,pad=0.2'})
        
        # Lower hemisphere label - Using direct dict instead of dict() for cleaner syntax
        label_x = CONCRETE_THICKNESS + radius * math.cos(math.radians(-angle))
        label_y = radius * math.sin(math.radians(-angle))
        ax.text(label_x, label_y, f"-{angle} degrees", 
              color='white', ha='center', va='center', fontsize=9,
              bbox={'facecolor': 'black', 'alpha': 0.6, 'boxstyle': 'round,pad=0.2'})
    
    # Add high dose regions contour lines
    high_dose_threshold = vmax - (vmax - vmid) * 0.3
    contour = ax.contour(X_plot, Y_plot, log_data, 
                       levels=[high_dose_threshold], 
                       colors=['yellow'], linewidths=1.5, alpha=0.7)
    
    # Add detector positions
    for distance in DETECTOR_DISTANCES:
        for angle in DETECTOR_ANGLES:
            angle_rad = math.radians(angle)
            det_x = CONCRETE_THICKNESS + distance * math.cos(angle_rad)
            det_y = distance * math.sin(angle_rad)
            
            # Add scatter points for detector positions
            ax.scatter(det_x, det_y, color='cyan', s=20, alpha=0.9, edgecolor='black', linewidth=0.5)
    
    # Add comprehensive title
    plt.title(f'Comprehensive Radiation Distribution Heatmap\n{energy} MeV Gamma-Ray, Channel Diameter: {channel_diameter} cm', 
             fontsize=14, fontweight='bold')
    
    # Add detailed parameter information box
    param_text = (
        f"Parameters:\n"
        f"• Source: {energy} MeV γ-rays\n"
        f"• Source Distance: {SOURCE_DISTANCE/30.48:.1f} ft\n"
        f"• Shield: {CONCRETE_THICKNESS/30.48:.1f} ft concrete\n"
        f"• Channel: {channel_diameter} cm diameter\n"
        f"• Top & Bottom Hemispheres Shown\n"
        f"• Color: Log10 Radiation Flux"
    )
    
    # Position the parameter box in the upper left corner
    plt.annotate(param_text, 
                xy=(0.02, 0.98), xycoords='figure fraction',
                bbox={'boxstyle': 'round,pad=0.5', 'facecolor': 'white', 'alpha': 0.8},
                fontsize=10, verticalalignment='top')
    
    # Add methodological information
    method_text = (
        f"Simulation Methods:\n"
        f"• Monte Carlo Transport\n"
        f"• Variance Reduction\n"
        f"• Adaptive Resolution\n"
        f"• σ={sigma:.2f} cm smoothing"
    )
    
    # Position the method box in the upper right corner
    plt.annotate(method_text, 
                xy=(0.98, 0.98), xycoords='figure fraction',
                bbox={'boxstyle': 'round,pad=0.5', 'facecolor': 'white', 'alpha': 0.8},
                fontsize=10, verticalalignment='top', horizontalalignment='right')
    
    # Add a scientific colorbar with better labeling
    cbar = plt.colorbar(im, ax=ax, pad=0.02, extend='both')
    cbar.set_label('Log₁₀(Radiation Flux) [particles/cm²]', fontsize=12)
    
    # Add axis labels
    ax.set_xlabel('Distance from Source (cm)', fontsize=12)
    ax.set_ylabel('Lateral Distance (cm)', fontsize=12)
    
    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    
    # Save figure with high resolution
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Comprehensive radiation heatmap saved to {filename}")

def visualize_full_configuration(mesh_data, mesh_dimensions, mesh_bounds, energy, channel_diameter, filename):
    """
    Create a comprehensive visualization of the full simulation configuration including 
    source, shield, channel, detector positions, and radiation distribution.

    Args:
        mesh_data (numpy.ndarray): Mesh tally data
        mesh_dimensions (list): Mesh dimensions [nx, ny, nz]
        mesh_bounds (list): Mesh bounds [xmin, xmax, ymin, ymax]
        energy (float): Gamma-ray energy in MeV
        channel_diameter (float): Diameter of the air channel in cm
        filename (str): Output filename
    """
    logger.info(f"Creating full configuration visualization for energy {energy} MeV and channel diameter {channel_diameter} cm...")

    # Reshape mesh data
    nx, ny, nz = mesh_dimensions
    xmin, xmax, ymin, ymax = mesh_bounds

    # Extract central slice (z=0)
    central_slice = mesh_data.reshape(nx, ny, nz)[:, :, nz//2]

    # Create coordinate meshes
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Apply modest smoothing for visualization
    sigma = max(0.5, channel_diameter / 4)
    smoothed_data = gaussian_filter(central_slice, sigma=sigma)

    # Apply log scale for better visualization with handling for zeros
    with np.errstate(divide='ignore', invalid='ignore'):
        log_data = np.log10(smoothed_data)
        min_val = np.min(log_data[np.isfinite(log_data)])
        log_data[~np.isfinite(log_data)] = min_val - 1

    # Create a large figure for detailed visualization
    plt.figure(figsize=(18, 10))

    # Plot radiation map
    cmap = plt.cm.viridis
    norm = colors.Normalize(vmin=min_val, vmax=np.max(log_data))
    plt.pcolormesh(X, Y, log_data, cmap=cmap, norm=norm, shading='auto', alpha=0.8)

    # Add colorbar
    cbar = plt.colorbar()
    cbar.set_label('Log₁₀(Flux) [particles/cm²]', fontsize=12)

    # Draw the concrete shield with semi-transparent gray
    shield_rect = plt.Rectangle((0, ymin), CONCRETE_THICKNESS, ymax-ymin, 
                              facecolor='gray', alpha=0.3, edgecolor='black', 
                              linewidth=2, label='Concrete Shield')
    plt.gca().add_patch(shield_rect)

    # Add channel (white space through shield)
    channel_radius = channel_diameter / 2.0
    channel_rect = plt.Rectangle((0, -channel_radius), CONCRETE_THICKNESS, channel_diameter, 
                               facecolor='white', edgecolor='black', linewidth=1.5,
                               label='Air Channel')
    plt.gca().add_patch(channel_rect)

    # Add source point with highlight
    plt.plot(-SOURCE_DISTANCE, 0, 'ro', markersize=12, markeredgecolor='black', label='Source')
    
    # Add source annotation
    plt.annotate(f'{energy} MeV γ-source', 
                xy=(-SOURCE_DISTANCE, 0), 
                xytext=(-SOURCE_DISTANCE-30, 15),
                arrowprops={'arrowstyle': '->',
                           'connectionstyle': 'arc3',
                           'color': 'red'},
                bbox={'boxstyle': 'round,pad=0.5', 'facecolor': 'white', 'alpha': 0.8})
                
    # Add angle label at 60cm from origin
    label_x = CONCRETE_THICKNESS + 60 * math.cos(angle_rad)
    label_y = 60 * math.sin(angle_rad)
    plt.text(label_x, label_y, f'{angle} degrees', 
            ha='center', va='center', fontsize=10,
            bbox={'facecolor': 'white', 'alpha': 0.7, 'boxstyle': 'round'})
            
    # Add distance label
    plt.text(CONCRETE_THICKNESS, distance + 5, f'{distance} cm', 
            ha='right', va='bottom', fontsize=10,
                arrowprops=dict(arrowstyle='->',
                              connectionstyle='arc3',
                              color='red'),
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

    # Add detector positions at various distances and angles
    for distance in DETECTOR_DISTANCES:
        for angle in DETECTOR_ANGLES:
            angle_rad = math.radians(angle)
            det_x = CONCRETE_THICKNESS + distance * math.cos(angle_rad)
            det_y = distance * math.sin(angle_rad)
            
            # Draw detector circle
            detector = plt.Circle((det_x, det_y), PHANTOM_DIAMETER/2, 
                                fill=False, edgecolor='blue', linestyle='-', 
                                linewidth=1, alpha=0.7)
            plt.gca().add_patch(detector)
            
    # Add special highlight for 0 degree, 100cm detector
    highlight_x = CONCRETE_THICKNESS + 100
    highlight_y = 0
    highlight_detector = plt.Circle((highlight_x, highlight_y), PHANTOM_DIAMETER/2, 
                                  fill=False, edgecolor='blue', linestyle='-', 
                                  linewidth=2.5, label='Detector (100cm, 0 degrees)')
    plt.gca().add_patch(highlight_detector)
    plt.annotate('Detector\n(100cm, 0 degrees)', 
                xy=(highlight_x, highlight_y), 
                xytext=(highlight_x + 20, highlight_y + 15),
                arrowprops=dict(arrowstyle='->',
                              connectionstyle='arc3',
                              color='blue'),
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

    # Add distance markers as semicircles from channel exit
    for distance in DETECTOR_DISTANCES:
        for angle in DETECTOR_ANGLES:
            angle_rad = math.radians(angle)
            det_x = CONCRETE_THICKNESS + distance * math.cos(angle_rad)
            det_y = distance * math.sin(angle_rad)
            
            # Draw detector circle
            detector = plt.Circle((det_x, det_y), PHANTOM_DIAMETER/2, 
                                fill=False, edgecolor='blue', linestyle='-', 
                                linewidth=1, alpha=0.7)
            plt.gca().add_patch(detector)
            
    # Add special highlight for 0 degree, 100cm detector
    highlight_x = CONCRETE_THICKNESS + 100
    highlight_y = 0
    highlight_detector = plt.Circle((highlight_x, highlight_y), PHANTOM_DIAMETER/2, 
                                  fill=False, edgecolor='blue', linestyle='-', 
                                  linewidth=2.5, label='Detector (100cm, 0 degrees)')
    plt.gca().add_patch(highlight_detector)
    plt.annotate('Detector\n(100cm, 0 degrees)', 
                xy=(highlight_x, highlight_y), 
                xytext=(highlight_x + 20, highlight_y + 15),
                arrowprops=dict(arrowstyle='->',
                              connectionstyle='arc3',
                              color='blue'),
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

    # Save figure with high resolution
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Full configuration visualization saved to {filename}")

def visualize_flux_spectra_vs_distance(results, energy, channel_diameter, filename):
    """
    Create a visualization of gamma-ray flux spectra as a function of distance from the shield.
    This plots spectra for different detector positions at angle 0 degrees.

    Args:
        results (dict): Results dictionary
        energy (float): Gamma-ray energy in MeV
        channel_diameter (float): Diameter of the air channel in cm
        filename (str): Output filename
    """
    logger.info(f"Creating flux spectra vs distance visualization for energy {energy} MeV and channel diameter {channel_diameter} cm...")

    # Create figure with optimized size
    plt.figure(figsize=(12, 8))
    
    # Check if we have data for this energy and diameter
    energy_str = str(energy)
    diameter_str = str(channel_diameter)
    
    if energy_str not in results or diameter_str not in results[energy_str]:
        plt.text(0.5, 0.5, f"No data available for energy {energy} MeV and diameter {channel_diameter} cm",
                ha='center', va='center', transform=plt.gca().transAxes, 
                fontsize=14, bbox=dict(facecolor='white', alpha=0.8))
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    # If we don't have explicit energy bins or flux spectrum, try to construct from detector results
    if ('energy_bins' not in results[energy_str][diameter_str] or 
        'flux_spectrum' not in results[energy_str][diameter_str]):
        
        # Create default energy bins centered around the source energy
        energy_range = [0.01, 0.05, 0.1, 0.5, energy*0.9, energy, energy*1.1, energy*1.5, energy*2.0]
        energy_bins = np.array(sorted(set(energy_range)))
        
        # Create a color map for distances
        cmap = plt.cm.viridis
        
        # Store distances for legend
        distances_with_data = []
        
        # Filter for 0-degree angle detectors
        angle = 0
        
        # Plot for each distance
        for i, distance in enumerate(sorted(DETECTOR_DISTANCES)):
            key = (distance, angle)
            key_str = str(key)
            
            if key_str in results[energy_str][diameter_str]['detector_results']:
                detector_data = results[energy_str][diameter_str]['detector_results'][key_str]
                
                # If we have flux data
                if 'flux' in detector_data:
                    flux = detector_data['flux']
                    
                    # Calculate flux at each energy bin by estimating a simple spectrum shape
                    # This is a simplification since we don't have detailed energy-dependent data
                    flux_per_bin = []
                    
                    # Create a simplified spectrum shape with peak at source energy
                    for e_low, e_high in zip(energy_bins[:-1], energy_bins[1:]):
                        e_center = (e_low + e_high) / 2
                        
                        # Simple normal distribution around source energy
                        rel_flux = np.exp(-((e_center - energy) / (energy * 0.2))**2)
                        
                        # Scale by total flux and bin width
                        bin_flux = flux * rel_flux * (e_high - e_low) / (energy_bins[-1] - energy_bins[0])
                        flux_per_bin.append(bin_flux)
                    
                    # Normalize to make different distances comparable
                    if flux_per_bin:
                        # Adjust by distance factor (1/r²)
                        distance_factor = (distance / min(DETECTOR_DISTANCES))**2
                        normalized_flux = np.array(flux_per_bin) * distance_factor
                        
                        # Plot spectrum
                        color = cmap(i / max(1, len(DETECTOR_DISTANCES) - 1))
                        plt.step(energy_bins[:-1], normalized_flux, where='post', 
                                color=color, linewidth=2,
                                label=f'Distance = {distance} cm')
                        
                        distances_with_data.append(distance)
        
        if not distances_with_data:
            plt.text(0.5, 0.5, "No flux data available for 0 degrees detectors",
                    ha='center', va='center', transform=plt.gca().transAxes,
                    fontsize=14, bbox=dict(facecolor='white', alpha=0.8))
        else:
            # Add source energy marker
            plt.axvline(x=energy, color='red', linestyle='--', linewidth=2,
                       label=f'Source Energy ({energy} MeV)')
            
            # Add annotation for 1/r² scaling
            plt.annotate('Note: Fluxes normalized by 1/r² factor\nfor better comparison',
                        xy=(0.75, 0.1), xycoords='axes fraction',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8),
                        fontsize=10)
    else:
        # Extract energy bins and base flux spectrum
        energy_bins = results[energy_str][diameter_str]['energy_bins']
        base_spectrum = results[energy_str][diameter_str]['flux_spectrum']
        
        # Plot baseline spectrum (at wall exit)
        plt.step(energy_bins[:-1], base_spectrum, where='post', color='black', 
                linewidth=2, label='Shield Exit (Wall Surface)')
        
        # Create a color map for distances
        cmap = plt.cm.viridis
        
        # Store distances for legend
        distances_with_data = []
        
        # Filter for 0-degree angle detectors
        angle = 0
        
        # Plot for each distance
        for i, distance in enumerate(sorted(DETECTOR_DISTANCES)):
            key = (distance, angle)
            key_str = str(key)
            
            if key_str in results[energy_str][diameter_str]['detector_results']:
                detector_data = results[energy_str][diameter_str]['detector_results'][key_str]
                
                # If we have flux data
                if 'flux' in detector_data:
                    total_flux = detector_data['flux']
                    
                    # Scale base spectrum shape by detector flux and distance attenuation
                    # This is an approximation - actual spectra would need energy-dependent detector tallies
                    distance_factor = distance**2
                    spectrum_at_distance = base_spectrum * (total_flux / sum(base_spectrum)) * distance_factor
                    
                    # Plot scaled spectrum
                    color = cmap(i / max(1, len(DETECTOR_DISTANCES) - 1))
                    plt.step(energy_bins[:-1], spectrum_at_distance, where='post', 
                            color=color, linewidth=2,
                            label=f'Distance = {distance} cm')
                    
                    distances_with_data.append(distance)
    
    # Add plot styling
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Energy (MeV)', fontsize=12)
    plt.ylabel('Relative Flux (arbitrary units)', fontsize=12)
    plt.title(f'Gamma-Ray Flux Spectra at Different Distances\nEnergy: {energy} MeV, Channel Diameter: {channel_diameter} cm',
             fontsize=14, fontweight='bold')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    # Add detailed legend with distance information
    if distances_with_data:
        plt.legend(fontsize=10)
        
        # Add information box with simulation details
        info_text = (
            f"Simulation Parameters:\n"
            f"• Source Energy: {energy} MeV\n"
            f"• Channel Diameter: {channel_diameter} cm\n"
            f"• Concrete Thickness: {CONCRETE_THICKNESS/30.48:.1f} ft\n"
            f"• Detector: ICRU phantom, {PHANTOM_DIAMETER} cm dia."
        )
        
        plt.annotate(info_text, xy=(0.02, 0.02), xycoords='figure fraction',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8),
                    fontsize=10, verticalalignment='bottom')
    
    # Save figure with high resolution
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Flux spectra vs distance visualization saved to {filename}")

def visualize_enhanced_dose_vs_angle_study(results, energy, filename):
    """
    Create an enhanced visualization showing dose rate as a function of angle
    for different distances with improved visual styling and comprehensive data display.

    Args:
        results (dict): Results dictionary
        energy (float): Gamma-ray energy in MeV
        filename (str): Output filename
    """
    logger.info(f"Creating enhanced dose angle study for energy {energy} MeV...")

    # Create figure with improved styling
    plt.figure(figsize=(12, 8))

    # Plot dose vs angle for each distance and channel diameter
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*']
    linestyles = ['-', '--', '-.', ':']
    colors = plt.cm.viridis(np.linspace(0, 1, len(CHANNEL_DIAMETERS)))
    
    # Count for labeling
    label_count = 0
    
    # Collect all data points for consistent scaling
    all_doses = []

    # Plot for each channel diameter
    for i, diameter in enumerate(CHANNEL_DIAMETERS):
        diameter_str = str(diameter)
        
        # Check if we have results for this energy
        energy_str = str(energy)
        if energy_str not in results or diameter_str not in results[energy_str]:
            continue

        # Plot for each distance
        for j, distance in enumerate(DETECTOR_DISTANCES):
            angles = []
            doses = []

            # Collect dose data for all angles at this distance
            for angle in DETECTOR_ANGLES:
                key = (distance, angle)
                key_str = str(key)

                if (key_str in results[energy_str][diameter_str]['detector_results'] and 
                    'dose' in results[energy_str][diameter_str]['detector_results'][key_str]):
                    dose = results[energy_str][diameter_str]['detector_results'][key_str]['dose']
                    angles.append(angle)
                    doses.append(dose)
                    all_doses.append(dose)

            # Only plot if we have data
            if angles and doses:
                marker_idx = j % len(markers)
                ls_idx = j % len(linestyles)
                
                # Create label with distance and diameter info
                label = f'D={diameter} cm, r={distance} cm'
                
                # Plot data with semilog y scale for better visualization of wide range of values
                plt.semilogy(angles, doses, marker=markers[marker_idx], linestyle=linestyles[ls_idx], 
                            color=colors[i], label=label, linewidth=2, markersize=8)
                label_count += 1

    # Only add threshold and styling if we have plotted data
    if label_count > 0:
        # Add threshold line if we have any dose data
        if all_doses:
            threshold = 0.1  # rem/hr
            plt.axhline(y=threshold, color='red', linestyle='--', linewidth=2,
                       label=f'Threshold ({threshold} rem/hr)')

        # Add detailed labels
        plt.xlabel('Detector Angle (degrees)', fontsize=12)
        plt.ylabel('Dose Rate (rem/hr)', fontsize=12)
        plt.title(f'Dose Rate vs. Angle for {energy} MeV Gamma Rays', fontsize=14, fontweight='bold')
        
        # Add grid
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        
        # Add legend with better placement and styling
        if label_count > 10:
            # Use a smaller font and place outside the plot if we have many labels
            plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=8)
        else:
            plt.legend(loc='best', fontsize=10)
            
        # Add information box
        info_text = (
            f"Energy: {energy} MeV\n"
            f"Detector Distances: {min(DETECTOR_DISTANCES)}-{max(DETECTOR_DISTANCES)} cm\n"
            f"Channel Diameters: {min(CHANNEL_DIAMETERS)}-{max(CHANNEL_DIAMETERS)} cm"
        )
        plt.annotate(info_text, xy=(0.02, 0.02), xycoords='figure fraction',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7),
                    fontsize=10, verticalalignment='bottom')
        
        # Adjust y-range for better visibility
        if all_doses:
            plt.ylim(min(all_doses) / 10, max(all_doses) * 5)  # Extend a bit beyond data range
            
        # Adjust x range to focus on the angles we have data for
        plt.xlim(-5, max(DETECTOR_ANGLES) + 5)
    else:
        # Add message if no data is available
        plt.text(0.5, 0.5, "No dose data available for this energy", 
                ha='center', va='center', transform=plt.gca().transAxes,
                fontsize=14, fontweight='bold')
    
    # Save with high resolution
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Enhanced dose angle study saved to {filename}")


if __name__ == "__main__":
    print("Script started")
    sys.stdout.flush()
    main()
    print("Script ended")
    sys.stdout.flush()
