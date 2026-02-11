//! GPU-compatible particle data structures

use bytemuck::{Pod, Zeroable};
use encase::ShaderType;

/// GPU-compatible 2D particle data (Structure of Arrays layout)
#[cfg(feature = "dim2")]
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, ShaderType)]
pub struct GpuParticle2D {
    /// Position (x, y) and padding
    pub position: [f32; 4],
    /// Velocity (x, y) and padding
    pub velocity: [f32; 4],
    /// Force accumulator (x, y) and padding
    pub force: [f32; 4],
    /// Density, pressure, mass, padding
    pub density_pressure_mass: [f32; 4],
}

/// GPU-compatible 3D particle data (Structure of Arrays layout)
#[cfg(feature = "dim3")]
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, ShaderType)]
pub struct GpuParticle3D {
    /// Position (x, y, z, padding)
    pub position: [f32; 4],
    /// Velocity (x, y, z, padding)
    pub velocity: [f32; 4],
    /// Force accumulator (x, y, z, padding)
    pub force: [f32; 4],
    /// Density, pressure, mass, padding
    pub density_pressure_mass: [f32; 4],
}

#[cfg(feature = "dim2")]
pub type GpuParticle = GpuParticle2D;

#[cfg(feature = "dim3")]
pub type GpuParticle = GpuParticle3D;

/// Simulation parameters passed to GPU shaders as uniforms
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, ShaderType)]
pub struct SimulationParams {
    /// Time step
    pub dt: f32,
    /// Inverse of time step
    pub inv_dt: f32,
    /// Smoothing kernel radius
    pub h: f32,
    /// Rest density
    pub density0: f32,
    
    /// Gravity (x, y, z, padding)
    pub gravity: [f32; 4],
    
    /// Particle radius
    pub particle_radius: f32,
    /// Particle mass
    pub particle_mass: f32,
    /// Number of particles
    pub num_particles: u32,
    /// Grid cell size
    pub grid_cell_size: f32,
    
    /// Viscosity coefficient
    pub viscosity: f32,
    /// Surface tension coefficient
    pub surface_tension: f32,
    /// Pressure stiffness
    pub pressure_stiffness: f32,
    /// Padding
    pub _padding: f32,
}

impl Default for SimulationParams {
    fn default() -> Self {
        Self {
            dt: 0.016,
            inv_dt: 1.0 / 0.016,
            h: 0.1,
            density0: 1000.0,
            gravity: [0.0, -9.81, 0.0, 0.0],
            particle_radius: 0.025,
            particle_mass: 1.0,
            num_particles: 0,
            grid_cell_size: 0.1,
            viscosity: 0.01,
            surface_tension: 0.01,
            pressure_stiffness: 1000.0,
            _padding: 0.0,
        }
    }
}

/// Spatial hash grid cell
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GridCell {
    /// Starting index in the particle index buffer
    pub start: u32,
    /// Number of particles in this cell
    pub count: u32,
}

/// Particle index with spatial hash
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct ParticleIndex {
    /// Particle index
    pub index: u32,
    /// Spatial hash value
    pub hash: u32,
}

impl PartialEq for ParticleIndex {
    fn eq(&self, other: &Self) -> bool {
        self.hash == other.hash
    }
}

impl Eq for ParticleIndex {}

impl PartialOrd for ParticleIndex {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ParticleIndex {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.hash.cmp(&other.hash)
    }
}

/// Delta update for particle additions/removals
pub struct ParticleDelta {
    /// Indices of particles to remove
    pub removals: Vec<usize>,
    /// New particles to add
    pub additions: Vec<GpuParticle>,
}

impl ParticleDelta {
    /// Creates a new empty delta
    pub fn new() -> Self {
        Self {
            removals: Vec::new(),
            additions: Vec::new(),
        }
    }

    /// Checks if the delta is empty
    pub fn is_empty(&self) -> bool {
        self.removals.is_empty() && self.additions.is_empty()
    }

    /// Clears the delta
    pub fn clear(&mut self) {
        self.removals.clear();
        self.additions.clear();
    }
}

impl Default for ParticleDelta {
    fn default() -> Self {
        Self::new()
    }
}
