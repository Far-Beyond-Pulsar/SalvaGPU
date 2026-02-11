//! GPU-accelerated fluid simulation using WGPU
//!
//! This module provides GPU acceleration for the entire fluid simulation,
//! keeping all particle data and computations on the GPU to minimize
//! CPU-GPU data transfers. A delta-based synchronization system is used
//! to efficiently update only changed data.
//!
//! # Features
//!
//! - Full GPU-resident simulation (positions, velocities, forces stay on GPU)
//! - Delta-based buffer synchronization
//! - Compute shader-based SPH operations:
//!   - Spatial hashing for neighbor search
//!   - Density computation
//!   - Pressure, viscosity, and surface tension forces
//!   - Time integration
//!
//! # Usage
//!
//! ```no_run
//! use salva::gpu::GpuFluidSolver;
//! use salva::math::Vector;
//!
//! # async fn example() -> Result<(), String> {
//! // Create GPU solver
//! let mut solver = GpuFluidSolver::new(10000).await?;
//!
//! // Initialize with fluid data
//! // solver.init_from_fluids(&fluids);
//!
//! // Run simulation steps on GPU
//! let dt = 0.016;
//! let gravity = Vector::new(0.0, -9.81);
//! solver.step(dt, &gravity);
//!
//! // Optionally read back data (expensive!)
//! let particles = solver.read_particles().await?;
//! # Ok(())
//! # }
//! ```

#[cfg(feature = "gpu-acceleration")]
mod buffer;
#[cfg(feature = "gpu-acceleration")]
mod context;
#[cfg(feature = "gpu-acceleration")]
mod particle_data;
#[cfg(feature = "gpu-acceleration")]
mod pipeline;
#[cfg(feature = "gpu-acceleration")]
mod solver;

#[cfg(feature = "gpu-acceleration")]
pub use buffer::{DeltaBufferManager, ReadbackBuffer};
#[cfg(feature = "gpu-acceleration")]
pub use context::{GpuContext, GpuContextError};
#[cfg(feature = "gpu-acceleration")]
pub use particle_data::{GpuParticle, ParticleDelta, SimulationParams};
#[cfg(feature = "gpu-acceleration")]
pub use pipeline::GpuPipelines;
#[cfg(feature = "gpu-acceleration")]
pub use solver::GpuFluidSolver;
