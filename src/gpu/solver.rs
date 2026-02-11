//! GPU-accelerated fluid solver

use crate::gpu::{
    buffer::DeltaBufferManager,
    context::GpuContext,
    particle_data::{GpuParticle, ParticleIndex, SimulationParams},
    pipeline::GpuPipelines,
};
use crate::math::{Real, Vector};
use crate::object::Fluid;
use std::mem::size_of;
use wgpu::{BufferUsages, CommandEncoderDescriptor, util::DeviceExt};

/// GPU-resident fluid solver that keeps all simulation state on the GPU
pub struct GpuFluidSolver {
    context: GpuContext,
    pipelines: GpuPipelines,
    
    // GPU buffers
    particles_buffer: DeltaBufferManager<GpuParticle>,
    particle_indices_buffer: DeltaBufferManager<ParticleIndex>,
    params_buffer: wgpu::Buffer,
    
    // Simulation parameters
    params: SimulationParams,
    
    // Particle count
    num_particles: usize,
}

impl GpuFluidSolver {
    /// Creates a new GPU fluid solver
    pub async fn new(initial_capacity: usize) -> Result<Self, String> {
        let context = GpuContext::new().await.map_err(|e| e.to_string())?;
        let pipelines = GpuPipelines::new(&context.device);
        
        // Create buffers
        let particles_buffer = DeltaBufferManager::new(
            &context.device,
            initial_capacity,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            "Particles Buffer",
        );
        
        let particle_indices_buffer = DeltaBufferManager::new(
            &context.device,
            initial_capacity,
            BufferUsages::STORAGE,
            "Particle Indices Buffer",
        );
        
        let params = SimulationParams::default();
        let params_buffer = context.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Simulation Params"),
            contents: bytemuck::cast_slice(&[params]),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        
        Ok(Self {
            context,
            pipelines,
            particles_buffer,
            particle_indices_buffer,
            params_buffer,
            params,
            num_particles: 0,
        })
    }
    
    /// Initializes the solver with fluid data from CPU
    pub fn init_from_fluids(&mut self, fluids: &[Fluid]) {
        // Convert CPU fluid data to GPU format
        let mut gpu_particles = Vec::new();
        
        for fluid in fluids {
            for i in 0..fluid.num_particles() {
                let pos = fluid.positions[i];
                let vel = fluid.velocities[i];
                let mass = fluid.particle_mass(i);
                
                let gpu_particle = GpuParticle {
                    #[cfg(feature = "dim2")]
                    position: [pos.x, pos.y, 0.0, 1.0],
                    #[cfg(feature = "dim3")]
                    position: [pos.x, pos.y, pos.z, 1.0],
                    
                    #[cfg(feature = "dim2")]
                    velocity: [vel.x, vel.y, 0.0, 0.0],
                    #[cfg(feature = "dim3")]
                    velocity: [vel.x, vel.y, vel.z, 0.0],
                    
                    force: [0.0, 0.0, 0.0, 0.0],
                    density_pressure_mass: [fluid.density0, 0.0, mass, 0.0],
                };
                
                gpu_particles.push(gpu_particle);
            }
        }
        
        self.num_particles = gpu_particles.len();
        self.params.num_particles = self.num_particles as u32;
        
        // Upload to GPU
        if gpu_particles.len() > self.particles_buffer.capacity() {
            self.particles_buffer.resize(
                &self.context.device,
                gpu_particles.len(),
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                "Particles Buffer",
            );
            self.particle_indices_buffer.resize(
                &self.context.device,
                gpu_particles.len(),
                BufferUsages::STORAGE,
                "Particle Indices Buffer",
            );
        }
        
        self.particles_buffer.update_range(0, &gpu_particles);
        self.particles_buffer.mark_all_dirty();
        
        // Update params
        self.context.queue.write_buffer(
            &self.params_buffer,
            0,
            bytemuck::cast_slice(&[self.params]),
        );
    }
    
    /// Performs one simulation step entirely on the GPU
    pub fn step(&mut self, dt: Real, gravity: &Vector<Real>) {
        // Update simulation parameters
        self.params.dt = dt;
        self.params.inv_dt = 1.0 / dt;
        
        #[cfg(feature = "dim2")]
        {
            self.params.gravity = [gravity.x, gravity.y, 0.0, 0.0];
        }
        #[cfg(feature = "dim3")]
        {
            self.params.gravity = [gravity.x, gravity.y, gravity.z, 0.0];
        }
        
        self.context.queue.write_buffer(
            &self.params_buffer,
            0,
            bytemuck::cast_slice(&[self.params]),
        );
        
        // Sync any pending CPU updates to GPU
        self.particles_buffer.sync_to_gpu(&self.context.queue);
        self.particle_indices_buffer.sync_to_gpu(&self.context.queue);
        
        // Create command encoder
        let mut encoder = self.context.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Simulation Step"),
        });
        
        let workgroup_size = 256;
        let num_workgroups = (self.num_particles as u32 + workgroup_size - 1) / workgroup_size;
        
        // Step 1: Compute spatial hashes
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Spatial Hash Pass"),
                timestamp_writes: None,
            });
            
            let bind_group = self.context.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Spatial Hash Bind Group"),
                layout: &self.pipelines.spatial_hash_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.particles_buffer.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.particle_indices_buffer.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.params_buffer.as_entire_binding(),
                    },
                ],
            });
            
            pass.set_pipeline(&self.pipelines.spatial_hash_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(num_workgroups, 1, 1);
        }
        
        // Step 2: Compute densities
        // Note: In a full implementation, we'd sort particle_indices here and build grid cells
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Density Pass"),
                timestamp_writes: None,
            });
            
            // Create a dummy grid cells buffer for now
            let dummy_grid_cells = vec![[0u32, self.num_particles as u32]; 1];
            let grid_cells_buffer = self.context.device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("Grid Cells"),
                    contents: bytemuck::cast_slice(&dummy_grid_cells),
                    usage: BufferUsages::STORAGE,
                }
            );
            
            let bind_group = self.context.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Density Bind Group"),
                layout: &self.pipelines.density_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.particles_buffer.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.particle_indices_buffer.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: grid_cells_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.params_buffer.as_entire_binding(),
                    },
                ],
            });
            
            pass.set_pipeline(&self.pipelines.density_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(num_workgroups, 1, 1);
        }
        
        // Step 3: Compute forces
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Forces Pass"),
                timestamp_writes: None,
            });
            
            let bind_group = self.context.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Forces Bind Group"),
                layout: &self.pipelines.forces_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.particles_buffer.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.particle_indices_buffer.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.params_buffer.as_entire_binding(),
                    },
                ],
            });
            
            pass.set_pipeline(&self.pipelines.forces_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(num_workgroups, 1, 1);
        }
        
        // Step 4: Integrate
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Integration Pass"),
                timestamp_writes: None,
            });
            
            let bind_group = self.context.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Integration Bind Group"),
                layout: &self.pipelines.integrate_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.particles_buffer.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.params_buffer.as_entire_binding(),
                    },
                ],
            });
            
            pass.set_pipeline(&self.pipelines.integrate_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(num_workgroups, 1, 1);
        }
        
        // Submit commands
        let _ = self.context.queue.submit(Some(encoder.finish()));
    }
    
    /// Reads particle data back from GPU (expensive - use sparingly!)
    pub async fn read_particles(&self) -> Result<Vec<GpuParticle>, String> {
        use crate::gpu::buffer::ReadbackBuffer;
        
        let readback = ReadbackBuffer::new(
            &self.context.device,
            self.num_particles,
            "Particle Readback",
        );
        
        let mut encoder = self.context.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Readback Encoder"),
        });
        
        encoder.copy_buffer_to_buffer(
            self.particles_buffer.buffer(),
            0,
            readback.buffer(),
            0,
            (self.num_particles * size_of::<GpuParticle>()) as u64,
        );
        
        let _ = self.context.queue.submit(Some(encoder.finish()));
        
        readback.read().await
    }
    
    /// Returns the number of particles in the simulation
    pub fn num_particles(&self) -> usize {
        self.num_particles
    }
    
    /// Updates simulation parameters
    pub fn set_params(&mut self, params: SimulationParams) {
        self.params = params;
        self.context.queue.write_buffer(
            &self.params_buffer,
            0,
            bytemuck::cast_slice(&[self.params]),
        );
    }
    
    /// Returns a reference to the GPU context
    pub fn context(&self) -> &GpuContext {
        &self.context
    }
}
