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
    grid_cells_buffer: wgpu::Buffer,
    dfsph_data_buffer: wgpu::Buffer,
    velocity_corrections_buffer: wgpu::Buffer,
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
        
        // Grid cells buffer (dummy 1 cell for now)
        let grid_cells_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Grid Cells Buffer"),
            size: (size_of::<[u32; 2]>() * 1) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // DFSPH data buffer (alpha, predicted_density, divergence per particle)
        let dfsph_data_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DFSPH Data Buffer"),
            size: (size_of::<[f32; 4]>() * initial_capacity) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Velocity corrections buffer
        let velocity_corrections_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Velocity Corrections Buffer"),
            size: (size_of::<[f32; 4]>() * initial_capacity) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
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
            grid_cells_buffer,
            dfsph_data_buffer,
            velocity_corrections_buffer,
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
                    density_pressure_mass_alpha: [fluid.density0, 0.0, mass, 0.0],
                    velocity_change: [0.0, 0.0, 0.0, 0.0],
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
    
    /// Performs one simulation step entirely on the GPU using DFSPH
    /// Matches the CPU DFSPHSolver::step() implementation
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
        
        let workgroup_size = 256;
        let num_workgroups = (self.num_particles as u32 + workgroup_size - 1) / workgroup_size;
        
        // DFSPH Algorithm (matching CPU implementation):
        // 1. Compute alpha (stiffness factors)
        // 2. Divergence solve (make velocity field divergence-free)
        // 3. Apply non-pressure forces and predict advection
        // 4. Integrate positions
        // 5. Pressure solve (correct density errors)
        
        let mut encoder = self.context.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("DFSPH Simulation Step"),
        });
        
        // Helper to create bind group for DFSPH passes (needs all 6 bindings)
        let create_dfsph_bind_group = |pipeline: &wgpu::ComputePipeline| {
            self.context.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("DFSPH Bind Group"),
                layout: &pipeline.get_bind_group_layout(0),
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
                        resource: self.grid_cells_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.dfsph_data_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: self.velocity_corrections_buffer.as_entire_binding(),
                    },
                ],
            })
        };
        
        // Step 1: Compute Alpha (stiffness factors for pressure solve)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Alpha"),
                timestamp_writes: None,
            });
            let bind_group = create_dfsph_bind_group(&self.pipelines.dfsph_compute_alpha_pipeline);
            pass.set_pipeline(&self.pipelines.dfsph_compute_alpha_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(num_workgroups, 1, 1);
        }
        
        // Step 2: Divergence Solve (iterative correction for divergence-free velocity)
        for iter in 0..self.params.max_divergence_iter {
            // Compute divergence
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("Compute Divergence Iter {}", iter)),
                    timestamp_writes: None,
                });
                let bind_group = create_dfsph_bind_group(&self.pipelines.dfsph_compute_divergence_pipeline);
                pass.set_pipeline(&self.pipelines.dfsph_compute_divergence_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(num_workgroups, 1, 1);
            }
            
            // Correct divergence error (computes velocity_change)
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("Correct Divergence Iter {}", iter)),
                    timestamp_writes: None,
                });
                let bind_group = create_dfsph_bind_group(&self.pipelines.dfsph_correct_divergence_pipeline);
                pass.set_pipeline(&self.pipelines.dfsph_correct_divergence_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(num_workgroups, 1, 1);
            }
            
            // TODO: Check convergence and break early if error < max_divergence_error
            // For now, just do fixed iterations
        }
        
        // Step 2.5: Apply velocity corrections from divergence solve
        // This matches CPU update_velocities(): velocity += velocity_change
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Apply Velocity Corrections"),
                timestamp_writes: None,
            });
            let bind_group = self.context.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Apply Velocity Corrections Bind Group"),
                layout: &self.pipelines.apply_velocity_corrections_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.particles_buffer.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.velocity_corrections_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.params_buffer.as_entire_binding(),
                    },
                ],
            });
            pass.set_pipeline(&self.pipelines.apply_velocity_corrections_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(num_workgroups, 1, 1);
        }
        
        // Step 2.6: Clear velocity corrections for next iteration
        // This matches CPU: velocity_changes.fill(0.0)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Clear Velocity Corrections"),
                timestamp_writes: None,
            });
            let bind_group = self.context.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Clear Velocity Corrections Bind Group"),
                layout: &self.pipelines.clear_velocity_corrections_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.particles_buffer.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.velocity_corrections_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.params_buffer.as_entire_binding(),
                    },
                ],
            });
            pass.set_pipeline(&self.pipelines.clear_velocity_corrections_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(num_workgroups, 1, 1);
        }
        
        // Step 3: Apply non-pressure forces (viscosity, surface tension, gravity)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Apply Forces"),
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
        
        // Step 4: Integrate positions (semi-implicit Euler)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Integrate"),
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
        
        // Step 5: Pressure Solve (iterative correction for constant density)
        for iter in 0..self.params.max_pressure_iter {
            // Predict density after position update
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("Predict Density Iter {}", iter)),
                    timestamp_writes: None,
                });
                let bind_group = create_dfsph_bind_group(&self.pipelines.dfsph_predict_density_pipeline);
                pass.set_pipeline(&self.pipelines.dfsph_predict_density_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(num_workgroups, 1, 1);
            }
            
            // Correct density error
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("Correct Density Iter {}", iter)),
                    timestamp_writes: None,
                });
                let bind_group = create_dfsph_bind_group(&self.pipelines.dfsph_correct_density_pipeline);
                pass.set_pipeline(&self.pipelines.dfsph_correct_density_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(num_workgroups, 1, 1);
            }
            
            // TODO: Check convergence and break early if error < max_density_error
            // For now, just do fixed iterations
        }
        
        // Step 5.5: Apply position corrections from pressure solve
        // This matches CPU update_positions(): position += (velocity + velocity_change) * dt
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Apply Position Corrections"),
                timestamp_writes: None,
            });
            let bind_group = self.context.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Apply Position Corrections Bind Group"),
                layout: &self.pipelines.apply_position_corrections_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.particles_buffer.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.velocity_corrections_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.params_buffer.as_entire_binding(),
                    },
                ],
            });
            pass.set_pipeline(&self.pipelines.apply_position_corrections_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(num_workgroups, 1, 1);
        }
        
        // Submit all commands
        let _ = self.context.queue.submit(Some(encoder.finish()));
    }
    
    /// Read particle data back from GPU for comparison/debugging
    pub async fn read_particles(&self) -> Result<Vec<GpuParticle>, String> {
        // Ensure all GPU work is complete
        let _ = self.context.device.poll(wgpu::Maintain::Wait);
        
        // Create staging buffer
        let staging_buffer = self.context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: (self.num_particles * size_of::<GpuParticle>()) as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Copy from GPU to staging
        let mut encoder = self.context.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Readback"),
        });
        encoder.copy_buffer_to_buffer(
            self.particles_buffer.buffer(),
            0,
            &staging_buffer,
            0,
            (self.num_particles * size_of::<GpuParticle>()) as u64,
        );
        let _ = self.context.queue.submit(Some(encoder.finish()));
        
        // Map and read
        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        
        let _ = self.context.device.poll(wgpu::Maintain::Wait);
        rx.await.unwrap().map_err(|e| format!("Failed to map buffer: {:?}", e))?;
        
        let data = buffer_slice.get_mapped_range();
        let particles: Vec<GpuParticle> = bytemuck::cast_slice(&data).to_vec();
        
        drop(data);
        staging_buffer.unmap();
        
        Ok(particles)
    }
    
    // /// Reads particle data back from GPU (expensive - use sparingly!)
    // pub async fn read_particles(&self) -> Result<Vec<GpuParticle>, String> {
    //     use crate::gpu::buffer::ReadbackBuffer;
        
    //     let readback = ReadbackBuffer::new(
    //         &self.context.device,
    //         self.num_particles,
    //         "Particle Readback",
    //     );
        
    //     let mut encoder = self.context.device.create_command_encoder(&CommandEncoderDescriptor {
    //         label: Some("Readback Encoder"),
    //     });
        
    //     encoder.copy_buffer_to_buffer(
    //         self.particles_buffer.buffer(),
    //         0,
    //         readback.buffer(),
    //         0,
    //         (self.num_particles * size_of::<GpuParticle>()) as u64,
    //     );
        
    //     let _ = self.context.queue.submit(Some(encoder.finish()));
        
    //     readback.read().await
    // }
    
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
