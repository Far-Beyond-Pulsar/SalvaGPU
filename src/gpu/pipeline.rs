//! GPU compute pipeline management

use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, ComputePipeline,
    ComputePipelineDescriptor, Device, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages, util::DeviceExt,
};

/// Manages all compute pipelines for the simulation
pub struct GpuPipelines {
    pub spatial_hash_pipeline: ComputePipeline,
    pub sort_pipeline: ComputePipeline,
    pub dfsph_compute_alpha_pipeline: ComputePipeline,
    pub dfsph_predict_density_pipeline: ComputePipeline,
    pub dfsph_correct_density_pipeline: ComputePipeline,
    pub dfsph_compute_divergence_pipeline: ComputePipeline,
    pub dfsph_correct_divergence_pipeline: ComputePipeline,
    pub forces_pipeline: ComputePipeline,
    pub integrate_pipeline: ComputePipeline,
}

impl GpuPipelines {
    /// Creates all compute pipelines
    pub fn new(device: &Device) -> Self {
        let spatial_hash_pipeline = Self::create_spatial_hash_pipeline(device);
        let sort_pipeline = Self::create_sort_pipeline(device);
        
        // DFSPH pipelines - matching CPU implementation
        let dfsph_shader = include_str!("shaders/dfsph.wgsl");
        let dfsph_compute_alpha_pipeline = Self::create_dfsph_pipeline(device, dfsph_shader, "compute_alpha", "DFSPH Compute Alpha");
        let dfsph_predict_density_pipeline = Self::create_dfsph_pipeline(device, dfsph_shader, "predict_density", "DFSPH Predict Density");
        let dfsph_correct_density_pipeline = Self::create_dfsph_pipeline(device, dfsph_shader, "correct_density_error", "DFSPH Correct Density");
        let dfsph_compute_divergence_pipeline = Self::create_dfsph_pipeline(device, dfsph_shader, "compute_divergence", "DFSPH Compute Divergence");
        let dfsph_correct_divergence_pipeline = Self::create_dfsph_pipeline(device, dfsph_shader, "correct_divergence_error", "DFSPH Correct Divergence");
        
        let forces_pipeline = Self::create_forces_pipeline(device);
        let integrate_pipeline = Self::create_integrate_pipeline(device);

        Self {
            spatial_hash_pipeline,
            sort_pipeline,
            dfsph_compute_alpha_pipeline,
            dfsph_predict_density_pipeline,
            dfsph_correct_density_pipeline,
            dfsph_compute_divergence_pipeline,
            dfsph_correct_divergence_pipeline,
            forces_pipeline,
            integrate_pipeline,
        }
    }

    fn create_spatial_hash_pipeline(device: &Device) -> ComputePipeline {
        let shader_source = include_str!("shaders/spatial_hash.wgsl");
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Spatial Hash Shader"),
            source: ShaderSource::Wgsl(shader_source.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Spatial Hash Bind Group Layout"),
            entries: &[
                // Particles buffer (read)
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Particle indices buffer (read_write)
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Simulation params (uniform)
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Spatial Hash Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Spatial Hash Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("compute_hashes"),
            compilation_options: Default::default(),
            cache: None,
        })
    }

    fn create_density_pipeline(device: &Device) -> ComputePipeline {
        let shader_source = include_str!("shaders/density.wgsl");
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Density Shader"),
            source: ShaderSource::Wgsl(shader_source.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Density Bind Group Layout"),
            entries: &[
                // Particles buffer (read_write)
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Particle indices buffer (read)
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Grid cells buffer (read)
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Simulation params (uniform)
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Density Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Density Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("compute_density"),
            compilation_options: Default::default(),
            cache: None,
        })
    }

    fn create_sort_pipeline(device: &Device) -> ComputePipeline {
        let shader_source = include_str!("shaders/sort.wgsl");
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Sort Shader"),
            source: ShaderSource::Wgsl(shader_source.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Sort Bind Group Layout"),
            entries: &[
                // Particle indices buffer (read_write)
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Simulation params (uniform)
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Sort Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Sort Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("bitonic_sort_step"),
            compilation_options: Default::default(),
            cache: None,
        })
    }

    fn create_dfsph_pipeline(device: &Device, shader_source: &str, entry_point: &str, label: &str) -> ComputePipeline {
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some(label),
            source: ShaderSource::Wgsl(shader_source.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some(&format!("{} Bind Group Layout", label)),
            entries: &[
                // Particles buffer (read_write) - binding 0
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Particle indices buffer (read) - binding 1
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Grid cells buffer (read) - binding 2
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Simulation params (uniform) - binding 3
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // DFSPH data buffer (read_write) - binding 4
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Velocity corrections buffer (read_write) - binding 5
                BindGroupLayoutEntry {
                    binding: 5,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some(&format!("{} Pipeline Layout", label)),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some(entry_point),
            compilation_options: Default::default(),
            cache: None,
        })
    }

    fn create_forces_pipeline(device: &Device) -> ComputePipeline {
        let shader_source = include_str!("shaders/forces.wgsl");
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Forces Shader"),
            source: ShaderSource::Wgsl(shader_source.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Forces Bind Group Layout"),
            entries: &[
                // Particles buffer (read_write)
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Particle indices buffer (read)
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Simulation params (uniform)
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Forces Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Forces Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("compute_forces"),
            compilation_options: Default::default(),
            cache: None,
        })
    }

    fn create_integrate_pipeline(device: &Device) -> ComputePipeline {
        let shader_source = include_str!("shaders/integrate.wgsl");
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Integration Shader"),
            source: ShaderSource::Wgsl(shader_source.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Integration Bind Group Layout"),
            entries: &[
                // Particles buffer (read_write)
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Simulation params (uniform)
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Integration Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Integration Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("integrate"),
            compilation_options: Default::default(),
            cache: None,
        })
    }
}
