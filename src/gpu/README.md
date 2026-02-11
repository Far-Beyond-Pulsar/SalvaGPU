# GPU Acceleration for Salva

This document describes the GPU acceleration features added to Salva using WGPU.

## Overview

The GPU acceleration module enables running the entire fluid simulation on the GPU, keeping all particle data GPU-resident to minimize expensive CPU-GPU transfers. A delta-based synchronization system efficiently updates only changed data.

## Features

- **Full GPU-resident simulation**: All particle positions, velocities, forces, and densities stay on the GPU
- **Delta synchronization**: Only modified data is transferred between CPU and GPU
- **Compute shader pipeline**:
  - Spatial hashing for neighbor search
  - SPH density computation with cubic spline kernel
  - Pressure forces (Tait equation of state)
  - Viscosity forces
  - Surface tension
  - Semi-implicit Euler integration
  - Boundary collision handling

## Building with GPU Support

To enable GPU acceleration, build with the `gpu-acceleration` feature:

```bash
# For 2D
cargo build --package salva2d --features gpu-acceleration

# For 3D
cargo build --package salva3d --features gpu-acceleration
```

## Usage

### Basic Example

```rust
use salva3d::gpu::GpuFluidSolver;
use salva3d::math::Vector;
use salva3d::object::Fluid;

#[tokio::main]
async fn main() -> Result<(), String> {
    // Create GPU solver with initial capacity
    let mut gpu_solver = GpuFluidSolver::new(10000).await?;
    
    // Create fluids on CPU
    let fluid = Fluid::new(positions, 0.025, 1.0);
    let fluids = vec![fluid];
    
    // Initialize GPU with fluid data
    gpu_solver.init_from_fluids(&fluids);
    
    // Run simulation steps on GPU
    let dt = 0.016;
    let gravity = Vector::new(0.0, -9.81, 0.0);
    
    for _ in 0..1000 {
        gpu_solver.step(dt, &gravity);
    }
    
    // Optionally read data back from GPU (expensive!)
    let particles = gpu_solver.read_particles().await?;
    
    Ok(())
}
```

### Performance Benchmark

Run the included benchmark to compare GPU vs CPU performance:

```bash
# 2D benchmark
cd examples2d
cargo run --bin gpu_benchmark --package examples3d --release

# 3D benchmark
cd examples3d
cargo run --bin gpu_benchmark --package examples3d --release
```

The benchmark tests multiple particle counts (100, 500, 1000, 2500, 5000, 10000) and runs each configuration 10 times for 100 frames.

## Architecture

### GPU Module Structure

```
src/gpu/
├── mod.rs              # Module exports
├── context.rs          # WGPU device/queue management
├── buffer.rs           # Delta-based buffer manager
├── particle_data.rs    # GPU-compatible data structures
├── pipeline.rs         # Compute pipeline management
├── solver.rs           # Main GPU solver
└── shaders/
    ├── spatial_hash.wgsl   # Neighbor search
    ├── density.wgsl        # Density computation
    ├── forces.wgsl         # Force calculation
    └── integrate.wgsl      # Time integration
```

### Delta Synchronization System

The `DeltaBufferManager<T>` tracks which particles have been modified on the CPU side and only uploads those changes to the GPU:

```rust
// Update single particle
buffer.update(particle_index, new_value);

// Update range
buffer.update_range(start_index, &values);

// Sync to GPU (only uploads modified data)
buffer.sync_to_gpu(&queue);
```

### Simulation Pipeline

Each simulation step executes the following GPU compute passes:

1. **Spatial Hash**: Compute grid hash for each particle
2. **Density**: Calculate density and pressure using SPH kernel
3. **Forces**: Compute pressure, viscosity, and surface tension forces
4. **Integration**: Update velocities and positions

All passes run entirely on the GPU with no CPU synchronization required between steps.

## Performance Considerations

### When to Use GPU Acceleration

GPU acceleration provides benefits when:
- Particle count > 1000 (overhead of GPU dispatch becomes worthwhile)
- Running many simulation steps without needing particle data on CPU
- Rendering directly from GPU buffers (future feature)

### When to Use CPU

CPU implementation may be faster when:
- Particle count < 1000
- Frequent readback to CPU is required
- Integration with CPU-based physics engines

### Optimization Tips

1. **Minimize CPU-GPU transfers**: Keep data on GPU as long as possible
2. **Batch updates**: Use `update_range()` instead of multiple `update()` calls
3. **Avoid readback**: Only call `read_particles()` when absolutely necessary
4. **Adjust workgroup size**: Shader workgroup size (currently 256) can be tuned for different GPUs

## GPU Requirements

- **Vulkan**, **Metal**, **DirectX 12**, or **WebGPU** support
- Compute shader support
- Minimum 256MB VRAM recommended for 10,000 particles

## Limitations

Current implementation limitations:

1. **Simplified neighbor search**: Currently checks all particles (O(n²)). Production version should use sorted spatial hash grid.
2. **Fixed boundaries**: Box boundaries are hardcoded in shader. Flexible boundary system needed.
3. **No boundary coupling**: Rigid body boundaries not yet implemented.
4. **Single fluid**: Multiple fluids with different properties not yet supported.
5. **No DFSPH iterations**: Simplified pressure solver (no density error correction iterations).

## Future Improvements

- [ ] Proper spatial hash grid with sorting
- [ ] Multiple fluid support
- [ ] Boundary shape integration
- [ ] DFSPH divergence-free solver
- [ ] Direct rendering from GPU buffers
- [ ] Async compute optimization
- [ ] Double buffering for continuous simulation

## WebGPU / WASM Support

The GPU acceleration module is compatible with WebGPU, allowing GPU-accelerated fluid simulation in browsers:

```bash
# Build for WASM
cargo build --target wasm32-unknown-unknown --features gpu-acceleration
```

## Dependencies

- `wgpu` ^23.0 - Cross-platform GPU API
- `bytemuck` ^1.14 - Safe type casting for GPU buffers
- `encase` ^0.10 - Shader type layout derivation
- `futures` ^0.3 - Async GPU operations

## Troubleshooting

### "No suitable GPU adapter found"

Ensure your system has:
- Updated GPU drivers
- Vulkan/Metal/DX12 runtime support
- For Linux: Install `vulkan-loader` package

### Slow performance on integrated GPU

Set power preference:
```rust
GpuContext::with_power_preference(PowerPreference::HighPerformance).await?
```

### Compilation errors about shader features

Ensure you're building with the `gpu-acceleration` feature flag enabled.

## Contributing

Contributions to improve GPU performance and features are welcome! Areas of interest:

- Optimized spatial data structures
- Advanced SPH methods (WCSPH, PCISPH, PBF)
- Multi-GPU support
- Vendor-specific optimizations

## License

Same as Salva: Apache-2.0
