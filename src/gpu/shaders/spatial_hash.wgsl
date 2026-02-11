// Spatial hashing compute shader for neighbor search

struct SimulationParams {
    dt: f32,
    inv_dt: f32,
    h: f32,
    density0: f32,
    gravity: vec4<f32>,
    particle_radius: f32,
    particle_mass: f32,
    num_particles: u32,
    grid_cell_size: f32,
    viscosity: f32,
    surface_tension: f32,
    pressure_stiffness: f32,
    _padding: f32,
}

struct Particle {
    position: vec4<f32>,
    velocity: vec4<f32>,
    force: vec4<f32>,
    density_pressure_mass: vec4<f32>,
}

struct ParticleIndex {
    index: u32,
    hash: u32,
}

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> particle_indices: array<ParticleIndex>;
@group(0) @binding(2) var<uniform> params: SimulationParams;

// Grid dimensions (should be large enough to cover simulation space)
const GRID_SIZE: vec3<i32> = vec3<i32>(256, 256, 256);

// Compute spatial hash for a position
fn spatial_hash(pos: vec3<f32>) -> u32 {
    let grid_pos = vec3<i32>(floor(pos / params.grid_cell_size));
    
    // Wrap to grid bounds
    let wrapped = vec3<u32>(
        u32(grid_pos.x) % u32(GRID_SIZE.x),
        u32(grid_pos.y) % u32(GRID_SIZE.y),
        u32(grid_pos.z) % u32(GRID_SIZE.z)
    );
    
    // Simple hash function
    let p1: u32 = 73856093u;
    let p2: u32 = 19349663u;
    let p3: u32 = 83492791u;
    
    return (wrapped.x * p1) ^ (wrapped.y * p2) ^ (wrapped.z * p3);
}

@compute @workgroup_size(256)
fn compute_hashes(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.num_particles) {
        return;
    }
    
    let particle = particles[idx];
    let hash = spatial_hash(particle.position.xyz);
    
    particle_indices[idx].index = idx;
    particle_indices[idx].hash = hash;
}
