// Apply velocity and position corrections from DFSPH solver
// Matches update_velocities() and update_positions() in CPU DFSPH solver

struct SimulationParams {
    // MUST match exact layout in particle_data.rs
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
    
    grid_size_x: u32,
    grid_size_y: u32,
    grid_size_z: u32,
    
    max_density_error: f32,
    max_divergence_error: f32,
    max_pressure_iter: u32,
    max_divergence_iter: u32,
    _padding0: u32,
    _padding1: u32,
};

struct Particle {
    position: vec4<f32>,         // xyz = position, w = unused
    velocity: vec4<f32>,         // xyz = velocity, w = unused
    acceleration: vec4<f32>,     // xyz = acceleration, w = unused
    density_pressure_mass: vec4<f32>, // x = density, y = pressure, z = mass, w = unused
    velocity_change: vec4<f32>,  // xyz = velocity correction from DFSPH, w = unused
};

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<storage, read> velocity_corrections: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: SimulationParams;

// Apply velocity corrections from divergence/pressure solve
// CPU equivalent: velocity += velocity_change
@compute @workgroup_size(256)
fn apply_velocity_corrections(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.num_particles) {
        return;
    }
    
    var particle = particles[idx];
    let correction = velocity_corrections[idx].xyz;
    
    // Apply correction: v += dv
    particle.velocity = vec4<f32>(particle.velocity.xyz + correction, 0.0);
    
    // Store velocity change for later use in update_positions
    particle.velocity_change = vec4<f32>(correction, 0.0);
    
    particles[idx] = particle;
}

// Apply position corrections after pressure solve
// CPU equivalent: position += (velocity + velocity_change) * dt
@compute @workgroup_size(256)
fn apply_position_corrections(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.num_particles) {
        return;
    }
    
    var particle = particles[idx];
    let vel = particle.velocity.xyz;
    let delta_v = particle.velocity_change.xyz;
    
    // Apply position correction: pos += (v + dv) * dt
    particle.position = vec4<f32>(particle.position.xyz + (vel + delta_v) * params.dt, 0.0);
    
    particles[idx] = particle;
}

// Clear velocity corrections (called after applying them)
// CPU equivalent: velocity_changes.fill(0.0)
@compute @workgroup_size(256)
fn clear_velocity_corrections(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.num_particles) {
        return;
    }
    
    var particle = particles[idx];
    particle.velocity_change = vec4<f32>(0.0);
    particles[idx] = particle;
}
