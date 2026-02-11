// DFSPH integration shader
// Matches integrate_and_clear_accelerations() in CPU
// This accumulates acceleration into velocity_change (NOT velocity)
// Position update happens AFTER pressure solve

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
    
    grid_size_x: u32,
    grid_size_y: u32,
    grid_size_z: u32,
    
    max_density_error: f32,
    max_divergence_error: f32,
    max_pressure_iter: u32,
    max_divergence_iter: u32,
    _padding0: u32,
    _padding1: u32,
}

struct Particle {
    position: vec4<f32>,         // xyz = position, w = unused
    velocity: vec4<f32>,         // xyz = velocity, w = unused
    force: vec4<f32>,            // xyz = force/acceleration, w = unused
    density_pressure_mass: vec4<f32>, // x = density, y = pressure, z = mass, w = unused
    velocity_change: vec4<f32>,  // xyz = velocity correction from DFSPH, w = unused
};

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: SimulationParams;

// CPU equivalent: integrate_and_clear_accelerations
// velocity_change += acceleration * dt
// acceleration = 0
@compute @workgroup_size(256)
fn integrate(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.num_particles) {
        return;
    }
    
    var particle = particles[idx];
    let accel = particle.force.xyz;  // force field contains acceleration
    let vel_change = particle.velocity_change.xyz;
    
    // Accumulate acceleration into velocity_change
    particle.velocity_change = vec4<f32>(vel_change + accel * params.dt, 0.0);
    
    // Clear acceleration for next iteration
    particle.force = vec4<f32>(0.0);
    
    particles[idx] = particle;
}

