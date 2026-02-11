// Particle integration shader (velocity and position updates)

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

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: SimulationParams;

// Boundary collision constants (simple box boundary for now)
const BOUNDARY_MIN: vec3<f32> = vec3<f32>(-5.0, -5.0, -5.0);
const BOUNDARY_MAX: vec3<f32> = vec3<f32>(5.0, 5.0, 5.0);
const DAMPING: f32 = 0.5; // Velocity damping on collision

@compute @workgroup_size(256)
fn integrate(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.num_particles) {
        return;
    }
    
    var particle = particles[idx];
    let mass = particle.density_pressure_mass.z;
    
    // Semi-implicit Euler integration
    // v_{n+1} = v_n + a * dt
    let acceleration = particle.force.xyz / mass;
    var new_velocity = particle.velocity.xyz + acceleration * params.dt;
    
    // x_{n+1} = x_n + v_{n+1} * dt
    var new_position = particle.position.xyz + new_velocity * params.dt;
    
    // Simple boundary collision
    // X boundary
    if (new_position.x < BOUNDARY_MIN.x) {
        new_position.x = BOUNDARY_MIN.x;
        new_velocity.x = -new_velocity.x * DAMPING;
    } else if (new_position.x > BOUNDARY_MAX.x) {
        new_position.x = BOUNDARY_MAX.x;
        new_velocity.x = -new_velocity.x * DAMPING;
    }
    
    // Y boundary
    if (new_position.y < BOUNDARY_MIN.y) {
        new_position.y = BOUNDARY_MIN.y;
        new_velocity.y = -new_velocity.y * DAMPING;
    } else if (new_position.y > BOUNDARY_MAX.y) {
        new_position.y = BOUNDARY_MAX.y;
        new_velocity.y = -new_velocity.y * DAMPING;
    }
    
    // Z boundary
    if (new_position.z < BOUNDARY_MIN.z) {
        new_position.z = BOUNDARY_MIN.z;
        new_velocity.z = -new_velocity.z * DAMPING;
    } else if (new_position.z > BOUNDARY_MAX.z) {
        new_position.z = BOUNDARY_MAX.z;
        new_velocity.z = -new_velocity.z * DAMPING;
    }
    
    // Update particle
    particle.position = vec4<f32>(new_position, 1.0);
    particle.velocity = vec4<f32>(new_velocity, 0.0);
    
    // Clear forces for next iteration
    particle.force = vec4<f32>(0.0);
    
    particles[idx] = particle;
}
