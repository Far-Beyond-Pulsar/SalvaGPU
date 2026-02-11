// Pressure, viscosity, and force computation shader

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

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<storage, read> particle_indices: array<ParticleIndex>;
@group(0) @binding(2) var<uniform> params: SimulationParams;

// Gradient of cubic spline kernel
fn cubic_spline_gradient(r_vec: vec3<f32>, h: f32) -> vec3<f32> {
    let r = length(r_vec);
    
    if (r < 1e-6 || r >= 2.0 * h) {
        return vec3<f32>(0.0);
    }
    
    let q = r / h;
    let sigma = 1.0 / (3.141592653589793 * h * h * h);
    
    var grad_kernel: f32;
    if (q >= 1.0) {
        let term = 2.0 - q;
        grad_kernel = -0.75 * sigma * term * term / h;
    } else {
        grad_kernel = sigma * (-3.0 * q + 2.25 * q * q) / h;
    }
    
    return grad_kernel * (r_vec / r);
}

// Viscosity kernel (Monaghan 1992)
fn viscosity_kernel(r: f32, h: f32) -> f32 {
    let q = r / h;
    let sigma = 15.0 / (2.0 * 3.141592653589793 * h * h * h);
    
    if (q >= 2.0) {
        return 0.0;
    } else if (q >= 1.0) {
        let term = 2.0 - q;
        return sigma * term * term * term / 3.0;
    } else {
        let q2 = q * q;
        let q3 = q2 * q;
        return sigma * (2.0/3.0 - q2 + 0.5 * q3);
    }
}

// Get neighboring cells (simplified - checks all particles for now)
fn find_neighbors(pos: vec3<f32>, h: f32) -> array<u32, 64> {
    var neighbors: array<u32, 64>;
    var count = 0u;
    
    for (var i = 0u; i < params.num_particles && count < 64u; i++) {
        let neighbor_pos = particles[i].position.xyz;
        let r = length(pos - neighbor_pos);
        if (r < 2.0 * h && r > 1e-6) {
            neighbors[count] = i;
            count++;
        }
    }
    
    // Fill rest with invalid index
    for (var i = count; i < 64u; i++) {
        neighbors[i] = 0xFFFFFFFFu;
    }
    
    return neighbors;
}

@compute @workgroup_size(256)
fn compute_forces(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.num_particles) {
        return;
    }
    
    var particle_i = particles[idx];
    let pos_i = particle_i.position.xyz;
    let vel_i = particle_i.velocity.xyz;
    let density_i = particle_i.density_pressure_mass.x;
    let pressure_i = particle_i.density_pressure_mass.y;
    let mass_i = particle_i.density_pressure_mass.z;
    
    var force_pressure = vec3<f32>(0.0);
    var force_viscosity = vec3<f32>(0.0);
    var force_surface = vec3<f32>(0.0);
    
    // Find neighbors
    let neighbors = find_neighbors(pos_i, params.h);
    
    // Compute forces from neighbors
    for (var n = 0u; n < 64u; n++) {
        let j = neighbors[n];
        if (j == 0xFFFFFFFFu) {
            break;
        }
        
        let particle_j = particles[j];
        let pos_j = particle_j.position.xyz;
        let vel_j = particle_j.velocity.xyz;
        let density_j = particle_j.density_pressure_mass.x;
        let pressure_j = particle_j.density_pressure_mass.y;
        let mass_j = particle_j.density_pressure_mass.z;
        
        let r_vec = pos_i - pos_j;
        let r = length(r_vec);
        
        if (r < 2.0 * params.h && r > 1e-6) {
            let grad = cubic_spline_gradient(r_vec, params.h);
            
            // Pressure force (symmetric SPH formulation)
            let pressure_term = (pressure_i / (density_i * density_i)) + 
                              (pressure_j / (density_j * density_j));
            force_pressure -= mass_j * pressure_term * grad;
            
            // Viscosity force (Monaghan 1992)
            let vel_diff = vel_j - vel_i;
            let visc_kernel = viscosity_kernel(r, params.h);
            force_viscosity += params.viscosity * mass_j * vel_diff * visc_kernel / density_j;
            
            // Simple surface tension (cohesion)
            force_surface -= params.surface_tension * mass_j * r_vec / r;
        }
    }
    
    // Total force
    let total_force = force_pressure + force_viscosity + force_surface + 
                     params.gravity.xyz * mass_i;
    
    particle_i.force = vec4<f32>(total_force, 0.0);
    particles[idx] = particle_i;
}
