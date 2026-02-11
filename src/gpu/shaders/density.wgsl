// Density computation using SPH kernel

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

struct GridCell {
    start: u32,
    count: u32,
}

struct ParticleIndex {
    index: u32,
    hash: u32,
}

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<storage, read> particle_indices: array<ParticleIndex>;
@group(0) @binding(2) var<storage, read> grid_cells: array<GridCell>;
@group(0) @binding(3) var<uniform> params: SimulationParams;

// Cubic spline kernel (Monaghan 1992)
fn cubic_spline_kernel(r: f32, h: f32) -> f32 {
    let q = r / h;
    let sigma = 1.0 / (3.141592653589793 * h * h * h); // 3D normalization
    
    if (q >= 2.0) {
        return 0.0;
    } else if (q >= 1.0) {
        let term = 2.0 - q;
        return sigma * 0.25 * term * term * term;
    } else {
        let q2 = q * q;
        let q3 = q2 * q;
        return sigma * (1.0 - 1.5 * q2 + 0.75 * q3);
    }
}

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

// Get neighboring cells for a particle
fn get_neighbor_cells(hash: u32) -> array<u32, 27> {
    var neighbors: array<u32, 27>;
    // For simplicity, we check the same cell and adjacent cells
    // In a full implementation, this would compute actual neighboring cell hashes
    neighbors[0] = hash;
    // Fill rest with invalid indices
    for (var i = 1u; i < 27u; i++) {
        neighbors[i] = 0xFFFFFFFFu;
    }
    return neighbors;
}

@compute @workgroup_size(256)
fn compute_density(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.num_particles) {
        return;
    }
    
    var particle_i = particles[idx];
    let pos_i = particle_i.position.xyz;
    var density = 0.0;
    
    // Find particle's hash
    var my_hash = 0u;
    for (var i = 0u; i < params.num_particles; i++) {
        if (particle_indices[i].index == idx) {
            my_hash = particle_indices[i].hash;
            break;
        }
    }
    
    // Get neighboring cells
    let neighbor_cells = get_neighbor_cells(my_hash);
    
    // Iterate through neighbor cells
    for (var cell_idx = 0u; cell_idx < 27u; cell_idx++) {
        let cell_hash = neighbor_cells[cell_idx];
        if (cell_hash == 0xFFFFFFFFu) {
            continue;
        }
        
        // Find cell in sorted list
        for (var i = 0u; i < params.num_particles; i++) {
            if (particle_indices[i].hash == cell_hash) {
                let j = particle_indices[i].index;
                let particle_j = particles[j];
                let pos_j = particle_j.position.xyz;
                
                let r_vec = pos_i - pos_j;
                let r = length(r_vec);
                
                if (r < 2.0 * params.h) {
                    let mass_j = particle_j.density_pressure_mass.z;
                    density += mass_j * cubic_spline_kernel(r, params.h);
                }
            }
        }
    }
    
    // Update density
    particle_i.density_pressure_mass.x = density;
    
    // Compute pressure using equation of state (Tait equation)
    let gamma = 7.0;
    let B = params.pressure_stiffness;
    let rho_ratio = density / params.density0;
    particle_i.density_pressure_mass.y = B * (pow(rho_ratio, gamma) - 1.0);
    
    particles[idx] = particle_i;
}
