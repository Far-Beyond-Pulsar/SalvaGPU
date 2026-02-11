// Full DFSPH (Divergence-Free SPH) solver with density error correction
// Based on Bender & Koschier 2015

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
    _padding: u32,
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

struct DFSPHData {
    alpha: f32,           // Stiffness factor
    predicted_density: f32,
    divergence: f32,
    _padding: f32,
}

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<storage, read> particle_indices: array<ParticleIndex>;
@group(0) @binding(2) var<storage, read> grid_cells: array<GridCell>;
@group(0) @binding(3) var<uniform> params: SimulationParams;
@group(0) @binding(4) var<storage, read_write> dfsph_data: array<DFSPHData>;
@group(0) @binding(5) var<storage, read_write> velocity_corrections: array<vec4<f32>>;

fn cubic_spline_gradient(r_vec: vec3<f32>, h: f32) -> vec3<f32> {
    let r = length(r_vec);
    
    if (r < 1e-6 || r >= 2.0 * h) {
        return vec3<f32>(0.0);
    }
    
    let q = r / h;
    
    #ifdef DIM_2D
    let sigma = 10.0 / (7.0 * 3.141592653589793 * h * h);
    #else
    let sigma = 1.0 / (3.141592653589793 * h * h * h);
    #endif
    
    var grad_kernel: f32;
    if (q >= 1.0) {
        let term = 2.0 - q;
        grad_kernel = -0.75 * sigma * term * term / h;
    } else {
        grad_kernel = sigma * (-3.0 * q + 2.25 * q * q) / h;
    }
    
    return grad_kernel * (r_vec / r);
}

fn get_neighbor_hash(base_pos: vec3<f32>, offset: vec3<i32>) -> u32 {
    let grid_pos = vec3<i32>(floor(base_pos / params.grid_cell_size));
    let neighbor_pos = grid_pos + offset;
    
    let wrapped = vec3<u32>(
        u32(neighbor_pos.x) & (params.grid_size_x - 1u),
        u32(neighbor_pos.y) & (params.grid_size_y - 1u),
        u32(neighbor_pos.z) & (params.grid_size_z - 1u)
    );
    
    let p1: u32 = 73856093u;
    let p2: u32 = 19349663u;
    let p3: u32 = 83492791u;
    
    return (wrapped.x * p1) ^ (wrapped.y * p2) ^ (wrapped.z * p3);
}

// Compute alpha (stiffness) factor for each particle
@compute @workgroup_size(256)
fn compute_alpha(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.num_particles) {
        return;
    }
    
    let particle_i = particles[idx];
    let pos_i = particle_i.position.xyz;
    let density_i = particle_i.density_pressure_mass.x;
    
    var grad_sum = vec3<f32>(0.0);
    var squared_grad_sum = 0.0;
    
    // Check neighboring cells
    for (var dx = -1; dx <= 1; dx++) {
        for (var dy = -1; dy <= 1; dy++) {
            for (var dz = -1; dz <= 1; dz++) {
                let neighbor_hash = get_neighbor_hash(pos_i, vec3<i32>(dx, dy, dz));
                let grid_size = params.grid_size_x * params.grid_size_y * params.grid_size_z;
                let cell_idx = neighbor_hash % grid_size;
                
                let cell = grid_cells[cell_idx];
                
                for (var i = 0u; i < cell.count; i++) {
                    let sorted_idx = cell.start + i;
                    if (sorted_idx >= params.num_particles) {
                        break;
                    }
                    
                    let j = particle_indices[sorted_idx].index;
                    if (j == idx) {
                        continue;
                    }
                    
                    let particle_j = particles[j];
                    let pos_j = particle_j.position.xyz;
                    let mass_j = particle_j.density_pressure_mass.z;
                    
                    let r_vec = pos_i - pos_j;
                    let r = length(r_vec);
                    
                    if (r < 2.0 * params.h) {
                        let grad = cubic_spline_gradient(r_vec, params.h);
                        let grad_scaled = grad * mass_j / density_i;
                        
                        squared_grad_sum += dot(grad_scaled, grad_scaled);
                        grad_sum += grad_scaled;
                    }
                }
            }
        }
    }
    
    let denominator = squared_grad_sum + dot(grad_sum, grad_sum);
    
    var alpha = 0.0;
    if (denominator > 1e-6) {
        alpha = 1.0 / denominator;
    }
    
    dfsph_data[idx].alpha = alpha;
}

// Predict densities after velocity update
@compute @workgroup_size(256)
fn predict_density(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.num_particles) {
        return;
    }
    
    let particle_i = particles[idx];
    let pos_i = particle_i.position.xyz;
    let vel_i = particle_i.velocity.xyz;
    let density_i = particle_i.density_pressure_mass.x;
    
    var density_change = 0.0;
    
    // Check neighboring cells
    for (var dx = -1; dx <= 1; dx++) {
        for (var dy = -1; dy <= 1; dy++) {
            for (var dz = -1; dz <= 1; dz++) {
                let neighbor_hash = get_neighbor_hash(pos_i, vec3<i32>(dx, dy, dz));
                let grid_size = params.grid_size_x * params.grid_size_y * params.grid_size_z;
                let cell_idx = neighbor_hash % grid_size;
                
                let cell = grid_cells[cell_idx];
                
                for (var i = 0u; i < cell.count; i++) {
                    let sorted_idx = cell.start + i;
                    if (sorted_idx >= params.num_particles) {
                        break;
                    }
                    
                    let j = particle_indices[sorted_idx].index;
                    let particle_j = particles[j];
                    let pos_j = particle_j.position.xyz;
                    let vel_j = particle_j.velocity.xyz;
                    let mass_j = particle_j.density_pressure_mass.z;
                    
                    let r_vec = pos_i - pos_j;
                    let r = length(r_vec);
                    
                    if (r < 2.0 * params.h) {
                        let vel_diff = vel_i - vel_j;
                        let grad = cubic_spline_gradient(r_vec, params.h);
                        density_change += mass_j * dot(vel_diff, grad);
                    }
                }
            }
        }
    }
    
    let predicted_density = density_i + params.dt * density_change;
    dfsph_data[idx].predicted_density = predicted_density;
}

// Correct density error (pressure solve iteration)
@compute @workgroup_size(256)
fn correct_density_error(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.num_particles) {
        return;
    }
    
    let particle_i = particles[idx];
    let pos_i = particle_i.position.xyz;
    let alpha_i = dfsph_data[idx].alpha;
    let predicted_density_i = dfsph_data[idx].predicted_density;
    let density_error = max(predicted_density_i - params.density0, 0.0);
    
    let k_i = alpha_i * density_error;
    var velocity_correction = vec3<f32>(0.0);
    
    // Check neighboring cells
    for (var dx = -1; dx <= 1; dx++) {
        for (var dy = -1; dy <= 1; dy++) {
            for (var dz = -1; dz <= 1; dz++) {
                let neighbor_hash = get_neighbor_hash(pos_i, vec3<i32>(dx, dy, dz));
                let grid_size = params.grid_size_x * params.grid_size_y * params.grid_size_z;
                let cell_idx = neighbor_hash % grid_size;
                
                let cell = grid_cells[cell_idx];
                
                for (var i = 0u; i < cell.count; i++) {
                    let sorted_idx = cell.start + i;
                    if (sorted_idx >= params.num_particles) {
                        break;
                    }
                    
                    let j = particle_indices[sorted_idx].index;
                    let particle_j = particles[j];
                    let pos_j = particle_j.position.xyz;
                    let mass_j = particle_j.density_pressure_mass.z;
                    let alpha_j = dfsph_data[j].alpha;
                    let predicted_density_j = dfsph_data[j].predicted_density;
                    let density_error_j = max(predicted_density_j - params.density0, 0.0);
                    
                    let k_j = alpha_j * density_error_j;
                    let k_ij = k_i + k_j;
                    
                    let r_vec = pos_i - pos_j;
                    let r = length(r_vec);
                    
                    if (r < 2.0 * params.h && k_ij > 0.0) {
                        let grad = cubic_spline_gradient(r_vec, params.h);
                        velocity_correction -= k_ij * mass_j * grad * params.inv_dt;
                    }
                }
            }
        }
    }
    
    velocity_corrections[idx] = vec4<f32>(velocity_correction, 0.0);
}

// Compute divergence
@compute @workgroup_size(256)
fn compute_divergence(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.num_particles) {
        return;
    }
    
    let particle_i = particles[idx];
    let pos_i = particle_i.position.xyz;
    let vel_i = particle_i.velocity.xyz + velocity_corrections[idx].xyz;
    
    var divergence = 0.0;
    
    // Check neighboring cells
    for (var dx = -1; dx <= 1; dx++) {
        for (var dy = -1; dy <= 1; dy++) {
            for (var dz = -1; dz <= 1; dz++) {
                let neighbor_hash = get_neighbor_hash(pos_i, vec3<i32>(dx, dy, dz));
                let grid_size = params.grid_size_x * params.grid_size_y * params.grid_size_z;
                let cell_idx = neighbor_hash % grid_size;
                
                let cell = grid_cells[cell_idx];
                
                for (var i = 0u; i < cell.count; i++) {
                    let sorted_idx = cell.start + i;
                    if (sorted_idx >= params.num_particles) {
                        break;
                    }
                    
                    let j = particle_indices[sorted_idx].index;
                    let particle_j = particles[j];
                    let pos_j = particle_j.position.xyz;
                    let vel_j = particle_j.velocity.xyz + velocity_corrections[j].xyz;
                    let mass_j = particle_j.density_pressure_mass.z;
                    
                    let r_vec = pos_i - pos_j;
                    let r = length(r_vec);
                    
                    if (r < 2.0 * params.h) {
                        let vel_diff = vel_i - vel_j;
                        let grad = cubic_spline_gradient(r_vec, params.h);
                        divergence += mass_j * dot(vel_diff, grad);
                    }
                }
            }
        }
    }
    
    dfsph_data[idx].divergence = divergence;
}

// Correct divergence error
@compute @workgroup_size(256)
fn correct_divergence_error(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.num_particles) {
        return;
    }
    
    let particle_i = particles[idx];
    let pos_i = particle_i.position.xyz;
    let alpha_i = dfsph_data[idx].alpha;
    let divergence_i = dfsph_data[idx].divergence;
    
    let k_i = alpha_i * divergence_i;
    var velocity_correction = vec3<f32>(0.0);
    
    // Check neighboring cells
    for (var dx = -1; dx <= 1; dx++) {
        for (var dy = -1; dy <= 1; dy++) {
            for (var dz = -1; dz <= 1; dz++) {
                let neighbor_hash = get_neighbor_hash(pos_i, vec3<i32>(dx, dy, dz));
                let grid_size = params.grid_size_x * params.grid_size_y * params.grid_size_z;
                let cell_idx = neighbor_hash % grid_size;
                
                let cell = grid_cells[cell_idx];
                
                for (var i = 0u; i < cell.count; i++) {
                    let sorted_idx = cell.start + i;
                    if (sorted_idx >= params.num_particles) {
                        break;
                    }
                    
                    let j = particle_indices[sorted_idx].index;
                    let particle_j = particles[j];
                    let pos_j = particle_j.position.xyz;
                    let mass_j = particle_j.density_pressure_mass.z;
                    let alpha_j = dfsph_data[j].alpha;
                    let divergence_j = dfsph_data[j].divergence;
                    
                    let k_j = alpha_j * divergence_j;
                    let k_ij = k_i + k_j;
                    
                    let r_vec = pos_i - pos_j;
                    let r = length(r_vec);
                    
                    if (r < 2.0 * params.h) {
                        let grad = cubic_spline_gradient(r_vec, params.h);
                        velocity_correction -= k_ij * mass_j * grad;
                    }
                }
            }
        }
    }
    
    velocity_corrections[idx] = vec4<f32>(velocity_correction, 0.0);
}

// Apply velocity corrections
@compute @workgroup_size(256)
fn apply_velocity_corrections(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.num_particles) {
        return;
    }
    
    var particle = particles[idx];
    particle.velocity += velocity_corrections[idx];
    particles[idx] = particle;
    
    // Clear corrections for next iteration
    velocity_corrections[idx] = vec4<f32>(0.0);
}
