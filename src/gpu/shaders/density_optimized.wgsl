// Optimized density computation using sorted spatial hash grid

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

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<storage, read> particle_indices: array<ParticleIndex>;
@group(0) @binding(2) var<storage, read> grid_cells: array<GridCell>;
@group(0) @binding(3) var<uniform> params: SimulationParams;

// Cubic spline kernel
fn cubic_spline_kernel(r: f32, h: f32) -> f32 {
    let q = r / h;
    
    #ifdef DIM_2D
    let sigma = 10.0 / (7.0 * 3.141592653589793 * h * h);
    #else
    let sigma = 1.0 / (3.141592653589793 * h * h * h);
    #endif
    
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

// Compute spatial hash for a position
fn spatial_hash_from_position(pos: vec3<f32>) -> u32 {
    let grid_pos = vec3<i32>(floor(pos / params.grid_cell_size));
    
    let wrapped = vec3<u32>(
        u32(grid_pos.x) & (params.grid_size_x - 1u),
        u32(grid_pos.y) & (params.grid_size_y - 1u),
        u32(grid_pos.z) & (params.grid_size_z - 1u)
    );
    
    let p1: u32 = 73856093u;
    let p2: u32 = 19349663u;
    let p3: u32 = 83492791u;
    
    return (wrapped.x * p1) ^ (wrapped.y * p2) ^ (wrapped.z * p3);
}

// Get hash for neighboring cell
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

@compute @workgroup_size(256)
fn compute_density_optimized(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.num_particles) {
        return;
    }
    
    var particle_i = particles[idx];
    let pos_i = particle_i.position.xyz;
    var density = 0.0;
    
    // Check 27 neighboring cells (3x3x3 grid)
    for (var dx = -1; dx <= 1; dx++) {
        for (var dy = -1; dy <= 1; dy++) {
            for (var dz = -1; dz <= 1; dz++) {
                let neighbor_hash = get_neighbor_hash(pos_i, vec3<i32>(dx, dy, dz));
                let grid_size = params.grid_size_x * params.grid_size_y * params.grid_size_z;
                let cell_idx = neighbor_hash % grid_size;
                
                let cell = grid_cells[cell_idx];
                
                // Iterate through particles in this cell
                for (var i = 0u; i < cell.count; i++) {
                    let sorted_idx = cell.start + i;
                    if (sorted_idx >= params.num_particles) {
                        break;
                    }
                    
                    let j = particle_indices[sorted_idx].index;
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
    }
    
    // Update density
    particle_i.density_pressure_mass.x = density;
    
    // Compute pressure using Tait equation of state
    let gamma = 7.0;
    let B = params.pressure_stiffness;
    let rho_ratio = density / params.density0;
    particle_i.density_pressure_mass.y = B * (pow(rho_ratio, gamma) - 1.0);
    
    particles[idx] = particle_i;
}

// Store density gradient for DFSPH
@group(0) @binding(4) var<storage, read_write> density_gradients: array<vec4<f32>>;

@compute @workgroup_size(256)
fn compute_density_gradient(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.num_particles) {
        return;
    }
    
    let particle_i = particles[idx];
    let pos_i = particle_i.position.xyz;
    let density_i = particle_i.density_pressure_mass.x;
    
    var grad_sum = vec3<f32>(0.0);
    
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
                        grad_sum += mass_j * grad;
                    }
                }
            }
        }
    }
    
    density_gradients[idx] = vec4<f32>(grad_sum, 0.0);
}
