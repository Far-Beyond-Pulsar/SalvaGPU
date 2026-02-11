// Multi-fluid simulation with different properties per fluid

struct FluidProperties {
    density0: f32,
    viscosity: f32,
    surface_tension: f32,
    color: vec4<f32>,
}

struct Particle {
    position: vec4<f32>,
    velocity: vec4<f32>,
    force: vec4<f32>,
    density_pressure_mass: vec4<f32>,
    fluid_id: u32,      // Which fluid this particle belongs to
    _padding: vec3<u32>,
}

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<storage, read> particle_indices: array<ParticleIndex>;
@group(0) @binding(2) var<storage, read> grid_cells: array<GridCell>;
@group(0) @binding(3) var<uniform> params: SimulationParams;
@group(0) @binding(4) var<storage, read> fluid_properties: array<FluidProperties>;

struct ParticleIndex {
    index: u32,
    hash: u32,
}

struct GridCell {
    start: u32,
    count: u32,
}

struct SimulationParams {
    dt: f32,
    inv_dt: f32,
    h: f32,
    particle_radius: f32,
    particle_mass: f32,
    num_particles: u32,
    grid_cell_size: f32,
    num_fluids: u32,
    gravity: vec4<f32>,
    grid_size_x: u32,
    grid_size_y: u32,
    grid_size_z: u32,
    pressure_stiffness: f32,
}

fn cubic_spline_kernel(r: f32, h: f32) -> f32 {
    let q = r / h;
    
    // Using 3D formula (works for 2D with z=0)
    let sigma = 1.0 / (3.141592653589793 * h * h * h);
    
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

fn cubic_spline_gradient(r_vec: vec3<f32>, h: f32) -> vec3<f32> {
    let r = length(r_vec);
    
    if (r < 1e-6 || r >= 2.0 * h) {
        return vec3<f32>(0.0);
    }
    
    let q = r / h;
    
    // Using 3D formula (works for 2D with z=0)
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

// Multi-fluid density computation
@compute @workgroup_size(256)
fn compute_density_multifluid(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.num_particles) {
        return;
    }
    
    var particle_i = particles[idx];
    let pos_i = particle_i.position.xyz;
    let fluid_id_i = particle_i.fluid_id;
    let props_i = fluid_properties[fluid_id_i];
    
    var density = 0.0;
    
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
                    
                    let r_vec = pos_i - pos_j;
                    let r = length(r_vec);
                    
                    if (r < 2.0 * params.h) {
                        density += mass_j * cubic_spline_kernel(r, params.h);
                    }
                }
            }
        }
    }
    
    // Update density
    particle_i.density_pressure_mass.x = density;
    
    // Compute pressure using fluid-specific rest density
    let gamma = 7.0;
    let B = params.pressure_stiffness;
    let rho_ratio = density / props_i.density0;
    particle_i.density_pressure_mass.y = B * (pow(rho_ratio, gamma) - 1.0);
    
    particles[idx] = particle_i;
}

// Multi-fluid force computation with different viscosities
@compute @workgroup_size(256)
fn compute_forces_multifluid(@builtin(global_invocation_id) global_id: vec3<u32>) {
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
    let fluid_id_i = particle_i.fluid_id;
    let props_i = fluid_properties[fluid_id_i];
    
    var force_pressure = vec3<f32>(0.0);
    var force_viscosity = vec3<f32>(0.0);
    var force_surface = vec3<f32>(0.0);
    
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
                    let vel_j = particle_j.velocity.xyz;
                    let density_j = particle_j.density_pressure_mass.x;
                    let pressure_j = particle_j.density_pressure_mass.y;
                    let mass_j = particle_j.density_pressure_mass.z;
                    let fluid_id_j = particle_j.fluid_id;
                    let props_j = fluid_properties[fluid_id_j];
                    
                    let r_vec = pos_i - pos_j;
                    let r = length(r_vec);
                    
                    if (r < 2.0 * params.h && r > 1e-6) {
                        let grad = cubic_spline_gradient(r_vec, params.h);
                        
                        // Pressure force (symmetric SPH formulation)
                        let pressure_term = (pressure_i / (density_i * density_i)) + 
                                          (pressure_j / (density_j * density_j));
                        force_pressure -= mass_j * pressure_term * grad;
                        
                        // Viscosity force (average viscosity between fluids)
                        let avg_viscosity = (props_i.viscosity + props_j.viscosity) * 0.5;
                        let vel_diff = vel_j - vel_i;
                        let visc_coeff = 2.0 * avg_viscosity / (density_i + density_j);
                        force_viscosity += visc_coeff * mass_j * vel_diff * dot(r_vec, grad) / 
                                         (dot(r_vec, r_vec) + 0.01 * params.h * params.h);
                        
                        // Surface tension (only between different fluids)
                        if (fluid_id_i != fluid_id_j) {
                            let avg_tension = (props_i.surface_tension + props_j.surface_tension) * 0.5;
                            force_surface -= avg_tension * mass_j * r_vec / r;
                        }
                    }
                }
            }
        }
    }
    
    // Total force
    let total_force = force_pressure + force_viscosity + force_surface + 
                     params.gravity.xyz * mass_i;
    
    particle_i.force = vec4<f32>(total_force, 0.0);
    particles[idx] = particle_i;
}
