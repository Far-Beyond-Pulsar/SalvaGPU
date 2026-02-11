// Flexible boundary system with various shapes and transforms

struct BoundaryParticle {
    position: vec4<f32>,
    normal: vec4<f32>,
    volume: f32,
    is_dynamic: u32,  // 0 = static, 1 = dynamic (can move)
    _padding: vec2<f32>,
}

struct BoundaryObject {
    transform: mat4x4<f32>,     // Transformation matrix
    inv_transform: mat4x4<f32>, // Inverse for collision detection
    velocity: vec4<f32>,         // Linear velocity
    angular_velocity: vec4<f32>, // Angular velocity
    shape_type: u32,             // 0=box, 1=sphere, 2=capsule, 3=mesh
    _padding: vec3<u32>,
    params: vec4<f32>,           // Shape-specific parameters
}

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<storage, read> boundary_particles: array<BoundaryParticle>;
@group(0) @binding(2) var<storage, read> boundary_objects: array<BoundaryObject>;
@group(0) @binding(3) var<uniform> params: SimulationParams;

struct Particle {
    position: vec4<f32>,
    velocity: vec4<f32>,
    force: vec4<f32>,
    density_pressure_mass: vec4<f32>,
}

struct SimulationParams {
    dt: f32,
    inv_dt: f32,
    h: f32,
    density0: f32,
    gravity: vec4<f32>,
    particle_radius: f32,
    particle_mass: f32,
    num_particles: u32,
    num_boundary_particles: u32,
    num_boundary_objects: u32,
    boundary_damping: f32,
    boundary_friction: f32,
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

// SDF for box
fn sdf_box(p: vec3<f32>, half_extents: vec3<f32>) -> f32 {
    let q = abs(p) - half_extents;
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

// SDF for sphere
fn sdf_sphere(p: vec3<f32>, radius: f32) -> f32 {
    return length(p) - radius;
}

// SDF for capsule
fn sdf_capsule(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, radius: f32) -> f32 {
    let pa = p - a;
    let ba = b - a;
    let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - radius;
}

// Get SDF and normal for a boundary object
fn boundary_sdf(obj: BoundaryObject, world_pos: vec3<f32>) -> vec2<f32> {
    // Transform to object space
    let local_pos = (obj.inv_transform * vec4<f32>(world_pos, 1.0)).xyz;
    
    var dist: f32;
    
    switch (obj.shape_type) {
        case 0u: { // Box
            dist = sdf_box(local_pos, obj.params.xyz);
        }
        case 1u: { // Sphere
            dist = sdf_sphere(local_pos, obj.params.x);
        }
        case 2u: { // Capsule
            let half_height = obj.params.x * 0.5;
            dist = sdf_capsule(
                local_pos,
                vec3<f32>(0.0, -half_height, 0.0),
                vec3<f32>(0.0, half_height, 0.0),
                obj.params.y
            );
        }
        default: {
            dist = 1000.0;
        }
    }
    
    return vec2<f32>(dist, 0.0);
}

// Compute boundary forces using sampled boundary particles
@compute @workgroup_size(256)
fn compute_boundary_forces(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.num_particles) {
        return;
    }
    
    var particle = particles[idx];
    let pos = particle.position.xyz;
    let vel = particle.velocity.xyz;
    let pressure = particle.density_pressure_mass.y;
    let density = particle.density_pressure_mass.x;
    let mass = particle.density_pressure_mass.z;
    
    var boundary_force = vec3<f32>(0.0);
    
    // Interact with boundary particles
    for (var i = 0u; i < params.num_boundary_particles; i++) {
        let boundary = boundary_particles[i];
        let boundary_pos = boundary.position.xyz;
        let boundary_normal = boundary.normal.xyz;
        let boundary_volume = boundary.volume;
        
        let r_vec = pos - boundary_pos;
        let r = length(r_vec);
        
        if (r < 2.0 * params.h) {
            let grad = cubic_spline_gradient(r_vec, params.h);
            
            // Pressure force from boundary
            let pressure_term = pressure / (density * density);
            boundary_force -= boundary_volume * params.density0 * pressure_term * grad;
            
            // Friction force
            let relative_vel = vel;
            let normal_vel = dot(relative_vel, boundary_normal) * boundary_normal;
            let tangent_vel = relative_vel - normal_vel;
            boundary_force -= params.boundary_friction * tangent_vel;
        }
    }
    
    // Add boundary force to total force
    particle.force += vec4<f32>(boundary_force, 0.0);
    particles[idx] = particle;
}

// Handle collisions with boundary objects using SDFs
@compute @workgroup_size(256)
fn handle_boundary_collisions(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.num_particles) {
        return;
    }
    
    var particle = particles[idx];
    var pos = particle.position.xyz;
    var vel = particle.velocity.xyz;
    
    // Check collision with each boundary object
    for (var i = 0u; i < params.num_boundary_objects; i++) {
        let obj = boundary_objects[i];
        let sdf_result = boundary_sdf(obj, pos);
        let dist = sdf_result.x;
        
        // Collision detection
        if (dist < params.particle_radius) {
            // Compute normal (gradient of SDF)
            let epsilon = 0.001;
            let normal = normalize(vec3<f32>(
                boundary_sdf(obj, pos + vec3<f32>(epsilon, 0.0, 0.0)).x - dist,
                boundary_sdf(obj, pos + vec3<f32>(0.0, epsilon, 0.0)).x - dist,
                boundary_sdf(obj, pos + vec3<f32>(0.0, 0.0, epsilon)).x - dist
            ));
            
            // Push particle out
            let penetration = params.particle_radius - dist;
            pos += normal * penetration;
            
            // Reflect velocity with damping
            let vel_normal = dot(vel, normal);
            if (vel_normal < 0.0) {
                vel -= (1.0 + params.boundary_damping) * vel_normal * normal;
                
                // Apply friction to tangential velocity
                let vel_tangent = vel - dot(vel, normal) * normal;
                vel -= params.boundary_friction * vel_tangent;
            }
        }
    }
    
    particle.position = vec4<f32>(pos, 1.0);
    particle.velocity = vec4<f32>(vel, 0.0);
    particles[idx] = particle;
}

// Two-way coupling: compute forces on dynamic boundaries
@group(0) @binding(4) var<storage, read_write> boundary_forces: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> boundary_torques: array<vec4<f32>>;

@compute @workgroup_size(256)
fn compute_boundary_coupling(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.num_particles) {
        return;
    }
    
    let particle = particles[idx];
    let pos = particle.position.xyz;
    let vel = particle.velocity.xyz;
    let mass = particle.density_pressure_mass.z;
    let pressure = particle.density_pressure_mass.y;
    let density = particle.density_pressure_mass.x;
    
    // Apply forces to dynamic boundary particles
    for (var i = 0u; i < params.num_boundary_particles; i++) {
        let boundary = boundary_particles[i];
        
        if (boundary.is_dynamic == 0u) {
            continue;
        }
        
        let boundary_pos = boundary.position.xyz;
        let r_vec = pos - boundary_pos;
        let r = length(r_vec);
        
        if (r < 2.0 * params.h) {
            let grad = cubic_spline_gradient(r_vec, params.h);
            let boundary_volume = boundary.volume;
            
            // Pressure force
            let pressure_term = pressure / (density * density);
            let force = boundary_volume * params.density0 * pressure_term * grad;
            
            // Accumulate force and torque
            atomicAdd(&boundary_forces[i].x, u32(force.x * 1000.0));
            atomicAdd(&boundary_forces[i].y, u32(force.y * 1000.0));
            atomicAdd(&boundary_forces[i].z, u32(force.z * 1000.0));
            
            // Torque = r × F
            let torque = cross(r_vec, force);
            atomicAdd(&boundary_torques[i].x, u32(torque.x * 1000.0));
            atomicAdd(&boundary_torques[i].y, u32(torque.y * 1000.0));
            atomicAdd(&boundary_torques[i].z, u32(torque.z * 1000.0));
        }
    }
}
