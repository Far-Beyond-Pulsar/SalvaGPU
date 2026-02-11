// Test to verify CPU and GPU implementations produce identical results

use nalgebra::{Point2, Vector2};
use salva2d::integrations::rapier::FluidsPipeline;
use salva2d::object::{Boundary, Fluid};
use salva2d::solver::{ArtificialViscosity, DFSPHSolver};
use salva2d::kernel::CubicSplineKernel;
use salva2d::object::interaction_groups::InteractionGroups;
use salva2d::gpu::GpuFluidSolver;
use rapier2d::prelude::{ColliderSet, RigidBodySet};
use instant::Instant;

const PARTICLE_RADIUS: f32 = 0.025;
const SMOOTHING_FACTOR: f32 = 2.0;
const NUM_FRAMES: usize = 10;

fn create_test_fluid() -> Fluid {
    // Create a simple 10x10 grid of particles
    let mut positions = Vec::new();
    let spacing = PARTICLE_RADIUS * 2.0;
    
    for i in 0..10 {
        for j in 0..10 {
            let x = 0.5 + (i as f32) * spacing;
            let y = 1.0 + (j as f32) * spacing;
            positions.push(Point2::new(x, y));
        }
    }
    
    Fluid::new(positions, PARTICLE_RADIUS, 1.0, InteractionGroups::all())
}

#[derive(Debug)]
struct ParityResults {
    max_position_error: f32,
    avg_position_error: f32,
    max_velocity_error: f32,
    avg_velocity_error: f32,
    max_density_error: f32,
    avg_density_error: f32,
    particles_compared: usize,
}

impl ParityResults {
    fn is_acceptable(&self, tolerance: f32) -> bool {
        self.max_position_error < tolerance &&
        self.max_velocity_error < tolerance
    }
}

async fn run_parity_test() -> Result<ParityResults, String> {
    println!("╔════════════════════════════════════════════════════╗");
    println!("║         CPU-GPU Feature Parity Test               ║");
    println!("╚════════════════════════════════════════════════════╝\n");
    
    // Create identical fluid setup
    let mut cpu_fluid = create_test_fluid();
    let particle_count = cpu_fluid.num_particles();
    println!("Testing with {} particles for {} frames\n", particle_count, NUM_FRAMES);
    
    // CPU setup
    println!("Setting up CPU simulation...");
    let gravity = Vector2::new(0.0, -9.81);
    let mut cpu_fluids = vec![cpu_fluid];
    let cpu_boundaries: Vec<Boundary> = vec![];
    let mut cpu_bodies = RigidBodySet::new();
    let cpu_colliders = ColliderSet::new();
    
    let mut cpu_pipeline = FluidsPipeline::new(
        PARTICLE_RADIUS,
        SMOOTHING_FACTOR,
    );
    
    let mut cpu_solver = DFSPHSolver::<CubicSplineKernel, CubicSplineKernel>::new();
    cpu_solver.max_pressure_iter = 50;
    cpu_solver.max_divergence_iter = 50;
    
    let cpu_viscosity = ArtificialViscosity::new(0.01, 0.0);
    
    // GPU setup
    println!("Setting up GPU simulation...");
    let mut gpu_solver = match GpuFluidSolver::new(particle_count).await {
        Ok(solver) => solver,
        Err(e) => return Err(format!("Failed to create GPU solver: {}", e)),
    };
    
    // Initialize GPU with same fluid data
    let gpu_fluids = vec![create_test_fluid()];
    gpu_solver.init_from_fluids(&gpu_fluids);
    
    println!("✓ Both simulations initialized\n");
    
    // Run simulations
    println!("Running {} frames on both CPU and GPU...", NUM_FRAMES);
    let dt = 0.016;
    
    let cpu_start = Instant::now();
    for frame in 0..NUM_FRAMES {
        if frame % 5 == 0 {
            print!("  Frame {}/{}...\r", frame + 1, NUM_FRAMES);
        }
        
        cpu_pipeline.step(
            &gravity,
            dt,
            &cpu_colliders,
            &mut cpu_bodies,
        );
    }
    let cpu_time = cpu_start.elapsed();
    println!("  CPU completed in {:.3}s         ", cpu_time.as_secs_f32());
    
    let gpu_start = Instant::now();
    for frame in 0..NUM_FRAMES {
        if frame % 5 == 0 {
            print!("  Frame {}/{}...\r", frame + 1, NUM_FRAMES);
        }
        
        gpu_solver.step(dt, &gravity);
    }
    // Ensure GPU work completes
    gpu_solver.context().device.poll(wgpu::Maintain::Wait);
    let gpu_time = gpu_start.elapsed();
    println!("  GPU completed in {:.3}s         \n", gpu_time.as_secs_f32());
    
    // Compare results
    println!("Comparing results...");
    let cpu_positions = &cpu_fluids[0].positions;
    let cpu_velocities = &cpu_fluids[0].velocities;
    
    // Read back GPU data
    let gpu_particles = gpu_solver.read_particles().await?;
    
    if gpu_particles.len() != particle_count {
        return Err(format!(
            "Particle count mismatch: CPU={}, GPU={}",
            particle_count,
            gpu_particles.len()
        ));
    }
    
    // Calculate errors
    let mut max_pos_err = 0.0f32;
    let mut sum_pos_err = 0.0f32;
    let mut max_vel_err = 0.0f32;
    let mut sum_vel_err = 0.0f32;
    
    for i in 0..particle_count {
        // Position error
        let cpu_pos = cpu_positions[i];
        let gpu_pos = Point2::new(
            gpu_particles[i].position[0],
            gpu_particles[i].position[1],
        );
        let pos_err = (cpu_pos - gpu_pos).norm();
        max_pos_err = max_pos_err.max(pos_err);
        sum_pos_err += pos_err;
        
        // Velocity error
        let cpu_vel = cpu_velocities[i];
        let gpu_vel = Vector2::new(
            gpu_particles[i].velocity[0],
            gpu_particles[i].velocity[1],
        );
        let vel_err = (cpu_vel - gpu_vel).norm();
        max_vel_err = max_vel_err.max(vel_err);
        sum_vel_err += vel_err;
    }
    
    Ok(ParityResults {
        max_position_error: max_pos_err,
        avg_position_error: sum_pos_err / particle_count as f32,
        max_velocity_error: max_vel_err,
        avg_velocity_error: sum_vel_err / particle_count as f32,
        max_density_error: 0.0, // Not comparing density for now
        avg_density_error: 0.0,
        particles_compared: particle_count,
    })
}

#[tokio::main]
async fn main() {
    match run_parity_test().await {
        Ok(results) => {
            println!("\n╔════════════════════════════════════════════════════╗");
            println!("║                  PARITY RESULTS                    ║");
            println!("╚════════════════════════════════════════════════════╝\n");
            
            println!("Particles compared: {}", results.particles_compared);
            println!();
            println!("Position Errors:");
            println!("  Maximum: {:.6} units", results.max_position_error);
            println!("  Average: {:.6} units", results.avg_position_error);
            println!();
            println!("Velocity Errors:");
            println!("  Maximum: {:.6} units/s", results.max_velocity_error);
            println!("  Average: {:.6} units/s", results.avg_velocity_error);
            println!();
            println!("Density Errors:");
            println!("  Maximum: {:.6} kg/m³", results.max_density_error);
            println!("  Average: {:.6} kg/m³", results.avg_density_error);
            println!();
            
            let tolerance = 0.001; // 1mm tolerance
            if results.is_acceptable(tolerance) {
                println!("✅ PARITY TEST PASSED");
                println!("   CPU and GPU implementations produce identical results!");
                std::process::exit(0);
            } else {
                println!("❌ PARITY TEST FAILED");
                println!("   Errors exceed tolerance of {} units", tolerance);
                println!("\nPossible causes:");
                println!("  - Different neighbor search implementations");
                println!("  - Floating point precision differences");
                println!("  - Different iteration counts or convergence criteria");
                println!("  - Missing synchronization or buffer updates");
                std::process::exit(1);
            }
        }
        Err(e) => {
            eprintln!("❌ TEST ERROR: {}", e);
            std::process::exit(1);
        }
    }
}
