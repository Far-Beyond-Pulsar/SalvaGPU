use instant::Instant;
use nalgebra::{Point2, Vector2};
use salva2d::gpu::GpuFluidSolver;
use salva2d::kernel::CubicSplineKernel;
use salva2d::object::interaction_groups::InteractionGroups;
use salva2d::object::{Fluid, FluidHandle};
use salva2d::solver::DFSPHSolver;
use salva2d::LiquidWorld;
use std::time::Duration;

/// Benchmark configuration
struct BenchmarkConfig {
    particle_counts: Vec<usize>,
    num_frames: usize,
    num_iterations: usize,
    dt: f32,
}

/// Results for a single benchmark run
struct BenchmarkResult {
    particle_count: usize,
    cpu_avg_time: Duration,
    cpu_total_time: Duration,
    gpu_avg_time: Duration,
    gpu_total_time: Duration,
    gpu_available: bool,
}

impl BenchmarkResult {
    fn print(&self) {
        println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("  Particle Count: {}", self.particle_count);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        
        println!("\n  CPU Performance:");
        println!("    Average frame time: {:.2} ms", self.cpu_avg_time.as_secs_f32() * 1000.0);
        println!("    Total time: {:.2} s", self.cpu_total_time.as_secs_f32());
        println!("    FPS: {:.1}", 1.0 / self.cpu_avg_time.as_secs_f32());
        
        if self.gpu_available {
            println!("\n  GPU Performance:");
            println!("    Average frame time: {:.2} ms", self.gpu_avg_time.as_secs_f32() * 1000.0);
            println!("    Total time: {:.2} s", self.gpu_total_time.as_secs_f32());
            println!("    FPS: {:.1}", 1.0 / self.gpu_avg_time.as_secs_f32());
            
            let speedup = self.cpu_avg_time.as_secs_f32() / self.gpu_avg_time.as_secs_f32();
            println!("\n  Speedup: {:.2}x", speedup);
            
            if speedup > 1.0 {
                println!("  GPU is {:.1}% faster", (speedup - 1.0) * 100.0);
            } else {
                println!("  CPU is {:.1}% faster", (1.0 / speedup - 1.0) * 100.0);
            }
        } else {
            println!("\n  GPU: NOT AVAILABLE");
            println!("    Build with --features gpu-acceleration to enable");
        }
        
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    }
}

/// Creates a fluid with particles in a grid
fn create_fluid(particle_count: usize, particle_radius: f32) -> Fluid {
    let particles_per_side = (particle_count as f32).sqrt() as usize;
    let spacing = particle_radius * 2.1;
    
    let mut positions = Vec::new();
    let mut velocities = Vec::new();
    
    for i in 0..particles_per_side {
        for j in 0..particles_per_side {
            if positions.len() >= particle_count {
                break;
            }
            
            let x = -1.0 + (i as f32) * spacing;
            let y = -1.0 + (j as f32) * spacing;
            
            positions.push(Point2::new(x, y));
            velocities.push(Vector2::new(0.0, 0.0));
        }
        
        if positions.len() >= particle_count {
            break;
        }
    }
    
    let mut fluid = Fluid::new(positions, particle_radius, 1.0, InteractionGroups::all());
    fluid.velocities = velocities;
    fluid
}

/// Runs CPU benchmark
fn benchmark_cpu(
    particle_count: usize,
    num_frames: usize,
    num_iterations: usize,
    dt: f32,
) -> Duration {
    let particle_radius = 0.025;
    let smoothing_factor = 2.0;
    
    let mut total_duration = Duration::ZERO;
    
    println!("  Running CPU benchmark ({} particles, {} iterations)...", 
             particle_count, num_iterations);
    
    for iteration in 0..num_iterations {
        print!("    Iteration {}/{}... ", iteration + 1, num_iterations);
        std::io::Write::flush(&mut std::io::stdout()).ok();
        
        // Create world and fluid
        let mut world: LiquidWorld = LiquidWorld::new(
            DFSPHSolver::<CubicSplineKernel, CubicSplineKernel>::new(),
            particle_radius,
            smoothing_factor,
        );
        
        let fluid = create_fluid(particle_count, particle_radius);
        world.add_fluid(fluid);
        
        let gravity = Vector2::new(0.0, -9.81);
        
        // Run simulation
        let start = Instant::now();
        
        for _ in 0..num_frames {
            world.step(dt, &gravity);
        }
        
        let duration = start.elapsed();
        total_duration += duration;
        
        println!("{:.2} s", duration.as_secs_f32());
    }
    
    total_duration
}

/// Runs GPU benchmark
async fn benchmark_gpu(
    particle_count: usize,
    num_frames: usize,
    num_iterations: usize,
    dt: f32,
) -> Duration {
    let particle_radius = 0.025;
    
    let mut total_duration = Duration::ZERO;
    
    println!("  Running GPU benchmark ({} particles, {} iterations)...", 
             particle_count, num_iterations);
    
    for iteration in 0..num_iterations {
        print!("    Iteration {}/{}... ", iteration + 1, num_iterations);
        std::io::Write::flush(&mut std::io::stdout()).ok();
        
        // Create GPU solver
        let mut gpu_solver = match GpuFluidSolver::new(particle_count).await {
            Ok(solver) => solver,
            Err(e) => {
                println!("Failed to create GPU solver: {}", e);
                return Duration::ZERO;
            }
        };
        
        // Create fluid data
        let fluid = create_fluid(particle_count, particle_radius);
        let fluids = vec![fluid];
        
        // Initialize GPU solver
        gpu_solver.init_from_fluids(&fluids);
        
        let gravity = Vector2::new(0.0, -9.81);
        
        // Run simulation
        let start = Instant::now();
        
        for _ in 0..num_frames {
            gpu_solver.step(dt, &gravity);
        }
        
        // Ensure GPU work is complete
        gpu_solver.context().device.poll(wgpu::Maintain::Wait);
        
        let duration = start.elapsed();
        total_duration += duration;
        
        println!("{:.2} s", duration.as_secs_f32());
    }
    
    total_duration
}

/// Main benchmark function
async fn run_benchmarks(config: BenchmarkConfig) {
    println!("\n╔════════════════════════════════════════════════════╗");
    println!("║       Salva GPU vs CPU Performance Benchmark      ║");
    println!("╚════════════════════════════════════════════════════╝\n");
    
    println!("Configuration:");
    println!("  Frames per run: {}", config.num_frames);
    println!("  Iterations: {}", config.num_iterations);
    println!("  Time step: {} s", config.dt);
    println!("  Particle counts: {:?}", config.particle_counts);
    
    let mut results = Vec::new();
    
    for &particle_count in &config.particle_counts {
        println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Testing with {} particles", particle_count);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        
        // CPU benchmark
        let cpu_total = benchmark_cpu(
            particle_count,
            config.num_frames,
            config.num_iterations,
            config.dt,
        );
        let cpu_avg = cpu_total / (config.num_iterations as u32);
        
        // GPU benchmark - always try
        let (gpu_total, gpu_avg, gpu_available) = {
            let gpu_total = benchmark_gpu(
                particle_count,
                config.num_frames,
                config.num_iterations,
                config.dt,
            ).await;
            
            let gpu_available = gpu_total != Duration::ZERO;
            let gpu_avg = if gpu_available {
                gpu_total / (config.num_iterations as u32)
            } else {
                Duration::ZERO
            };
            
            (gpu_total, gpu_avg, gpu_available)
        };
        
        results.push(BenchmarkResult {
            particle_count,
            cpu_avg_time: cpu_avg / config.num_frames as u32,
            cpu_total_time: cpu_total,
            gpu_avg_time: gpu_avg / config.num_frames as u32,
            gpu_total_time: gpu_total,
            gpu_available,
        });
    }
    
    // Print summary
    println!("\n\n╔════════════════════════════════════════════════════╗");
    println!("║                  BENCHMARK RESULTS                 ║");
    println!("╚════════════════════════════════════════════════════╝");
    
    for result in results {
        result.print();
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    let config = BenchmarkConfig {
        particle_counts: vec![100, 500, 1000, 2500, 5000, 10000],
        num_frames: 100,
        num_iterations: 10,
        dt: 0.016, // ~60 FPS
    };
    
    // Use pollster to run async code
    pollster::block_on(run_benchmarks(config));
}

#[cfg(target_arch = "wasm32")]
fn main() {
    println!("This example is not supported on WASM (requires high-precision timing)");
}
