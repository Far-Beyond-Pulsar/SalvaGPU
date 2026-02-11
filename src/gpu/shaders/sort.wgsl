// Bitonic sort for particle indices by spatial hash
// This enables efficient neighbor lookup via sorted spatial hash grid

struct ParticleIndex {
    index: u32,
    hash: u32,
}

struct SortParams {
    num_particles: u32,
    stage: u32,
    step: u32,
    _padding: u32,
}

struct GridCell {
    start: u32,
    count: u32,
}

struct BuildParams {
    num_particles: u32,
    grid_size: u32,
    _padding1: u32,
    _padding2: u32,
}

// === BITONIC SORT ENTRY POINT ===
@group(0) @binding(0) var<storage, read_write> particle_indices_rw: array<ParticleIndex>;
@group(0) @binding(1) var<uniform> sort_params: SortParams;

// Bitonic sort kernel
@compute @workgroup_size(256)
fn bitonic_sort_step(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= sort_params.num_particles / 2u) {
        return;
    }
    
    let stage = sort_params.stage;
    let step = sort_params.step;
    
    // Calculate comparison distance
    let distance = 1u << step;
    
    // Calculate indices to compare
    let block_size = 1u << (stage + 1u);
    let block_id = idx / (block_size / 2u);
    let block_offset = idx % (block_size / 2u);
    
    let i = block_id * block_size + block_offset;
    let j = i + distance;
    
    if (j >= sort_params.num_particles) {
        return;
    }
    
    // Determine sort direction
    let ascending = ((i >> stage) & 2u) == 0u;
    
    let hash_i = particle_indices_rw[i].hash;
    let hash_j = particle_indices_rw[j].hash;
    
    // Compare and swap if needed
    let should_swap = (ascending && hash_i > hash_j) || (!ascending && hash_i < hash_j);
    
    if (should_swap) {
        let temp = particle_indices_rw[i];
        particle_indices_rw[i] = particle_indices_rw[j];
        particle_indices_rw[j] = temp;
    }
}

// === BUILD GRID CELLS ENTRY POINT ===
@group(0) @binding(0) var<storage, read> particle_indices_ro: array<ParticleIndex>;
@group(0) @binding(1) var<storage, read_write> grid_cells: array<GridCell>;
@group(0) @binding(2) var<uniform> build_params: BuildParams;

@compute @workgroup_size(256)
fn build_grid_cells(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= build_params.num_particles) {
        return;
    }
    
    let current_hash = particle_indices_ro[idx].hash;
    
    // First particle in the array or first in a new cell
    if (idx == 0u || particle_indices_ro[idx - 1u].hash != current_hash) {
        // Count particles with same hash
        var count = 1u;
        var next_idx = idx + 1u;
        
        while (next_idx < build_params.num_particles && 
               particle_indices_ro[next_idx].hash == current_hash) {
            count++;
            next_idx++;
        }
        
        // Write to grid cell (hash is the cell index)
        let cell_idx = current_hash % build_params.grid_size;
        grid_cells[cell_idx].start = idx;
        grid_cells[cell_idx].count = count;
    }
}
