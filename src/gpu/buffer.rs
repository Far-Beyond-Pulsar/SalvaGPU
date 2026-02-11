//! GPU buffer management with delta synchronization

use bytemuck::{Pod, Zeroable};
use std::collections::HashSet;
use wgpu::{
    Buffer, BufferAddress, BufferDescriptor, BufferUsages, Device, Queue, util::DeviceExt,
};

/// Manages GPU buffers with delta-based synchronization to minimize CPU-GPU transfers
pub struct DeltaBufferManager<T: Pod> {
    /// The GPU buffer
    buffer: Buffer,
    /// CPU-side copy of the data
    cpu_data: Vec<T>,
    /// Indices of particles that have been modified since last sync
    dirty_indices: HashSet<usize>,
    /// Whether the entire buffer needs to be reuploaded
    full_sync_required: bool,
    /// Capacity of the buffer
    capacity: usize,
}

impl<T: Pod + Zeroable> DeltaBufferManager<T> {
    /// Creates a new delta buffer manager with the given capacity
    pub fn new(device: &Device, capacity: usize, usage: BufferUsages, label: &str) -> Self {
        let buffer_size = (capacity * std::mem::size_of::<T>()) as BufferAddress;
        
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some(label),
            size: buffer_size,
            usage: usage | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            buffer,
            cpu_data: vec![T::zeroed(); capacity],
            dirty_indices: HashSet::new(),
            full_sync_required: false,
            capacity,
        }
    }

    /// Creates a buffer initialized with the given data
    pub fn with_data(device: &Device, data: &[T], usage: BufferUsages, label: &str) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage: usage | BufferUsages::COPY_DST,
        });

        Self {
            buffer,
            cpu_data: data.to_vec(),
            dirty_indices: HashSet::new(),
            full_sync_required: false,
            capacity: data.len(),
        }
    }

    /// Updates a single element and marks it as dirty
    pub fn update(&mut self, index: usize, value: T) {
        if index < self.capacity {
            self.cpu_data[index] = value;
            self.dirty_indices.insert(index);
        }
    }

    /// Updates multiple elements and marks them as dirty
    pub fn update_range(&mut self, start_index: usize, values: &[T]) {
        let end_index = (start_index + values.len()).min(self.capacity);
        let count = end_index - start_index;
        
        if count > 0 {
            self.cpu_data[start_index..end_index].copy_from_slice(&values[..count]);
            for i in start_index..end_index {
                self.dirty_indices.insert(i);
            }
        }
    }

    /// Marks all data as dirty, requiring a full sync
    pub fn mark_all_dirty(&mut self) {
        self.full_sync_required = true;
    }

    /// Synchronizes dirty data to the GPU
    pub fn sync_to_gpu(&mut self, queue: &Queue) {
        if self.full_sync_required {
            // Full buffer upload
            queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(&self.cpu_data));
            self.full_sync_required = false;
            self.dirty_indices.clear();
        } else if !self.dirty_indices.is_empty() {
            // Delta upload - group contiguous ranges for efficiency
            let mut sorted_indices: Vec<_> = self.dirty_indices.iter().copied().collect();
            sorted_indices.sort_unstable();

            let mut range_start = sorted_indices[0];
            let mut range_end = range_start;

            for &idx in sorted_indices.iter().skip(1) {
                if idx == range_end + 1 {
                    range_end = idx;
                } else {
                    // Upload the current range
                    self.upload_range(queue, range_start, range_end + 1);
                    range_start = idx;
                    range_end = idx;
                }
            }
            // Upload the last range
            self.upload_range(queue, range_start, range_end + 1);

            self.dirty_indices.clear();
        }
    }

    /// Uploads a contiguous range to the GPU
    fn upload_range(&self, queue: &Queue, start: usize, end: usize) {
        let offset = (start * std::mem::size_of::<T>()) as BufferAddress;
        let data = &self.cpu_data[start..end];
        queue.write_buffer(&self.buffer, offset, bytemuck::cast_slice(data));
    }

    /// Resizes the buffer (requires full reallocation)
    pub fn resize(&mut self, device: &Device, new_capacity: usize, usage: BufferUsages) {
        if new_capacity != self.capacity {
            self.cpu_data.resize(new_capacity, T::zeroed());
            
            let buffer_size = (new_capacity * std::mem::size_of::<T>()) as BufferAddress;
            self.buffer = device.create_buffer(&BufferDescriptor {
                label: self.buffer.label(),
                size: buffer_size,
                usage: usage | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            
            self.capacity = new_capacity;
            self.full_sync_required = true;
        }
    }

    /// Returns a reference to the GPU buffer
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// Returns the current capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns a reference to the CPU-side data
    pub fn cpu_data(&self) -> &[T] {
        &self.cpu_data
    }

    /// Returns a mutable reference to the CPU-side data (marks all as dirty)
    pub fn cpu_data_mut(&mut self) -> &mut [T] {
        self.full_sync_required = true;
        &mut self.cpu_data
    }

    /// Checks if there are pending changes to sync
    pub fn has_pending_changes(&self) -> bool {
        self.full_sync_required || !self.dirty_indices.is_empty()
    }
}

/// A read-back buffer for getting data from GPU to CPU
pub struct ReadbackBuffer<T: Pod> {
    buffer: Buffer,
    capacity: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Pod> ReadbackBuffer<T> {
    /// Creates a new readback buffer
    pub fn new(device: &Device, capacity: usize, label: &str) -> Self {
        let buffer_size = (capacity * std::mem::size_of::<T>()) as BufferAddress;
        
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some(label),
            size: buffer_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            buffer,
            capacity,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Returns the buffer for use in copy operations
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// Reads data from the buffer (async operation)
    pub async fn read(&self) -> Result<Vec<T>, String> {
        let buffer_slice = self.buffer.slice(..);
        let (sender, receiver) = futures::channel::oneshot::channel();
        
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).ok();
        });

        receiver.await
            .map_err(|_| "Failed to receive map result".to_string())?
            .map_err(|e| format!("Failed to map buffer: {:?}", e))?;

        let data = buffer_slice.get_mapped_range();
        let result = bytemuck::cast_slice(&data).to_vec();
        
        drop(data);
        self.buffer.unmap();

        Ok(result)
    }
}
