//! GPU context management for WGPU-based simulation

use wgpu::{
    Adapter, Backends, Device, DeviceDescriptor, Features, Instance, InstanceDescriptor, Limits,
    PowerPreference, Queue, RequestAdapterOptions,
};

/// GPU context that manages the WGPU device and queue
pub struct GpuContext {
    /// The WGPU device
    pub device: Device,
    /// The WGPU command queue
    pub queue: Queue,
    /// The WGPU adapter info
    pub adapter: Adapter,
}

impl GpuContext {
    /// Creates a new GPU context with default settings
    pub async fn new() -> Result<Self, GpuContextError> {
        Self::with_power_preference(PowerPreference::HighPerformance).await
    }

    /// Creates a new GPU context with specified power preference
    pub async fn with_power_preference(
        power_preference: PowerPreference,
    ) -> Result<Self, GpuContextError> {
        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or(GpuContextError::NoAdapter)?;

        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: Some("Salva GPU Device"),
                    required_features: Features::empty(),
                    required_limits: Limits::default(),
                    memory_hints: Default::default(),
                },
                None,
            )
            .await
            .map_err(|e| GpuContextError::DeviceRequest(e.to_string()))?;

        Ok(Self {
            device,
            queue,
            adapter,
        })
    }

    /// Returns information about the GPU adapter
    pub fn adapter_info(&self) -> wgpu::AdapterInfo {
        self.adapter.get_info()
    }

    /// Checks if the GPU context is available and ready
    pub fn is_ready(&self) -> bool {
        true // If we got here, device is ready
    }
}

/// Errors that can occur when creating a GPU context
#[derive(Debug)]
pub enum GpuContextError {
    /// No suitable GPU adapter found
    NoAdapter,
    /// Failed to request device
    DeviceRequest(String),
}

impl std::fmt::Display for GpuContextError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuContextError::NoAdapter => write!(f, "No suitable GPU adapter found"),
            GpuContextError::DeviceRequest(msg) => write!(f, "Failed to request device: {}", msg),
        }
    }
}

impl std::error::Error for GpuContextError {}
