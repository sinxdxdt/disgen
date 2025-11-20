"""GPU and device management utilities."""

import warnings
from typing import Optional, Union

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not installed. GPU acceleration unavailable.")


def get_device(device: Optional[Union[str, "torch.device"]] = None) -> str:
    """Get the appropriate device for computation.
    
    Args:
        device: Requested device ('cpu', 'cuda', 'mps', or None for auto-detect)
        
    Returns:
        Device string ('cpu', 'cuda', or 'mps')
    """
    if not TORCH_AVAILABLE:
        return "cpu"
    
    if device is not None:
        if isinstance(device, str):
            return device
        return str(device)
    
    # Auto-detect best available device
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def check_gpu_available() -> bool:
    """Check if GPU is available.
    
    Returns:
        True if CUDA or MPS GPU is available
    """
    if not TORCH_AVAILABLE:
        return False
    
    return torch.cuda.is_available() or (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )


def get_gpu_info() -> dict:
    """Get information about available GPUs.
    
    Returns:
        Dictionary with GPU information
    """
    info = {
        "torch_available": TORCH_AVAILABLE,
        "cuda_available": False,
        "mps_available": False,
        "device_count": 0,
        "current_device": "cpu",
    }
    
    if not TORCH_AVAILABLE:
        return info
    
    info["cuda_available"] = torch.cuda.is_available()
    info["mps_available"] = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    
    if info["cuda_available"]:
        info["device_count"] = torch.cuda.device_count()
        info["current_device"] = "cuda"
        info["cuda_version"] = torch.version.cuda
        info["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(info["device_count"])]
    elif info["mps_available"]:
        info["current_device"] = "mps"
        info["device_count"] = 1
    else:
        info["current_device"] = "cpu"
    
    return info


def print_device_info():
    """Print information about available compute devices."""
    info = get_gpu_info()
    
    print("=" * 60)
    print("DEVICE INFORMATION")
    print("=" * 60)
    print(f"PyTorch Available: {info['torch_available']}")
    print(f"CUDA Available: {info['cuda_available']}")
    print(f"MPS Available: {info['mps_available']}")
    print(f"Current Device: {info['current_device']}")
    print(f"Device Count: {info['device_count']}")
    
    if info.get("gpu_names"):
        print("\nGPU Devices:")
        for i, name in enumerate(info["gpu_names"]):
            print(f"  [{i}] {name}")
    
    print("=" * 60)


def to_device(tensor: "torch.Tensor", device: str) -> "torch.Tensor":
    """Move tensor to specified device.
    
    Args:
        tensor: Input tensor
        device: Target device
        
    Returns:
        Tensor on target device
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available")
    
    return tensor.to(device)


class DeviceManager:
    """Context manager for device management."""
    
    def __init__(self, device: Optional[str] = None):
        """Initialize device manager.
        
        Args:
            device: Device to use (or None for auto-detect)
        """
        self.device = get_device(device)
        self.original_device = None
    
    def __enter__(self):
        """Enter device context."""
        if TORCH_AVAILABLE and self.device.startswith("cuda"):
            self.original_device = torch.cuda.current_device()
        return self.device
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit device context."""
        if (TORCH_AVAILABLE and 
            self.original_device is not None and 
            self.device.startswith("cuda")):
            torch.cuda.set_device(self.original_device)
