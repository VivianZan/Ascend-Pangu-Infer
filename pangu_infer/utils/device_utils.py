"""Device management utilities for Ascend hardware."""

import torch
import os
from typing import Optional


def get_ascend_device(device_id: int = 0) -> torch.device:
    """Get Ascend device.
    
    Args:
        device_id: Device ID
        
    Returns:
        PyTorch device object
    """
    # Check if Ascend NPU is available
    try:
        import torch_npu
        if torch.npu.is_available():
            return torch.device(f'npu:{device_id}')
    except ImportError:
        pass
    
    # Fallback to CUDA if available
    if torch.cuda.is_available():
        return torch.device(f'cuda:{device_id}')
    
    # Fallback to CPU
    return torch.device('cpu')


def set_device(device_id: int = 0) -> torch.device:
    """Set default device.
    
    Args:
        device_id: Device ID
        
    Returns:
        PyTorch device object
    """
    device = get_ascend_device(device_id)
    
    # Set environment variables for Ascend
    if device.type == 'npu':
        os.environ['NPU_VISIBLE_DEVICES'] = str(device_id)
    elif device.type == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
    
    return device


def get_device_count() -> int:
    """Get number of available devices.
    
    Returns:
        Number of devices
    """
    try:
        import torch_npu
        if torch.npu.is_available():
            return torch.npu.device_count()
    except ImportError:
        pass
    
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    
    return 1  # CPU


def get_device_info(device_id: int = 0) -> dict:
    """Get device information.
    
    Args:
        device_id: Device ID
        
    Returns:
        Dictionary with device information
    """
    info = {
        "device_type": "cpu",
        "device_id": device_id,
        "device_name": "CPU",
        "total_memory": 0,
        "available_memory": 0,
    }
    
    try:
        import torch_npu
        if torch.npu.is_available():
            info["device_type"] = "npu"
            info["device_name"] = f"Ascend NPU {device_id}"
            info["total_memory"] = torch.npu.get_device_properties(device_id).total_memory
            info["available_memory"] = torch.npu.memory_allocated(device_id)
            return info
    except (ImportError, AttributeError):
        pass
    
    if torch.cuda.is_available():
        info["device_type"] = "cuda"
        info["device_name"] = torch.cuda.get_device_name(device_id)
        info["total_memory"] = torch.cuda.get_device_properties(device_id).total_memory
        info["available_memory"] = torch.cuda.memory_allocated(device_id)
    
    return info


def synchronize(device: Optional[torch.device] = None):
    """Synchronize device.
    
    Args:
        device: Device to synchronize (if None, synchronize current device)
    """
    if device is None:
        device = get_ascend_device()
    
    if device.type == 'npu':
        try:
            import torch_npu
            torch.npu.synchronize()
        except ImportError:
            pass
    elif device.type == 'cuda':
        torch.cuda.synchronize()


def empty_cache(device: Optional[torch.device] = None):
    """Empty device cache.
    
    Args:
        device: Device to clear cache (if None, clear current device)
    """
    if device is None:
        device = get_ascend_device()
    
    if device.type == 'npu':
        try:
            import torch_npu
            torch.npu.empty_cache()
        except ImportError:
            pass
    elif device.type == 'cuda':
        torch.cuda.empty_cache()
