"""Model loading and saving utilities."""

import torch
import torch.nn as nn
import os
from typing import Optional, Dict, Any


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: Optional[torch.device] = None,
    strict: bool = True
) -> nn.Module:
    """Load model checkpoint.
    
    Args:
        model: Model to load checkpoint into
        checkpoint_path: Path to checkpoint file
        device: Device to load checkpoint on
        strict: Whether to strictly enforce state dict keys match
        
    Returns:
        Model with loaded checkpoint
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    if device is None:
        device = torch.device('cpu')
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract state dict
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Load state dict into model
    model.load_state_dict(state_dict, strict=strict)
    
    return model


def save_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    optimizer: Optional[Any] = None,
    epoch: Optional[int] = None,
    additional_info: Optional[Dict[str, Any]] = None
):
    """Save model checkpoint.
    
    Args:
        model: Model to save
        checkpoint_path: Path to save checkpoint
        optimizer: Optional optimizer state to save
        epoch: Optional epoch number
        additional_info: Optional additional information to save
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    # Prepare checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if epoch is not None:
        checkpoint['epoch'] = epoch
    
    if additional_info is not None:
        checkpoint.update(additional_info)
    
    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def load_pretrained_weights(
    model: nn.Module,
    pretrained_path: str,
    prefix: str = "",
    exclude_keys: Optional[list] = None
) -> nn.Module:
    """Load pretrained weights with flexible key matching.
    
    Args:
        model: Model to load weights into
        pretrained_path: Path to pretrained weights
        prefix: Prefix to add to keys when loading
        exclude_keys: Keys to exclude from loading
        
    Returns:
        Model with loaded pretrained weights
    """
    if exclude_keys is None:
        exclude_keys = []
    
    # Load pretrained weights
    pretrained_state = torch.load(pretrained_path, map_location='cpu')
    
    if isinstance(pretrained_state, dict) and 'model_state_dict' in pretrained_state:
        pretrained_state = pretrained_state['model_state_dict']
    
    # Filter and adjust keys
    model_state = model.state_dict()
    filtered_state = {}
    
    for key, value in pretrained_state.items():
        # Skip excluded keys
        if any(exclude_key in key for exclude_key in exclude_keys):
            continue
        
        # Add prefix if specified
        if prefix:
            new_key = f"{prefix}.{key}"
        else:
            new_key = key
        
        # Check if key exists in model
        if new_key in model_state:
            # Check shape compatibility
            if value.shape == model_state[new_key].shape:
                filtered_state[new_key] = value
            else:
                print(f"Shape mismatch for {new_key}: {value.shape} vs {model_state[new_key].shape}")
        else:
            print(f"Key not found in model: {new_key}")
    
    # Load filtered state
    model.load_state_dict(filtered_state, strict=False)
    
    print(f"Loaded {len(filtered_state)} parameters from {pretrained_path}")
    
    return model


def convert_checkpoint_format(
    input_path: str,
    output_path: str,
    source_format: str = "pytorch",
    target_format: str = "pytorch"
):
    """Convert checkpoint between different formats.
    
    Args:
        input_path: Input checkpoint path
        output_path: Output checkpoint path
        source_format: Source format (pytorch, mindspore, etc.)
        target_format: Target format (pytorch, mindspore, etc.)
    """
    if source_format == "pytorch" and target_format == "pytorch":
        # Simple copy
        checkpoint = torch.load(input_path, map_location='cpu')
        torch.save(checkpoint, output_path)
    
    elif source_format == "mindspore" and target_format == "pytorch":
        # MindSpore to PyTorch conversion
        try:
            import mindspore as ms
            param_dict = ms.load_checkpoint(input_path)
            
            # Convert MindSpore parameters to PyTorch format
            pytorch_state = {}
            for name, param in param_dict.items():
                pytorch_state[name] = torch.from_numpy(param.asnumpy())
            
            torch.save({'model_state_dict': pytorch_state}, output_path)
        except ImportError:
            raise ImportError("MindSpore is required for conversion from MindSpore format")
    
    elif source_format == "pytorch" and target_format == "mindspore":
        # PyTorch to MindSpore conversion
        try:
            import mindspore as ms
            import numpy as np
            
            checkpoint = torch.load(input_path, map_location='cpu')
            
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Convert PyTorch parameters to MindSpore format
            ms_params = []
            for name, param in state_dict.items():
                param_np = param.cpu().numpy()
                ms_param = ms.Parameter(param_np, name=name)
                ms_params.append({'name': name, 'data': ms_param})
            
            ms.save_checkpoint(ms_params, output_path)
        except ImportError:
            raise ImportError("MindSpore is required for conversion to MindSpore format")
    
    else:
        raise ValueError(f"Unsupported conversion: {source_format} to {target_format}")
    
    print(f"Checkpoint converted from {source_format} to {target_format}")
    print(f"Saved to {output_path}")
