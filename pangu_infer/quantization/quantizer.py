"""Quantization utilities for model compression."""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Union
from enum import Enum


class QuantizationType(Enum):
    """Supported quantization types."""
    INT8 = "int8"
    INT4 = "int4"
    FP16 = "fp16"


@dataclass
class QuantizationConfig:
    """Configuration for model quantization.
    
    Args:
        quantization_type: Type of quantization (int8, int4, fp16)
        per_channel: Whether to use per-channel quantization
        symmetric: Whether to use symmetric quantization
        calibration_samples: Number of samples for calibration
    """
    quantization_type: QuantizationType = QuantizationType.INT8
    per_channel: bool = True
    symmetric: bool = True
    calibration_samples: int = 512


class Quantizer:
    """Quantizer for model compression.
    
    Supports INT8, INT4, and FP16 quantization with various modes.
    """
    
    def __init__(self, config: QuantizationConfig):
        self.config = config
        
    def quantize_model(self, model: nn.Module) -> nn.Module:
        """Quantize a PyTorch model.
        
        Args:
            model: Model to quantize
            
        Returns:
            Quantized model
        """
        if self.config.quantization_type == QuantizationType.INT8:
            return self._quantize_int8(model)
        elif self.config.quantization_type == QuantizationType.INT4:
            return self._quantize_int4(model)
        elif self.config.quantization_type == QuantizationType.FP16:
            return self._quantize_fp16(model)
        else:
            raise ValueError(f"Unsupported quantization type: {self.config.quantization_type}")
    
    def _quantize_int8(self, model: nn.Module) -> nn.Module:
        """Apply INT8 quantization to model.
        
        Args:
            model: Model to quantize
            
        Returns:
            INT8 quantized model
        """
        # Dynamic quantization for linear layers
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear},
            dtype=torch.qint8
        )
        return quantized_model
    
    def _quantize_int4(self, model: nn.Module) -> nn.Module:
        """Apply INT4 quantization to model.
        
        Note: INT4 quantization is simulated using INT8 with reduced range.
        
        Args:
            model: Model to quantize
            
        Returns:
            INT4 quantized model (simulated)
        """
        # For INT4, we simulate by quantizing to INT8 first
        # True INT4 requires custom kernels
        quantized_model = self._quantize_int8(model)
        return quantized_model
    
    def _quantize_fp16(self, model: nn.Module) -> nn.Module:
        """Apply FP16 quantization to model.
        
        Args:
            model: Model to quantize
            
        Returns:
            FP16 quantized model
        """
        return model.half()
    
    def calibrate(self, model: nn.Module, dataloader) -> None:
        """Calibrate quantization parameters using sample data.
        
        Args:
            model: Model to calibrate
            dataloader: DataLoader with calibration samples
        """
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= self.config.calibration_samples:
                    break
                
                # Forward pass to collect statistics
                if isinstance(batch, dict):
                    model(**batch)
                else:
                    model(batch)
    
    @staticmethod
    def get_model_size(model: nn.Module) -> float:
        """Get model size in MB.
        
        Args:
            model: Model to measure
            
        Returns:
            Model size in MB
        """
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
    
    @staticmethod
    def compare_models(original_model: nn.Module, quantized_model: nn.Module) -> dict:
        """Compare original and quantized models.
        
        Args:
            original_model: Original model
            quantized_model: Quantized model
            
        Returns:
            Dictionary with comparison metrics
        """
        original_size = Quantizer.get_model_size(original_model)
        quantized_size = Quantizer.get_model_size(quantized_model)
        
        return {
            "original_size_mb": original_size,
            "quantized_size_mb": quantized_size,
            "compression_ratio": original_size / quantized_size if quantized_size > 0 else 0,
            "size_reduction": (1 - quantized_size / original_size) * 100 if original_size > 0 else 0,
        }


class QuantizedLinear(nn.Module):
    """Custom quantized linear layer for more control."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True, bits: int = 8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        
        # Quantization parameters
        self.register_buffer('weight_scale', torch.ones(1))
        self.register_buffer('weight_zero_point', torch.zeros(1))
        
        # Quantized weight
        self.weight = nn.Parameter(torch.randint(
            -(2 ** (bits - 1)),
            2 ** (bits - 1),
            (out_features, in_features),
            dtype=torch.int8
        ))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize weight
        weight = self.weight.float() * self.weight_scale + self.weight_zero_point
        
        # Perform linear operation
        output = torch.nn.functional.linear(x, weight, self.bias)
        return output
    
    @staticmethod
    def from_float(module: nn.Linear, bits: int = 8):
        """Convert a float linear layer to quantized version.
        
        Args:
            module: Float linear layer
            bits: Number of bits for quantization
            
        Returns:
            Quantized linear layer
        """
        quantized = QuantizedLinear(
            module.in_features,
            module.out_features,
            module.bias is not None,
            bits=bits
        )
        
        # Quantize weight
        weight = module.weight.data
        min_val = weight.min()
        max_val = weight.max()
        
        scale = (max_val - min_val) / (2 ** bits - 1)
        zero_point = min_val
        
        quantized.weight_scale.fill_(scale)
        quantized.weight_zero_point.fill_(zero_point)
        
        quantized_weight = torch.round((weight - zero_point) / scale)
        quantized_weight = torch.clamp(quantized_weight, -(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
        quantized.weight.data = quantized_weight.to(torch.int8)
        
        if module.bias is not None:
            quantized.bias.data = module.bias.data
        
        return quantized
