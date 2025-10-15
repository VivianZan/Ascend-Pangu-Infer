"""Pruning utilities for model optimization."""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, List, Callable
from enum import Enum


class PruningMethod(Enum):
    """Supported pruning methods."""
    MAGNITUDE = "magnitude"
    STRUCTURED = "structured"
    UNSTRUCTURED = "unstructured"


@dataclass
class PruningConfig:
    """Configuration for model pruning.
    
    Args:
        pruning_method: Type of pruning method
        pruning_ratio: Ratio of weights to prune (0.0 to 1.0)
        structured: Whether to use structured pruning
        granularity: Granularity for structured pruning (e.g., 'channel', 'layer')
    """
    pruning_method: PruningMethod = PruningMethod.MAGNITUDE
    pruning_ratio: float = 0.5
    structured: bool = False
    granularity: str = "channel"


class Pruner:
    """Pruner for model optimization.
    
    Supports magnitude-based, structured, and unstructured pruning.
    """
    
    def __init__(self, config: PruningConfig):
        self.config = config
        
    def prune_model(self, model: nn.Module) -> nn.Module:
        """Prune a PyTorch model.
        
        Args:
            model: Model to prune
            
        Returns:
            Pruned model
        """
        if self.config.pruning_method == PruningMethod.MAGNITUDE:
            return self._prune_magnitude(model)
        elif self.config.pruning_method == PruningMethod.STRUCTURED:
            return self._prune_structured(model)
        elif self.config.pruning_method == PruningMethod.UNSTRUCTURED:
            return self._prune_unstructured(model)
        else:
            raise ValueError(f"Unsupported pruning method: {self.config.pruning_method}")
    
    def _prune_magnitude(self, model: nn.Module) -> nn.Module:
        """Apply magnitude-based pruning to model.
        
        Args:
            model: Model to prune
            
        Returns:
            Pruned model
        """
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Compute magnitude of weights
                weight = module.weight.data
                weight_abs = torch.abs(weight)
                
                # Compute threshold
                threshold = torch.quantile(weight_abs.flatten(), self.config.pruning_ratio)
                
                # Create mask
                mask = weight_abs > threshold
                
                # Apply mask
                module.weight.data *= mask.float()
                
                # Register mask as buffer
                module.register_buffer('weight_mask', mask)
        
        return model
    
    def _prune_structured(self, model: nn.Module) -> nn.Module:
        """Apply structured pruning to model.
        
        Args:
            model: Model to prune
            
        Returns:
            Pruned model
        """
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                weight = module.weight.data
                
                if self.config.granularity == "channel":
                    # Prune entire output channels
                    channel_importance = torch.norm(weight, p=2, dim=1)
                    num_channels_to_prune = int(self.config.pruning_ratio * weight.size(0))
                    
                    _, indices_to_prune = torch.topk(
                        channel_importance,
                        num_channels_to_prune,
                        largest=False
                    )
                    
                    # Zero out pruned channels
                    weight[indices_to_prune, :] = 0
                    
                elif self.config.granularity == "row":
                    # Prune entire rows
                    row_importance = torch.norm(weight, p=2, dim=0)
                    num_rows_to_prune = int(self.config.pruning_ratio * weight.size(1))
                    
                    _, indices_to_prune = torch.topk(
                        row_importance,
                        num_rows_to_prune,
                        largest=False
                    )
                    
                    # Zero out pruned rows
                    weight[:, indices_to_prune] = 0
                
                module.weight.data = weight
        
        return model
    
    def _prune_unstructured(self, model: nn.Module) -> nn.Module:
        """Apply unstructured pruning to model.
        
        Args:
            model: Model to prune
            
        Returns:
            Pruned model
        """
        # Unstructured pruning is similar to magnitude pruning
        return self._prune_magnitude(model)
    
    def iterative_pruning(
        self,
        model: nn.Module,
        target_ratio: float,
        num_iterations: int = 5,
        fine_tune_fn: Optional[Callable] = None
    ) -> nn.Module:
        """Apply iterative pruning with optional fine-tuning.
        
        Args:
            model: Model to prune
            target_ratio: Final pruning ratio to achieve
            num_iterations: Number of pruning iterations
            fine_tune_fn: Optional function to fine-tune model after each iteration
            
        Returns:
            Pruned model
        """
        ratio_per_iteration = target_ratio / num_iterations
        
        for i in range(num_iterations):
            print(f"Pruning iteration {i+1}/{num_iterations}")
            
            # Update pruning ratio for this iteration
            self.config.pruning_ratio = ratio_per_iteration
            
            # Prune model
            model = self.prune_model(model)
            
            # Fine-tune if function provided
            if fine_tune_fn is not None:
                model = fine_tune_fn(model)
        
        return model
    
    @staticmethod
    def get_sparsity(model: nn.Module) -> float:
        """Calculate model sparsity.
        
        Args:
            model: Model to analyze
            
        Returns:
            Sparsity ratio (0.0 to 1.0)
        """
        total_params = 0
        zero_params = 0
        
        for param in model.parameters():
            total_params += param.numel()
            zero_params += (param == 0).sum().item()
        
        sparsity = zero_params / total_params if total_params > 0 else 0.0
        return sparsity
    
    @staticmethod
    def analyze_pruning(model: nn.Module) -> dict:
        """Analyze pruning statistics.
        
        Args:
            model: Model to analyze
            
        Returns:
            Dictionary with pruning statistics
        """
        stats = {
            "total_parameters": 0,
            "zero_parameters": 0,
            "sparsity": 0.0,
            "layer_sparsity": {}
        }
        
        for name, param in model.named_parameters():
            if 'weight' in name:
                total = param.numel()
                zeros = (param == 0).sum().item()
                sparsity = zeros / total if total > 0 else 0.0
                
                stats["total_parameters"] += total
                stats["zero_parameters"] += zeros
                stats["layer_sparsity"][name] = sparsity
        
        stats["sparsity"] = (
            stats["zero_parameters"] / stats["total_parameters"]
            if stats["total_parameters"] > 0 else 0.0
        )
        
        return stats
    
    def remove_pruning_masks(self, model: nn.Module) -> nn.Module:
        """Remove pruning masks and make pruning permanent.
        
        Args:
            model: Model with pruning masks
            
        Returns:
            Model with permanent pruning
        """
        for module in model.modules():
            if hasattr(module, 'weight_mask'):
                # Apply mask permanently
                if isinstance(module, nn.Linear):
                    module.weight.data *= module.weight_mask
                
                # Remove mask buffer
                delattr(module, 'weight_mask')
        
        return model


class PruningScheduler:
    """Scheduler for gradual pruning during training."""
    
    def __init__(
        self,
        initial_sparsity: float = 0.0,
        final_sparsity: float = 0.5,
        begin_step: int = 0,
        end_step: int = 1000,
        frequency: int = 100
    ):
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        self.begin_step = begin_step
        self.end_step = end_step
        self.frequency = frequency
        
    def get_sparsity(self, step: int) -> float:
        """Get target sparsity for current step.
        
        Args:
            step: Current training step
            
        Returns:
            Target sparsity ratio
        """
        if step < self.begin_step:
            return self.initial_sparsity
        elif step >= self.end_step:
            return self.final_sparsity
        else:
            # Linear interpolation
            progress = (step - self.begin_step) / (self.end_step - self.begin_step)
            return self.initial_sparsity + (self.final_sparsity - self.initial_sparsity) * progress
    
    def should_prune(self, step: int) -> bool:
        """Check if pruning should be applied at current step.
        
        Args:
            step: Current training step
            
        Returns:
            True if pruning should be applied
        """
        return (
            step >= self.begin_step and
            step < self.end_step and
            step % self.frequency == 0
        )
