"""Basic tests for Pangu inference framework."""

import torch
import pytest
from pangu_infer.models import PanguModel, PanguConfig
from pangu_infer.quantization import Quantizer, QuantizationConfig, QuantizationType
from pangu_infer.pruning import Pruner, PruningConfig, PruningMethod


def test_model_creation():
    """Test basic model creation."""
    config = PanguConfig.pangu_1b()
    model = PanguModel(config)
    assert model is not None
    assert isinstance(model, torch.nn.Module)


def test_model_forward():
    """Test model forward pass."""
    config = PanguConfig(
        vocab_size=1000,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        intermediate_size=512,
    )
    model = PanguModel(config)
    model.eval()
    
    input_ids = torch.randint(0, 1000, (2, 10))
    
    with torch.no_grad():
        outputs = model(input_ids)
    
    assert "logits" in outputs
    assert outputs["logits"].shape == (2, 10, 1000)


def test_model_generation():
    """Test text generation."""
    config = PanguConfig(
        vocab_size=1000,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        intermediate_size=512,
    )
    model = PanguModel(config)
    model.eval()
    
    input_ids = torch.randint(0, 1000, (1, 5))
    
    with torch.no_grad():
        generated = model.generate(input_ids, max_length=20, do_sample=False)
    
    assert generated.shape[0] == 1
    assert generated.shape[1] <= 20


def test_quantization():
    """Test model quantization."""
    config = PanguConfig(
        vocab_size=1000,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        intermediate_size=512,
    )
    model = PanguModel(config)
    
    quant_config = QuantizationConfig(
        quantization_type=QuantizationType.INT8,
    )
    quantizer = Quantizer(quant_config)
    
    quantized_model = quantizer.quantize_model(model)
    assert quantized_model is not None


def test_pruning():
    """Test model pruning."""
    config = PanguConfig(
        vocab_size=1000,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        intermediate_size=512,
    )
    model = PanguModel(config)
    
    prune_config = PruningConfig(
        pruning_method=PruningMethod.MAGNITUDE,
        pruning_ratio=0.3,
    )
    pruner = Pruner(prune_config)
    
    pruned_model = pruner.prune_model(model)
    sparsity = Pruner.get_sparsity(pruned_model)
    
    assert sparsity > 0.0
    assert pruned_model is not None


def test_pangu_configs():
    """Test predefined configurations."""
    config_1b = PanguConfig.pangu_1b()
    assert config_1b.hidden_size == 2048
    assert config_1b.num_layers == 24
    
    config_7b = PanguConfig.pangu_7b()
    assert config_7b.hidden_size == 4096
    assert config_7b.num_layers == 40


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
