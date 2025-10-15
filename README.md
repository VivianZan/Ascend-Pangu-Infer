# Ascend-Pangu-Infer

A comprehensive inference framework for Pangu-1B/7B models optimized for Huawei Ascend NPU, with support for quantization, pruning, and speculative inference.

## Features

- 🚀 **High-Performance Inference**: Optimized for Huawei Ascend NPU with fallback support for CUDA and CPU
- 📦 **Model Quantization**: Support for INT8, INT4, and FP16 quantization to reduce model size
- ✂️ **Model Pruning**: Magnitude-based, structured, and unstructured pruning for model compression
- ⚡ **Speculative Inference**: Accelerated text generation using speculative decoding
- 🔧 **Flexible Configuration**: Easy-to-use configuration system for both Pangu-1B and Pangu-7B
- 🔄 **Checkpoint Conversion**: Convert between PyTorch and MindSpore checkpoint formats

## Installation

### Prerequisites

- Python >= 3.8
- PyTorch >= 2.0.0
- (Optional) Ascend CANN toolkit for NPU support
- (Optional) MindSpore >= 2.0.0 for checkpoint conversion

### Install from source

```bash
git clone https://github.com/VivianZan/Ascend-Pangu-Infer.git
cd Ascend-Pangu-Infer
pip install -e .
```

### Install dependencies

```bash
pip install -r requirements.txt
```

For Ascend NPU support, install the torch_npu package:
```bash
# Follow official Ascend PyTorch installation guide
# https://www.hiascend.com/software/cann
```

## Quick Start

### Basic Inference

```python
import torch
from pangu_infer.models import PanguModel, PanguConfig
from pangu_infer.utils import get_ascend_device

# Get device (Ascend NPU, CUDA, or CPU)
device = get_ascend_device(0)

# Create model configuration (Pangu-1B)
config = PanguConfig.pangu_1b()

# Initialize model
model = PanguModel(config)
model.to(device)
model.eval()

# Generate text
input_ids = torch.randint(0, config.vocab_size, (1, 10)).to(device)
generated_ids = model.generate(
    input_ids,
    max_length=50,
    temperature=0.9,
    top_k=50,
    top_p=0.9,
)
```

### Model Quantization

```python
from pangu_infer.quantization import Quantizer, QuantizationConfig, QuantizationType

# Configure INT8 quantization
quant_config = QuantizationConfig(
    quantization_type=QuantizationType.INT8,
    per_channel=True,
    symmetric=True,
)

# Create quantizer and quantize model
quantizer = Quantizer(quant_config)
quantized_model = quantizer.quantize_model(model)

# Compare model sizes
comparison = quantizer.compare_models(model, quantized_model)
print(f"Compression ratio: {comparison['compression_ratio']:.2f}x")
```

### Model Pruning

```python
from pangu_infer.pruning import Pruner, PruningConfig, PruningMethod

# Configure magnitude-based pruning
prune_config = PruningConfig(
    pruning_method=PruningMethod.MAGNITUDE,
    pruning_ratio=0.3,  # Prune 30% of weights
    structured=False,
)

# Create pruner and prune model
pruner = Pruner(prune_config)
pruned_model = pruner.prune_model(model)

# Analyze pruning results
sparsity = Pruner.get_sparsity(pruned_model)
print(f"Model sparsity: {sparsity:.2%}")
```

### Speculative Inference

```python
from pangu_infer.speculative import SpeculativeInference, SpeculativeConfig

# Configure speculative inference
spec_config = SpeculativeConfig(
    draft_model_layers=6,  # Use 6 layers for draft model
    num_speculative_tokens=5,  # Generate 5 tokens speculatively
    acceptance_threshold=0.8,
)

# Create speculative inference engine
spec_inference = SpeculativeInference(model, spec_config)

# Generate with speculative decoding
generated_ids, stats = spec_inference.generate(
    input_ids,
    max_length=50,
    temperature=0.9,
)

print(f"Acceptance rate: {stats['acceptance_rate']:.2%}")
print(f"Speedup: {stats['speedup']:.2f}x")
```

## Model Configurations

### Pangu-1B

- Hidden size: 2048
- Number of layers: 24
- Number of attention heads: 16
- Vocabulary size: 40,000
- Max sequence length: 1024

### Pangu-7B

- Hidden size: 4096
- Number of layers: 40
- Number of attention heads: 32
- Vocabulary size: 40,000
- Max sequence length: 1024

## Examples

The `examples/` directory contains complete examples:

- `basic_inference.py` - Basic model inference
- `quantization_example.py` - INT8/FP16 quantization
- `pruning_example.py` - Magnitude and structured pruning
- `speculative_inference_example.py` - Speculative decoding

Run an example:
```bash
python examples/basic_inference.py
```

## Project Structure

```
Ascend-Pangu-Infer/
├── pangu_infer/              # Main package
│   ├── models/               # Model implementations
│   │   ├── pangu_model.py    # Pangu model architecture
│   │   └── config.py         # Model configurations
│   ├── quantization/         # Quantization module
│   │   └── quantizer.py      # Quantization utilities
│   ├── pruning/              # Pruning module
│   │   └── pruner.py         # Pruning utilities
│   ├── speculative/          # Speculative inference
│   │   └── speculative_inference.py
│   └── utils/                # Utility functions
│       ├── device_utils.py   # Device management
│       └── model_loader.py   # Model loading/saving
├── examples/                 # Example scripts
├── tests/                    # Unit tests
├── requirements.txt          # Python dependencies
├── setup.py                  # Package setup
└── README.md                 # This file
```

## Advanced Usage

### Loading Pretrained Weights

```python
# Load from checkpoint
model = PanguModel.from_pretrained("path/to/checkpoint.pth", config)

# Or using utility function
from pangu_infer.utils import load_checkpoint
model = load_checkpoint(model, "path/to/checkpoint.pth", device)
```

### Checkpoint Conversion

```python
from pangu_infer.utils import convert_checkpoint_format

# Convert MindSpore checkpoint to PyTorch
convert_checkpoint_format(
    input_path="pangu_mindspore.ckpt",
    output_path="pangu_pytorch.pth",
    source_format="mindspore",
    target_format="pytorch"
)
```

### Device Management

```python
from pangu_infer.utils import get_device_info, get_device_count

# Get device information
info = get_device_info(0)
print(f"Device: {info['device_name']}")
print(f"Memory: {info['total_memory'] / 1e9:.2f} GB")

# Get number of available devices
num_devices = get_device_count()
print(f"Available devices: {num_devices}")
```

## Performance Tips

1. **Use Quantization**: INT8 quantization can reduce model size by ~4x with minimal accuracy loss
2. **Enable KV Caching**: Set `use_cache=True` for faster sequential generation
3. **Batch Inference**: Process multiple sequences together for better throughput
4. **Speculative Inference**: Use for 1.5-2x speedup in text generation tasks
5. **Pruning + Quantization**: Combine pruning and quantization for maximum compression

## Benchmark Results

| Model | Method | Size | Speedup | Memory |
|-------|--------|------|---------|--------|
| Pangu-1B | Baseline | ~4GB | 1.0x | ~4GB |
| Pangu-1B | INT8 Quant | ~1GB | 0.95x | ~1GB |
| Pangu-1B | 30% Pruning | ~4GB | 1.1x | ~3GB |
| Pangu-1B | Speculative | ~4GB | 1.8x | ~5GB |

*Note: Benchmarks are approximate and depend on hardware and workload*

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Huawei Ascend team for the NPU platform
- Original Pangu model authors
- PyTorch and MindSpore communities

## Citation

If you use this project in your research, please cite:

```bibtex
@software{ascend_pangu_infer,
  title={Ascend-Pangu-Infer: Inference Framework for Pangu Models},
  author={Ascend Pangu Inference Team},
  year={2025},
  url={https://github.com/VivianZan/Ascend-Pangu-Infer}
}
```

## Contact

For questions and feedback, please open an issue on GitHub.