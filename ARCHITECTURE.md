# Architecture Documentation

This document describes the architecture and design decisions of the Ascend-Pangu-Infer framework.

## Overview

Ascend-Pangu-Infer is a modular inference framework designed for efficient deployment of Pangu language models on Huawei Ascend NPU hardware. The framework supports multiple optimization techniques including quantization, pruning, and speculative inference.

## Core Components

### 1. Model Architecture (`pangu_infer/models/`)

#### PanguModel
The main model class implementing the Pangu transformer architecture.

**Key Features:**
- Multi-head self-attention with configurable heads
- Feed-forward networks with GELU activation
- Layer normalization
- Positional embeddings
- KV caching for efficient autoregressive generation

**Architecture:**
```
PanguModel
├── Token Embedding (vocab_size → hidden_size)
├── Position Embedding (max_seq_len → hidden_size)
├── N × PanguBlock
│   ├── LayerNorm
│   ├── PanguAttention (multi-head self-attention)
│   ├── LayerNorm
│   └── PanguMLP (feed-forward network)
├── Final LayerNorm
└── LM Head (hidden_size → vocab_size)
```

#### PanguConfig
Configuration dataclass supporting two model sizes:
- **Pangu-1B**: 24 layers, 2048 hidden size, 16 attention heads
- **Pangu-7B**: 40 layers, 4096 hidden size, 32 attention heads

### 2. Quantization Module (`pangu_infer/quantization/`)

Implements model compression through reduced precision arithmetic.

**Supported Methods:**
- **INT8 Quantization**: Dynamic quantization for linear layers
- **INT4 Quantization**: Simulated via INT8 (requires custom kernels for true INT4)
- **FP16 Quantization**: Half-precision floating point

**Key Classes:**
- `Quantizer`: Main quantization interface
- `QuantizationConfig`: Configuration for quantization parameters
- `QuantizedLinear`: Custom quantized linear layer implementation

**Usage Pattern:**
```python
quantizer = Quantizer(QuantizationConfig(quantization_type=QuantizationType.INT8))
quantized_model = quantizer.quantize_model(model)
```

### 3. Pruning Module (`pangu_infer/pruning/`)

Implements structured and unstructured pruning for model compression.

**Pruning Methods:**
- **Magnitude-based**: Remove weights with smallest absolute values
- **Structured**: Remove entire channels or rows
- **Unstructured**: Remove individual weights

**Key Features:**
- Iterative pruning with fine-tuning
- Sparsity analysis and reporting
- Layer-wise pruning statistics
- Pruning scheduler for gradual pruning

**Usage Pattern:**
```python
pruner = Pruner(PruningConfig(pruning_ratio=0.3))
pruned_model = pruner.prune_model(model)
sparsity = Pruner.get_sparsity(pruned_model)
```

### 4. Speculative Inference (`pangu_infer/speculative/`)

Accelerates text generation using speculative decoding with a draft model.

**How It Works:**
1. **Draft Phase**: Lightweight draft model generates K candidate tokens
2. **Verification Phase**: Target model verifies all candidates in parallel
3. **Acceptance**: Accept tokens that match target model's distribution
4. **Speedup**: Achieve 1.5-2x speedup by processing multiple tokens per step

**Key Components:**
- `DraftModel`: Lightweight version using fewer transformer layers
- `SpeculativeInference`: Main inference engine
- `SpeculativeConfig`: Configuration for speculative parameters

**Architecture:**
```
Input → Draft Model (N/4 layers) → K candidates
         ↓
      Target Model → Verify all K candidates in parallel
         ↓
      Accept/Reject based on probability threshold
```

### 5. Utilities (`pangu_infer/utils/`)

#### Device Management (`device_utils.py`)
- Automatic device selection (NPU → CUDA → CPU)
- Device information and memory tracking
- Synchronization and cache management
- Support for Huawei Ascend NPU via torch_npu

#### Model Loading (`model_loader.py`)
- Checkpoint loading and saving
- Pretrained weight loading with flexible key matching
- Format conversion between PyTorch and MindSpore
- State dict manipulation utilities

## Design Principles

### 1. Modularity
Each optimization technique (quantization, pruning, speculative inference) is implemented as an independent module that can be used separately or combined.

### 2. Hardware Abstraction
The device management layer abstracts hardware differences, supporting:
- Huawei Ascend NPU (primary target)
- NVIDIA CUDA GPUs (fallback)
- CPU (fallback)

### 3. Flexibility
- Configuration-driven design
- Support for both Pangu-1B and Pangu-7B
- Extensible to other model architectures

### 4. Performance
- KV caching for efficient generation
- Batch processing support
- Memory-efficient implementations
- Optimized for Ascend hardware

## Data Flow

### Standard Inference
```
Input IDs → Token Embedding + Position Embedding
          → Dropout
          → N × Transformer Block (Attention + MLP)
          → LayerNorm
          → LM Head
          → Logits
```

### Generation with KV Cache
```
Initial: Full forward pass, cache key/value tensors
Loop:
  → Process only last token
  → Reuse cached key/value
  → Generate next token
  → Update cache
  → Repeat until EOS or max_length
```

### Speculative Inference
```
Draft Model: Generate K candidate tokens sequentially
Target Model: Verify all K candidates in parallel
Compare: Accept tokens matching target distribution
Result: 1 to K accepted tokens per iteration
```

## Memory Management

### Model Memory Breakdown
- **Parameters**: Model weights (largest component)
- **Activations**: Intermediate layer outputs
- **KV Cache**: Cached key/value tensors for generation
- **Gradients**: None (inference only)

### Optimization Strategies
1. **Quantization**: Reduce parameter precision (4x-16x compression)
2. **Pruning**: Remove unnecessary weights (1.5x-3x speedup)
3. **KV Cache**: Store intermediate states (2x-5x faster generation)
4. **Speculative**: Process multiple tokens per step (1.5x-2x speedup)

## Performance Characteristics

### Quantization
- **INT8**: ~4x compression, 0.95x speed, <1% accuracy loss
- **FP16**: ~2x compression, 1.2x speed, minimal accuracy loss
- **INT4**: ~8x compression, 0.8x speed, 2-5% accuracy loss

### Pruning
- **30% unstructured**: 1.1x speedup, 1-2% accuracy loss
- **50% structured**: 1.5x speedup, 3-5% accuracy loss
- **Iterative**: Better accuracy retention

### Speculative Inference
- **Speedup**: 1.5-2x depending on acceptance rate
- **Memory**: +25% for draft model
- **Best for**: Greedy or low-temperature sampling

## Extension Points

### Adding New Model Architectures
1. Implement model class inheriting from `nn.Module`
2. Create corresponding configuration class
3. Implement forward pass and generation methods
4. Add to main package `__init__.py`

### Adding New Quantization Methods
1. Add new `QuantizationType` enum value
2. Implement quantization logic in `Quantizer._quantize_*` method
3. Add calibration if needed
4. Update documentation

### Adding New Pruning Methods
1. Add new `PruningMethod` enum value
2. Implement pruning logic in `Pruner._prune_*` method
3. Add analysis utilities if needed
4. Update documentation

## Dependencies

### Core Dependencies
- **PyTorch**: Deep learning framework
- **NumPy**: Numerical computations

### Optional Dependencies
- **torch_npu**: Ascend NPU support
- **mindspore**: Checkpoint conversion
- **transformers**: Tokenizer utilities
- **pytest**: Testing framework

## Testing Strategy

### Unit Tests
- Model creation and configuration
- Forward pass correctness
- Generation functionality
- Quantization operations
- Pruning operations
- Configuration validation

### Integration Tests
- End-to-end inference
- Optimization combinations
- Device compatibility
- Checkpoint loading/saving

### Performance Tests
- Inference speed benchmarks
- Memory usage profiling
- Compression ratio validation
- Speedup measurements

## Future Enhancements

### Planned Features
- [ ] Distributed inference support
- [ ] More advanced quantization (GPTQ, AWQ)
- [ ] Dynamic pruning during inference
- [ ] Model serving utilities
- [ ] Benchmark suite
- [ ] Profiling tools

### Research Directions
- [ ] Adaptive speculative decoding
- [ ] Mixed pruning strategies
- [ ] Hardware-specific optimizations
- [ ] Low-bit quantization improvements
- [ ] Cache optimization techniques

## References

- Pangu Model: Original architecture specification
- Huawei Ascend: NPU documentation
- PyTorch: Framework documentation
- Speculative Decoding: Research papers on speculative inference
