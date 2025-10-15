# Examples

This directory contains example scripts demonstrating various features of the Ascend-Pangu-Infer framework.

## Available Examples

### 1. Basic Inference (`basic_inference.py`)

Demonstrates basic model loading and text generation.

```bash
python examples/basic_inference.py
```

**Features:**
- Model initialization
- Simple forward pass
- Text generation with sampling

### 2. Quantization (`quantization_example.py`)

Shows how to quantize models for reduced memory footprint.

```bash
python examples/quantization_example.py
```

**Features:**
- INT8 quantization
- FP16 quantization
- Model size comparison
- Compression ratio analysis

### 3. Pruning (`pruning_example.py`)

Demonstrates model pruning techniques for optimization.

```bash
python examples/pruning_example.py
```

**Features:**
- Magnitude-based pruning
- Structured pruning
- Sparsity analysis
- Per-layer statistics

### 4. Speculative Inference (`speculative_inference_example.py`)

Shows accelerated text generation using speculative decoding.

```bash
python examples/speculative_inference_example.py
```

**Features:**
- Draft model creation
- Speculative token generation
- Acceptance rate tracking
- Speedup measurement

### 5. Combined Optimization (`combined_optimization.py`)

Demonstrates using multiple optimization techniques together.

```bash
python examples/combined_optimization.py
```

**Features:**
- Baseline performance measurement
- Progressive optimization application
- Performance comparison across techniques
- Best practice recommendations

## Running Examples

All examples can be run directly:

```bash
cd Ascend-Pangu-Infer
python examples/<example_name>.py
```

## Customization

Each example can be easily modified:

1. **Change model size**: Switch between Pangu-1B and Pangu-7B
   ```python
   config = PanguConfig.pangu_7b()  # Instead of pangu_1b()
   ```

2. **Adjust generation parameters**:
   ```python
   generated = model.generate(
       input_ids,
       max_length=100,      # Longer generation
       temperature=0.7,     # Less random
       top_k=40,            # Narrower selection
       top_p=0.95,          # Different nucleus sampling
   )
   ```

3. **Modify quantization settings**:
   ```python
   quant_config = QuantizationConfig(
       quantization_type=QuantizationType.INT4,  # More aggressive
       per_channel=False,
       symmetric=False,
   )
   ```

4. **Adjust pruning ratio**:
   ```python
   prune_config = PruningConfig(
       pruning_method=PruningMethod.STRUCTURED,
       pruning_ratio=0.5,  # Prune 50% of weights
       granularity="channel",
   )
   ```

## Expected Output

Each example will print progress information and results. For example:

```
Using device: npu:0
Initializing model...
Generating text...
Generated shape: torch.Size([1, 50])
Inference completed successfully!
```

## Notes

- Examples use CPU by default if Ascend NPU is not available
- Random token IDs are used for demonstration (in practice, use a proper tokenizer)
- Generation quality depends on model initialization (random weights vs. pretrained)
- Performance metrics are approximate and hardware-dependent
