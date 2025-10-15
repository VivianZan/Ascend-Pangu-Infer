"""Example combining multiple optimization techniques."""

import torch
import time
from pangu_infer.models import PanguModel, PanguConfig
from pangu_infer.quantization import Quantizer, QuantizationConfig, QuantizationType
from pangu_infer.pruning import Pruner, PruningConfig, PruningMethod
from pangu_infer.speculative import create_speculative_inference
from pangu_infer.utils import get_ascend_device


def print_section(title):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}\n")


def main():
    # Configuration
    device = get_ascend_device(0)
    print(f"Using device: {device}")
    
    # Create base model (smaller for demo)
    print_section("Creating Base Model")
    config = PanguConfig(
        vocab_size=5000,
        hidden_size=256,
        num_layers=6,
        num_heads=8,
        intermediate_size=1024,
        max_position_embeddings=128,
    )
    
    base_model = PanguModel(config)
    base_model.to(device)
    base_model.eval()
    
    base_size = Quantizer.get_model_size(base_model)
    print(f"Base model size: {base_size:.2f} MB")
    print(f"Parameters: {sum(p.numel() for p in base_model.parameters()):,}")
    
    # Test input
    input_ids = torch.randint(0, config.vocab_size, (1, 10)).to(device)
    
    # Baseline inference
    print_section("Baseline Inference")
    start_time = time.time()
    with torch.no_grad():
        baseline_output = base_model.generate(
            input_ids,
            max_length=30,
            temperature=0.9,
            do_sample=True,
        )
    baseline_time = time.time() - start_time
    print(f"Generation time: {baseline_time:.4f}s")
    print(f"Generated {baseline_output.size(1)} tokens")
    
    # Apply Pruning
    print_section("Applying Pruning (30%)")
    prune_config = PruningConfig(
        pruning_method=PruningMethod.MAGNITUDE,
        pruning_ratio=0.3,
        structured=False,
    )
    pruner = Pruner(prune_config)
    pruned_model = pruner.prune_model(base_model)
    
    sparsity = Pruner.get_sparsity(pruned_model)
    print(f"Model sparsity: {sparsity:.2%}")
    
    # Test pruned model
    start_time = time.time()
    with torch.no_grad():
        pruned_output = pruned_model.generate(
            input_ids,
            max_length=30,
            temperature=0.9,
            do_sample=True,
        )
    pruned_time = time.time() - start_time
    print(f"Pruned generation time: {pruned_time:.4f}s")
    print(f"Speedup: {baseline_time/pruned_time:.2f}x")
    
    # Apply Quantization
    print_section("Applying INT8 Quantization")
    quant_config = QuantizationConfig(
        quantization_type=QuantizationType.INT8,
        per_channel=True,
        symmetric=True,
    )
    quantizer = Quantizer(quant_config)
    quantized_model = quantizer.quantize_model(pruned_model)
    
    quantized_size = Quantizer.get_model_size(quantized_model)
    compression = quantizer.compare_models(base_model, quantized_model)
    
    print(f"Quantized model size: {quantized_size:.2f} MB")
    print(f"Compression ratio: {compression['compression_ratio']:.2f}x")
    print(f"Size reduction: {compression['size_reduction']:.2f}%")
    
    # Test quantized model
    start_time = time.time()
    with torch.no_grad():
        quantized_output = quantized_model(input_ids)
    quantized_time = time.time() - start_time
    print(f"Quantized inference time: {quantized_time:.4f}s")
    
    # Apply Speculative Inference (on base model)
    print_section("Applying Speculative Inference")
    spec_inference = create_speculative_inference(
        base_model,
        draft_layers=2,  # Use 2 layers for draft
        num_speculative_tokens=3,
    )
    
    # Test speculative inference
    start_time = time.time()
    with torch.no_grad():
        spec_output, stats = spec_inference.generate(
            input_ids,
            max_length=30,
            temperature=0.9,
        )
    spec_time = time.time() - start_time
    
    print(f"Speculative generation time: {spec_time:.4f}s")
    print(f"Acceptance rate: {stats['acceptance_rate']:.2%}")
    print(f"Theoretical speedup: {stats['speedup']:.2f}x")
    print(f"Actual speedup: {baseline_time/spec_time:.2f}x")
    
    # Final Summary
    print_section("Optimization Summary")
    
    print("Technique Comparison:")
    print(f"  Baseline:        {baseline_time:.4f}s (1.00x), {base_size:.2f} MB")
    print(f"  + Pruning:       {pruned_time:.4f}s ({baseline_time/pruned_time:.2f}x), {base_size:.2f} MB")
    print(f"  + Quantization:  N/A, {quantized_size:.2f} MB ({compression['compression_ratio']:.2f}x smaller)")
    print(f"  + Speculative:   {spec_time:.4f}s ({baseline_time/spec_time:.2f}x), {base_size:.2f} MB")
    
    print("\nBest for:")
    print(f"  Speed:           Speculative Inference ({baseline_time/spec_time:.2f}x speedup)")
    print(f"  Memory:          Quantization ({compression['compression_ratio']:.2f}x compression)")
    print(f"  Balance:         Pruning + Quantization")
    
    print("\nRecommendations:")
    print("  - Use quantization for memory-constrained deployments")
    print("  - Use pruning for balanced speed/memory optimization")
    print("  - Use speculative inference for latency-critical applications")
    print("  - Combine pruning + quantization for maximum compression")


if __name__ == "__main__":
    main()
