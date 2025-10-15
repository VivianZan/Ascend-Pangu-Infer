"""Example of model quantization."""

import torch
from pangu_infer.models import PanguModel, PanguConfig
from pangu_infer.quantization import Quantizer, QuantizationConfig, QuantizationType
from pangu_infer.utils import get_ascend_device


def main():
    # Configuration
    device = get_ascend_device(0)
    print(f"Using device: {device}")
    
    # Create model
    print("Creating model...")
    config = PanguConfig.pangu_1b()
    model = PanguModel(config)
    model.to(device)
    model.eval()
    
    # Measure original model size
    original_size = Quantizer.get_model_size(model)
    print(f"Original model size: {original_size:.2f} MB")
    
    # Configure quantization
    print("\nQuantizing model to INT8...")
    quant_config = QuantizationConfig(
        quantization_type=QuantizationType.INT8,
        per_channel=True,
        symmetric=True,
    )
    
    quantizer = Quantizer(quant_config)
    
    # Quantize model
    quantized_model = quantizer.quantize_model(model)
    quantized_model.to(device)
    quantized_model.eval()
    
    # Measure quantized model size
    quantized_size = Quantizer.get_model_size(quantized_model)
    print(f"Quantized model size: {quantized_size:.2f} MB")
    
    # Compare models
    comparison = quantizer.compare_models(model, quantized_model)
    print(f"\nCompression ratio: {comparison['compression_ratio']:.2f}x")
    print(f"Size reduction: {comparison['size_reduction']:.2f}%")
    
    # Test inference
    print("\nTesting inference on quantized model...")
    input_ids = torch.randint(0, config.vocab_size, (1, 10)).to(device)
    
    with torch.no_grad():
        outputs = quantized_model(input_ids)
        logits = outputs["logits"]
    
    print(f"Output logits shape: {logits.shape}")
    print("Quantization completed successfully!")
    
    # Try FP16 quantization
    print("\n--- FP16 Quantization ---")
    quant_config_fp16 = QuantizationConfig(
        quantization_type=QuantizationType.FP16,
    )
    quantizer_fp16 = Quantizer(quant_config_fp16)
    
    fp16_model = quantizer_fp16.quantize_model(model)
    fp16_size = Quantizer.get_model_size(fp16_model)
    print(f"FP16 model size: {fp16_size:.2f} MB")
    
    comparison_fp16 = quantizer_fp16.compare_models(model, fp16_model)
    print(f"FP16 compression ratio: {comparison_fp16['compression_ratio']:.2f}x")


if __name__ == "__main__":
    main()
