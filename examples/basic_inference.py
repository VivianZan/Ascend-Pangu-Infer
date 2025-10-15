"""Basic inference example for Pangu models."""

import torch
from pangu_infer.models import PanguModel, PanguConfig
from pangu_infer.utils import get_ascend_device


def main():
    # Configuration
    device = get_ascend_device(0)
    print(f"Using device: {device}")
    
    # Create model configuration
    # For Pangu-1B
    config = PanguConfig.pangu_1b()
    # For Pangu-7B, use: config = PanguConfig.pangu_7b()
    
    # Initialize model
    print("Initializing model...")
    model = PanguModel(config)
    model.to(device)
    model.eval()
    
    # Example input
    input_text = "Once upon a time"
    # In practice, you would use a tokenizer here
    # For this example, we use random token IDs
    input_ids = torch.randint(0, config.vocab_size, (1, 10)).to(device)
    
    print(f"Input shape: {input_ids.shape}")
    
    # Generate text
    print("\nGenerating text...")
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            max_length=50,
            temperature=0.9,
            top_k=50,
            top_p=0.9,
            do_sample=True,
        )
    
    print(f"Generated shape: {generated_ids.shape}")
    print(f"Generated tokens: {generated_ids[0].tolist()}")
    
    # Simple forward pass (for logits)
    print("\nForward pass...")
    with torch.no_grad():
        outputs = model(input_ids, use_cache=False)
        logits = outputs["logits"]
    
    print(f"Logits shape: {logits.shape}")
    print("Inference completed successfully!")


if __name__ == "__main__":
    main()
