"""Example of speculative inference."""

import torch
import time
from pangu_infer.models import PanguModel, PanguConfig
from pangu_infer.speculative import SpeculativeInference, SpeculativeConfig
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
    
    # Create speculative inference engine
    print("\nInitializing speculative inference...")
    spec_config = SpeculativeConfig(
        draft_model_layers=6,  # Use 6 layers for draft model
        num_speculative_tokens=5,  # Generate 5 tokens speculatively
        acceptance_threshold=0.8,
        use_temperature=True,
        temperature=0.9,
    )
    
    spec_inference = SpeculativeInference(model, spec_config)
    spec_inference.to(device)
    
    # Example input
    input_ids = torch.randint(0, config.vocab_size, (1, 10)).to(device)
    print(f"Input shape: {input_ids.shape}")
    
    # Standard generation
    print("\n--- Standard Generation ---")
    start_time = time.time()
    with torch.no_grad():
        standard_output = model.generate(
            input_ids,
            max_length=50,
            temperature=0.9,
            top_k=50,
            top_p=0.9,
            do_sample=True,
        )
    standard_time = time.time() - start_time
    
    print(f"Standard generation time: {standard_time:.4f}s")
    print(f"Generated {standard_output.size(1)} tokens")
    
    # Speculative generation
    print("\n--- Speculative Generation ---")
    start_time = time.time()
    with torch.no_grad():
        speculative_output, stats = spec_inference.generate(
            input_ids,
            max_length=50,
            temperature=0.9,
            top_k=50,
            top_p=0.9,
        )
    speculative_time = time.time() - start_time
    
    print(f"Speculative generation time: {speculative_time:.4f}s")
    print(f"Generated {speculative_output.size(1)} tokens")
    
    # Statistics
    print("\n--- Speculative Inference Statistics ---")
    print(f"Total tokens: {stats['total_tokens']}")
    print(f"Accepted tokens: {stats['accepted_tokens']}")
    print(f"Rejected tokens: {stats['rejected_tokens']}")
    print(f"Acceptance rate: {stats['acceptance_rate']:.2%}")
    print(f"Theoretical speedup: {stats['speedup']:.2f}x")
    
    # Actual speedup
    if standard_time > 0:
        actual_speedup = standard_time / speculative_time
        print(f"Actual speedup: {actual_speedup:.2f}x")
    
    print("\nSpeculative inference completed successfully!")


if __name__ == "__main__":
    main()
