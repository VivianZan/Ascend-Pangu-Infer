"""Example of model pruning."""

import torch
from pangu_infer.models import PanguModel, PanguConfig
from pangu_infer.pruning import Pruner, PruningConfig, PruningMethod
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
    
    # Analyze original model
    original_sparsity = Pruner.get_sparsity(model)
    print(f"Original model sparsity: {original_sparsity:.4f}")
    
    # Configure pruning
    print("\nPruning model with magnitude-based pruning...")
    prune_config = PruningConfig(
        pruning_method=PruningMethod.MAGNITUDE,
        pruning_ratio=0.3,  # Prune 30% of weights
        structured=False,
    )
    
    pruner = Pruner(prune_config)
    
    # Prune model
    pruned_model = pruner.prune_model(model)
    
    # Analyze pruned model
    pruned_sparsity = Pruner.get_sparsity(pruned_model)
    print(f"Pruned model sparsity: {pruned_sparsity:.4f}")
    
    # Get detailed statistics
    stats = Pruner.analyze_pruning(pruned_model)
    print(f"\nTotal parameters: {stats['total_parameters']}")
    print(f"Zero parameters: {stats['zero_parameters']}")
    print(f"Overall sparsity: {stats['sparsity']:.4f}")
    
    # Show per-layer sparsity for first few layers
    print("\nPer-layer sparsity (first 5 layers):")
    for i, (name, sparsity) in enumerate(list(stats['layer_sparsity'].items())[:5]):
        print(f"  {name}: {sparsity:.4f}")
    
    # Test inference
    print("\nTesting inference on pruned model...")
    input_ids = torch.randint(0, config.vocab_size, (1, 10)).to(device)
    
    with torch.no_grad():
        outputs = pruned_model(input_ids)
        logits = outputs["logits"]
    
    print(f"Output logits shape: {logits.shape}")
    
    # Try structured pruning
    print("\n--- Structured Pruning ---")
    prune_config_structured = PruningConfig(
        pruning_method=PruningMethod.STRUCTURED,
        pruning_ratio=0.2,  # Prune 20% of channels
        structured=True,
        granularity="channel",
    )
    
    pruner_structured = Pruner(prune_config_structured)
    structured_pruned_model = pruner_structured.prune_model(PanguModel(config).to(device))
    
    structured_sparsity = Pruner.get_sparsity(structured_pruned_model)
    print(f"Structured pruned model sparsity: {structured_sparsity:.4f}")
    
    print("\nPruning completed successfully!")


if __name__ == "__main__":
    main()
