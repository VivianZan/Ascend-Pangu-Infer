"""Speculative inference for accelerated generation."""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple, List


@dataclass
class SpeculativeConfig:
    """Configuration for speculative inference.
    
    Args:
        draft_model_layers: Number of layers for draft model
        num_speculative_tokens: Number of tokens to generate speculatively
        acceptance_threshold: Threshold for accepting speculative tokens
        use_temperature: Whether to use temperature sampling
        temperature: Temperature for sampling
    """
    draft_model_layers: int = 8
    num_speculative_tokens: int = 5
    acceptance_threshold: float = 0.8
    use_temperature: bool = True
    temperature: float = 1.0


class DraftModel(nn.Module):
    """Lightweight draft model for speculative decoding."""
    
    def __init__(self, target_model: nn.Module, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        
        # Copy embeddings from target model
        self.token_embedding = target_model.token_embedding
        self.position_embedding = target_model.position_embedding
        self.dropout = target_model.dropout
        
        # Use first N layers from target model
        self.blocks = nn.ModuleList(target_model.blocks[:num_layers])
        
        # Copy output layers
        self.ln_f = target_model.ln_f
        self.lm_head = target_model.lm_head
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: bool = False,
    ):
        batch_size, seq_length = input_ids.size()
        device = input_ids.device
        
        # Get position IDs
        if past_key_values is not None:
            position_ids = torch.arange(
                past_key_values[0][0].size(2),
                seq_length + past_key_values[0][0].size(2),
                dtype=torch.long,
                device=device,
            )
        else:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Compute embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        hidden_states = token_embeds + position_embeds
        hidden_states = self.dropout(hidden_states)
        
        # Prepare attention mask
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min
        
        # Apply transformer blocks
        presents = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            past_key_value = past_key_values[i] if past_key_values is not None else None
            hidden_states, present = block(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )
            if use_cache:
                presents.append(present)
        
        # Apply final layer norm
        hidden_states = self.ln_f(hidden_states)
        
        # Compute logits
        logits = self.lm_head(hidden_states)
        
        return {
            "logits": logits,
            "past_key_values": tuple(presents) if use_cache else None,
        }


class SpeculativeInference:
    """Speculative inference engine for accelerated generation.
    
    Uses a smaller draft model to generate candidate tokens speculatively,
    which are then verified by the target model in parallel.
    """
    
    def __init__(
        self,
        target_model: nn.Module,
        config: SpeculativeConfig
    ):
        self.target_model = target_model
        self.config = config
        
        # Create draft model
        self.draft_model = DraftModel(target_model, config.draft_model_layers)
        
        # Ensure models are in eval mode
        self.target_model.eval()
        self.draft_model.eval()
        
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 128,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> Tuple[torch.Tensor, dict]:
        """Generate text using speculative decoding.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_length]
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            
        Returns:
            Tuple of (generated_ids, statistics)
        """
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # Statistics
        stats = {
            "total_tokens": 0,
            "accepted_tokens": 0,
            "rejected_tokens": 0,
            "acceptance_rate": 0.0,
            "speedup": 0.0,
        }
        
        past_key_values_target = None
        past_key_values_draft = None
        
        generated_tokens = 0
        
        while input_ids.size(1) < max_length:
            # Phase 1: Draft model generates speculative tokens
            draft_tokens = []
            draft_logits_list = []
            
            draft_input = input_ids if past_key_values_draft is None else input_ids[:, -1:]
            
            for _ in range(self.config.num_speculative_tokens):
                outputs = self.draft_model(
                    draft_input,
                    past_key_values=past_key_values_draft,
                    use_cache=True,
                )
                
                logits = outputs["logits"][:, -1, :] / temperature
                past_key_values_draft = outputs["past_key_values"]
                
                # Sample next token
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                draft_tokens.append(next_token)
                draft_logits_list.append(logits)
                draft_input = next_token
            
            # Concatenate draft tokens
            draft_sequence = torch.cat(draft_tokens, dim=-1)
            
            # Phase 2: Target model verifies all draft tokens in parallel
            verify_input = torch.cat([input_ids, draft_sequence], dim=-1)
            
            outputs = self.target_model(
                verify_input if past_key_values_target is None else verify_input[:, -len(draft_tokens)-1:],
                past_key_values=past_key_values_target,
                use_cache=True,
            )
            
            target_logits = outputs["logits"]
            past_key_values_target = outputs["past_key_values"]
            
            # Phase 3: Compare and accept/reject tokens
            accepted_tokens = 0
            for i in range(len(draft_tokens)):
                # Get target distribution for this position
                target_probs = torch.softmax(target_logits[:, i, :] / temperature, dim=-1)
                draft_token = draft_tokens[i]
                
                # Check if draft token is acceptable
                target_prob = target_probs[0, draft_token.item()]
                
                if target_prob >= self.config.acceptance_threshold:
                    # Accept token
                    accepted_tokens += 1
                    stats["accepted_tokens"] += 1
                else:
                    # Reject token and resample from target
                    new_token = torch.multinomial(target_probs, num_samples=1)
                    draft_tokens[i] = new_token
                    stats["rejected_tokens"] += 1
                    break
            
            # Update input_ids with accepted tokens
            if accepted_tokens > 0:
                accepted_sequence = torch.cat(draft_tokens[:accepted_tokens], dim=-1)
                input_ids = torch.cat([input_ids, accepted_sequence], dim=-1)
                generated_tokens += accepted_tokens
            else:
                # If no tokens accepted, use target's next token
                target_probs = torch.softmax(target_logits[:, 0, :] / temperature, dim=-1)
                next_token = torch.multinomial(target_probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                generated_tokens += 1
            
            stats["total_tokens"] = generated_tokens
            
            # Check for EOS
            if (input_ids[:, -1] == self.target_model.config.eos_token_id).all():
                break
        
        # Calculate statistics
        if stats["total_tokens"] > 0:
            stats["acceptance_rate"] = stats["accepted_tokens"] / (
                stats["accepted_tokens"] + stats["rejected_tokens"]
            ) if (stats["accepted_tokens"] + stats["rejected_tokens"]) > 0 else 0.0
            
            # Estimate speedup (rough approximation)
            # Speedup = tokens generated / equivalent sequential steps
            sequential_steps = stats["total_tokens"]
            speculative_steps = sequential_steps / (1 + stats["acceptance_rate"] * (self.config.num_speculative_tokens - 1))
            stats["speedup"] = sequential_steps / speculative_steps if speculative_steps > 0 else 1.0
        
        return input_ids, stats
    
    def to(self, device):
        """Move models to device.
        
        Args:
            device: Target device
        """
        self.target_model.to(device)
        self.draft_model.to(device)
        return self


def create_speculative_inference(
    target_model: nn.Module,
    draft_layers: Optional[int] = None,
    num_speculative_tokens: int = 5
) -> SpeculativeInference:
    """Create a speculative inference engine.
    
    Args:
        target_model: Target model for generation
        draft_layers: Number of layers for draft model (default: 1/4 of target)
        num_speculative_tokens: Number of tokens to generate speculatively
        
    Returns:
        SpeculativeInference instance
    """
    if draft_layers is None:
        # Use 1/4 of target model layers by default
        total_layers = len(target_model.blocks)
        draft_layers = max(1, total_layers // 4)
    
    config = SpeculativeConfig(
        draft_model_layers=draft_layers,
        num_speculative_tokens=num_speculative_tokens,
    )
    
    return SpeculativeInference(target_model, config)
