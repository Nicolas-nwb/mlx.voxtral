"""
LoRA (Low-Rank Adaptation) utilities for fine-tuning Voxtral models.
"""
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import json


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adaptation.

    Implements Low-Rank Adaptation by adding trainable low-rank matrices
    to a frozen linear layer: W = W_frozen + (B @ A) * scale
    """

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.r = r
        self.scale = alpha / r

        # Frozen base linear layer
        self.linear = nn.Linear(input_dims, output_dims, bias=bias)

        # LoRA trainable parameters
        # A: (r, input_dims) - initialized with random values
        # B: (output_dims, r) - initialized with zeros
        scale_init = 1.0 / mx.sqrt(mx.array(input_dims, dtype=mx.float32))
        self.lora_a = mx.random.normal((r, input_dims)) * scale_init
        self.lora_b = mx.zeros((output_dims, r))

        # Dropout layer (applied only during training)
        if dropout > 0.0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None

    def __call__(self, x: mx.array) -> mx.array:
        # Base linear transformation (frozen)
        base_output = self.linear(x)

        # LoRA adaptation: x @ A.T @ B.T * scale
        lora_output = x @ self.lora_a.T  # (batch, seq, r)
        lora_output = lora_output @ self.lora_b.T  # (batch, seq, output_dims)
        lora_output = lora_output * self.scale

        # Apply dropout only during training
        if self.dropout is not None:
            lora_output = self.dropout(lora_output)

        return base_output + lora_output


def inject_lora_layers(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_modules: Optional[List[str]] = None,
) -> Tuple[nn.Module, int]:
    """
    Inject LoRA adapters into a Voxtral model.

    Args:
        model: VoxtralForConditionalGeneration model
        rank: LoRA rank (r)
        alpha: LoRA scaling factor (alpha / r)
        dropout: Dropout probability for LoRA layers
        target_modules: List of module names to target (default: attention projections)

    Returns:
        (modified_model, num_trainable_params)

    Example:
        >>> model = VoxtralForConditionalGeneration.from_pretrained("mistralai/Voxtral-Mini-3B")
        >>> model, num_params = inject_lora_layers(model, rank=8)
        >>> print(f"Trainable parameters: {num_params:,}")
    """
    from mlx.nn import Linear

    if target_modules is None:
        # Target attention projections in the language model
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

    trainable_params = 0

    def _inject_recursive(module, path=""):
        nonlocal trainable_params

        # Use MLX's .items() to get sub-modules (works for both dict and list attributes)
        try:
            module_items = list(module.items())
        except (AttributeError, TypeError):
            module_items = []

        for name, child in module_items:
            full_path = f"{path}.{name}" if path else name

            # Handle lists/tuples of modules (e.g., self.layers = [Block(), ...])
            if isinstance(child, (list, tuple)):
                for idx, submodule in enumerate(child):
                    subpath = f"{full_path}.{idx}"
                    _inject_recursive(submodule, subpath)
                continue

            # Check if this module should be replaced
            if any(target in name for target in target_modules):
                if isinstance(child, Linear):
                    # Create LoRA layer
                    lora_layer = LoRALinear(
                        input_dims=child.weight.shape[1],
                        output_dims=child.weight.shape[0],
                        r=rank,
                        alpha=alpha,
                        dropout=dropout,
                        bias=child.bias is not None,
                    )

                    # Copy existing weights (frozen)
                    lora_layer.linear.weight = child.weight
                    if child.bias is not None:
                        lora_layer.linear.bias = child.bias

                    # Replace in parent module
                    setattr(module, name, lora_layer)

                    # Count trainable LoRA parameters
                    trainable_params += rank * (
                        child.weight.shape[0] + child.weight.shape[1]
                    )

                    print(f"✓ Injected LoRA: {full_path} (rank={rank})")

            # Recurse into non-leaf modules
            if hasattr(child, "children"):
                _inject_recursive(child, full_path)

    # Inject LoRA layers first
    _inject_recursive(model)

    # Now freeze all parameters
    model.freeze()

    # Unfreeze only LoRA parameters (lora_a and lora_b)
    # Must unfreeze recursively on each LoRA module
    def unfreeze_lora_recursive(module):
        """Recursively unfreeze LoRA parameters in all LoRA layers."""
        unfrozen_count = 0

        # Use MLX's .items() to get sub-modules
        try:
            module_items = list(module.items())
        except (AttributeError, TypeError):
            module_items = []

        for name, child in module_items:
            # Handle lists/tuples of modules
            if isinstance(child, (list, tuple)):
                for submodule in child:
                    unfrozen_count += unfreeze_lora_recursive(submodule)
            elif isinstance(child, LoRALinear):
                # Unfreeze local LoRA parameters in this specific module
                child.unfreeze(keys=["lora_a", "lora_b"])
                unfrozen_count += 2
            elif hasattr(child, "children"):
                unfrozen_count += unfreeze_lora_recursive(child)
        return unfrozen_count

    unfrozen = unfreeze_lora_recursive(model)
    if unfrozen > 0:
        print(f"✓ Unfrozen {unfrozen} LoRA parameters")

    return model, trainable_params


def compute_loss(
    model: nn.Module,
    input_features: mx.array,
    input_ids: mx.array,
    labels: mx.array,
    attention_mask: Optional[mx.array] = None,
) -> mx.array:
    """
    Compute loss for a batch during training.

    Args:
        model: Model with LoRA adapters
        input_features: Audio features [batch, n_chunks, n_mels, n_frames]
        input_ids: Token IDs [batch, seq_len] - input prompt with audio tokens
        labels: Token IDs [batch, seq_len] - target transcription tokens
        attention_mask: Attention mask [batch, seq_len]

    Returns:
        Scalar loss tensor
    """
    # Forward pass
    outputs = model(
        input_ids=input_ids,
        input_features=input_features,
        attention_mask=attention_mask,
    )

    logits = outputs.logits if hasattr(outputs, "logits") else outputs

    # Shift for language modeling (predict next token)
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    # Cross-entropy with ignore_index=-100 to mask prompt tokens
    loss = nn.losses.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
        reduction="none",
    )

    # Reshape loss back to sequence shape
    loss = loss.reshape(shift_labels.shape)

    # Create mask: ignore positions where labels == -100
    valid_mask = (shift_labels != -100).astype(mx.float32)

    # Apply mask and compute mean only over valid positions
    masked_loss = loss * valid_mask
    num_valid = valid_mask.sum()

    if num_valid > 0:
        return masked_loss.sum() / num_valid
    else:
        # Fallback if no valid labels (shouldn't happen)
        return loss.mean()


def train_step(
    model: nn.Module,
    optimizer: optim.Optimizer,
    batch: Dict[str, mx.array],
) -> float:
    """
    Execute one training step.

    Args:
        model: Model with LoRA adapters
        optimizer: MLX optimizer
        batch: Dict with input_features, input_ids, labels, attention_mask

    Returns:
        Loss value (float)
    """

    def loss_fn(mdl):
        return compute_loss(
            mdl,
            batch["input_features"],
            batch["input_ids"],
            batch["labels"],
            batch.get("attention_mask"),
        )

    # Forward + backward using nn.value_and_grad (correct API for modules)
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    loss, grads = loss_and_grad_fn(model)

    # Update weights
    optimizer.update(model, grads)

    # Evaluate to get scalar value
    mx.eval(model.parameters(), optimizer.state)

    return loss.item()


def save_lora_adapters(
    model: nn.Module,
    output_dir: Union[str, Path],
    metadata: Optional[Dict] = None,
):
    """
    Save only LoRA adapter weights.

    Args:
        model: Model with LoRA adapters
        output_dir: Output directory path
        metadata: Optional metadata (config, metrics, etc.)

    Example:
        >>> save_lora_adapters(
        ...     model,
        ...     "lora_adapters/",
        ...     metadata={"rank": 8, "final_loss": 0.42}
        ... )
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract only LoRA weights (must flatten nested dict and lists)
    def flatten_params(params, prefix=""):
        """Flatten nested parameter dict, handling lists/tuples."""
        flat = {}
        for key, value in params.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                flat.update(flatten_params(value, full_key))
            elif isinstance(value, (list, tuple)):
                # Handle lists of modules (e.g., layers)
                for idx, item in enumerate(value):
                    if isinstance(item, dict):
                        flat.update(flatten_params(item, f"{full_key}.{idx}"))
                    else:
                        flat[f"{full_key}.{idx}"] = item
            else:
                flat[full_key] = value
        return flat

    flat_params = flatten_params(model.parameters())
    lora_weights = {
        k: v for k, v in flat_params.items() if "lora" in k.lower()
    }

    # Save
    mx.save_safetensors(str(output_dir / "lora_adapters.safetensors"), lora_weights)

    # Save metadata
    if metadata:
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    print(f"✓ LoRA adapters saved to {output_dir}")


def load_lora_adapters(model: nn.Module, lora_dir: Union[str, Path]) -> nn.Module:
    """
    Load LoRA adapter weights into a model.

    Args:
        model: Base model (must have matching architecture)
        lora_dir: Directory containing lora_adapters.safetensors

    Returns:
        Model with loaded LoRA adapters

    Example:
        >>> model = VoxtralForConditionalGeneration.from_pretrained("mistralai/Voxtral-Mini-3B")
        >>> model = load_lora_adapters(model, "lora_adapters/")
    """
    lora_path = Path(lora_dir) / "lora_adapters.safetensors"

    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA file not found: {lora_path}")

    # Load weights (flattened dict with dot-separated keys like "q_proj.lora_a")
    lora_weights_flat = mx.load(str(lora_path))

    # Reconstruct nested structure from flattened keys
    # e.g., "language_model.layers.0.q_proj.lora_a" -> nested dict
    def unflatten_dict(flat_dict):
        """Reconstruct nested dict/list structure from dot-separated keys.

        Example: "layers.0.q_proj.lora_a" -> {"layers": [{"q_proj": {"lora_a": value}}]}
        """
        nested = {}

        for key, value in flat_dict.items():
            parts = key.split(".")
            current = nested

            # Navigate/create the path
            for i, part in enumerate(parts[:-1]):
                next_part = parts[i + 1]

                if part.isdigit():
                    # Current part is an index, parent should already be a list
                    idx = int(part)
                    while len(current) <= idx:
                        current.append({})
                    current = current[idx]
                elif next_part.isdigit():
                    # Next part is numeric → create/get list
                    if part not in current:
                        current[part] = []
                    current = current[part]
                else:
                    # Next part is a string → create/get dict
                    if part not in current:
                        current[part] = {}
                    current = current[part]

            # Set the final value
            final_key = parts[-1]
            if isinstance(current, list):
                # Final key is an index into a list
                idx = int(final_key)
                while len(current) <= idx:
                    current.append(None)
                current[idx] = value
            else:
                current[final_key] = value

        return nested

    lora_weights = unflatten_dict(lora_weights_flat)

    # Apply to model
    model.update(lora_weights)

    print(f"✓ LoRA adapters loaded from {lora_dir}")
    return model
