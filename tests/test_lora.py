"""
Unit tests for LoRA fine-tuning functionality.
"""
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import pytest
import tempfile
import json
from pathlib import Path


def flatten_params(params, prefix=""):
    """Flatten nested parameter dict to get all leaf tensors, handling lists/tuples."""
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


def create_dummy_model():
    """Create a simple dummy model for testing."""

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(10, 20)
            self.q_proj = nn.Linear(20, 20)
            self.v_proj = nn.Linear(20, 20)
            self.output = nn.Linear(20, 10)

        def __call__(self, x):
            x = self.linear1(x)
            x = self.q_proj(x)
            x = self.v_proj(x)
            x = self.output(x)
            return x

    return DummyModel()


def test_inject_lora_basic():
    """Test basic LoRA injection into a model."""
    from mlx_voxtral.lora import inject_lora_layers, LoRALinear

    model = create_dummy_model()

    # Inject LoRA
    model, num_params = inject_lora_layers(
        model, rank=4, alpha=8.0, target_modules=["q_proj", "v_proj"]
    )

    assert num_params > 0
    assert isinstance(num_params, int)

    # Check that target modules have been replaced with LoRALinear
    assert isinstance(model.q_proj, LoRALinear)
    assert isinstance(model.v_proj, LoRALinear)

    # Non-target modules should remain as Linear
    assert isinstance(model.linear1, nn.Linear)
    assert isinstance(model.output, nn.Linear)


def test_lora_trainable_params():
    """Test that LoRA parameters exist in the model."""
    from mlx_voxtral.lora import inject_lora_layers

    model = create_dummy_model()
    model, num_params = inject_lora_layers(model, rank=4)

    # Check that LoRA parameters were injected (must flatten nested dict)
    all_params = flatten_params(model.parameters())

    # LoRA parameters should exist
    lora_params = [k for k in all_params.keys() if "lora" in k.lower()]
    assert len(lora_params) > 0, "LoRA parameters should be injected"

    # Base model parameters should still exist
    base_params = [k for k in all_params.keys() if "lora" not in k.lower()]
    assert len(base_params) > 0, "Base model parameters should be preserved"

    # Trainable params count should be positive
    assert num_params > 0


def test_save_and_load_lora_adapters():
    """Test saving and loading LoRA adapters."""
    from mlx_voxtral.lora import inject_lora_layers, save_lora_adapters, load_lora_adapters

    model = create_dummy_model()
    model, _ = inject_lora_layers(model, rank=4)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "lora_test"

        # Save
        metadata = {"rank": 4, "alpha": 8.0, "test": True}
        save_lora_adapters(model, output_dir, metadata=metadata)

        # Check files exist
        assert (output_dir / "lora_adapters.safetensors").exists()
        assert (output_dir / "metadata.json").exists()

        # Load metadata
        with open(output_dir / "metadata.json") as f:
            loaded_metadata = json.load(f)

        assert loaded_metadata["rank"] == 4
        assert loaded_metadata["test"] is True

        # Load adapters into new model
        new_model = create_dummy_model()
        new_model, _ = inject_lora_layers(new_model, rank=4)
        new_model = load_lora_adapters(new_model, output_dir)

        # Models should have same LoRA parameters (must flatten)
        original_lora = {
            k: v for k, v in flatten_params(model.parameters()).items() if "lora" in k.lower()
        }
        loaded_lora = {
            k: v for k, v in flatten_params(new_model.parameters()).items() if "lora" in k.lower()
        }

        assert len(original_lora) == len(loaded_lora)

        # Verify that loaded values match original values
        for key in original_lora.keys():
            assert key in loaded_lora, f"Key {key} missing in loaded model"
            assert mx.allclose(original_lora[key], loaded_lora[key]), \
                f"Values mismatch for {key}"


def test_compute_loss():
    """Test loss computation function with dummy model."""
    from mlx_voxtral.lora import compute_loss

    # Create a simple model that returns logits
    class DummyLMModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 50)  # 50 = vocab size

        def __call__(self, input_ids, input_features=None, attention_mask=None):
            # Simulate model output (input_features and attention_mask unused in dummy)
            _ = (input_features, attention_mask)  # Mark as intentionally unused
            batch_size, seq_len = input_ids.shape
            logits = mx.random.normal((batch_size, seq_len, 50))

            class Output:
                def __init__(self, logits):
                    self.logits = logits

            return Output(logits)

    model = DummyLMModel()

    # Create dummy batch
    _input_features = mx.random.normal((1, 1, 128, 3000))
    input_ids = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)
    labels = mx.array([[-100, -100, 3, 4, 5]], dtype=mx.int32)  # Mask first 2 tokens

    # Compute loss
    loss = compute_loss(model, _input_features, input_ids, labels)

    # Check that loss is a scalar
    assert loss.shape == ()
    assert not mx.isnan(loss)
    loss_value = float(loss)
    assert loss_value >= 0  # Loss should be positive


def test_train_step():
    """Test single training step with dummy model."""
    from mlx_voxtral.lora import inject_lora_layers, train_step

    # Create a model that can actually compute loss
    class DummyLMModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(10, 10)
            self.v_proj = nn.Linear(10, 10)
            self.output_layer = nn.Linear(10, 50)

        def __call__(self, input_ids, input_features=None, attention_mask=None):
            # Simulate model processing (input_features and attention_mask unused)
            _ = (input_features, attention_mask)  # Mark as intentionally unused
            batch_size, seq_len = input_ids.shape
            hidden = mx.random.normal((batch_size, seq_len, 10))
            hidden = self.q_proj(hidden)
            hidden = self.v_proj(hidden)
            logits = self.output_layer(hidden)

            class Output:
                def __init__(self, logits):
                    self.logits = logits

            return Output(logits)

    model = DummyLMModel()
    model, _ = inject_lora_layers(model, rank=4)

    optimizer = optim.SGD(learning_rate=0.01)

    # Create dummy batch
    batch = {
        "input_features": mx.random.normal((1, 1, 128, 3000)),
        "input_ids": mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32),
        "labels": mx.array([[-100, -100, 3, 4, 5]], dtype=mx.int32),
        "attention_mask": mx.ones((1, 5)),
    }

    # Execute training step
    loss_value = train_step(model, optimizer, batch)

    # Check that loss is valid
    assert isinstance(loss_value, float)
    assert loss_value >= 0
    assert not mx.isnan(mx.array(loss_value))


def test_lora_forward_pass():
    """Test that LoRA model can do forward pass."""
    from mlx_voxtral.lora import inject_lora_layers

    model = create_dummy_model()
    model, _ = inject_lora_layers(model, rank=4)

    # Forward pass
    x = mx.random.normal((2, 10))
    output = model(x)

    assert output.shape == (2, 10)
    assert not mx.any(mx.isnan(output))


def test_lora_gradient_flow():
    """Test that gradients flow through LoRA layers."""
    from mlx_voxtral.lora import inject_lora_layers

    model = create_dummy_model()
    model, _ = inject_lora_layers(model, rank=4)

    def loss_fn(mdl):
        x = mx.random.normal((2, 10))
        output = mdl(x)
        return mx.mean(output**2)

    # Compute gradients using nn.value_and_grad
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    _loss, grads = loss_and_grad_fn(model)

    # Check that LoRA parameters have gradients (must flatten)
    flat_grads = flatten_params(grads)
    lora_grads = {k: v for k, v in flat_grads.items() if "lora" in k.lower()}
    assert len(lora_grads) > 0

    # Check that at least some gradients are non-zero
    # Note: lora_a may have zero gradient on first iteration since lora_b=0 initially
    has_nonzero = any(not mx.all(grad == 0) for grad in lora_grads.values())
    assert has_nonzero, "At least some LoRA gradients should be non-zero"


def test_lora_different_ranks():
    """Test LoRA with different rank values."""
    from mlx_voxtral.lora import inject_lora_layers

    for rank in [2, 4, 8, 16]:
        model = create_dummy_model()
        model, num_params = inject_lora_layers(model, rank=rank)

        # Number of trainable parameters should increase with rank
        assert num_params > 0


def test_lora_with_dropout():
    """Test LoRA injection with dropout."""
    from mlx_voxtral.lora import inject_lora_layers

    model = create_dummy_model()
    model, _ = inject_lora_layers(model, rank=4, dropout=0.1)

    # Model should still work
    x = mx.random.normal((2, 10))
    output = model(x)

    assert output.shape == (2, 10)


def test_lora_parameter_count():
    """Test that LoRA reduces trainable parameter count."""
    from mlx_voxtral.lora import inject_lora_layers

    model = create_dummy_model()

    # Count original parameters (must flatten nested dict)
    flat_params = flatten_params(model.parameters())
    original_params = sum(
        int(p.size) for p in flat_params.values()
    )

    model, lora_trainable = inject_lora_layers(model, rank=4)

    # LoRA trainable parameters should be much smaller than total original parameters
    assert lora_trainable < original_params
