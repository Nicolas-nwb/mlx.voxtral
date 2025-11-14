"""
Unit tests for batch processing functionality.
"""
import mlx.core as mx
import numpy as np
import pytest
import tempfile
from pathlib import Path


def test_preprocess_batch_with_numpy_arrays():
    """Test batch preprocessing with numpy arrays."""
    from mlx_voxtral import VoxtralProcessor

    processor = VoxtralProcessor()

    # Create 3 dummy audios of different lengths
    audio1 = np.random.randn(16000 * 2).astype(np.float32)  # 2 seconds
    audio2 = np.random.randn(16000 * 3).astype(np.float32)  # 3 seconds
    audio3 = np.random.randn(16000 * 1).astype(np.float32)  # 1 second

    results = processor.preprocess_batch(
        [audio1, audio2, audio3], language="fr", return_tensors="np"
    )

    # Should return list of results
    assert isinstance(results, list)
    assert len(results) == 3

    # Check each result
    for result in results:
        assert "input_features" in result

        input_features = result["input_features"]

        # Should be 3D: [n_chunks, n_mels, n_frames]
        assert input_features.ndim == 3
        assert input_features.shape[1] == 128  # n_mels
        assert input_features.shape[2] == 3000  # n_frames

        # Should have at least 1 chunk
        assert input_features.shape[0] >= 1


def test_preprocess_batch_with_mlx_tensors():
    """Test batch preprocessing with MLX tensor output."""
    from mlx_voxtral import VoxtralProcessor

    processor = VoxtralProcessor()

    # Create dummy audios
    audio1 = np.random.randn(16000 * 2).astype(np.float32)
    audio2 = np.random.randn(16000 * 2).astype(np.float32)

    results = processor.preprocess_batch(
        [audio1, audio2], language="en", return_tensors="mlx"
    )

    # Check that results is a list
    assert isinstance(results, list)
    assert len(results) == 2

    # Check that each result contains MLX arrays
    mlx_array_type = type(mx.array(0))
    for result in results:
        assert isinstance(result["input_features"], mlx_array_type)
        assert hasattr(result["input_features"], "shape")


def test_preprocess_batch_with_file_paths():
    """Test batch preprocessing with actual audio files."""
    from mlx_voxtral import VoxtralProcessor
    import soundfile as sf

    processor = VoxtralProcessor()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create temporary audio files
        audio_paths = []
        for i in range(3):
            audio_path = Path(tmpdir) / f"audio_{i}.wav"
            audio_data = np.random.randn(16000 * (i + 1)).astype(np.float32)
            sf.write(audio_path, audio_data, 16000)
            audio_paths.append(str(audio_path))

        results = processor.preprocess_batch(
            audio_paths, language="auto", return_tensors="mlx"
        )

        assert isinstance(results, list)
        assert len(results) == 3

        mlx_array_type = type(mx.array(0))
        for result in results:
            assert isinstance(result["input_features"], mlx_array_type)
            assert hasattr(result["input_features"], "shape")


def test_encoder_backward_compatibility():
    """Test that encoder still works with 3D single-sample input."""
    from mlx_voxtral.modeling_voxtral import VoxtralEncoder
    from mlx_voxtral.configuration_voxtral import VoxtralEncoderConfig

    config = VoxtralEncoderConfig()
    encoder = VoxtralEncoder(config)

    # 3D input: [batch=1, n_mels=128, seq_len=3000]
    input_features = mx.random.normal((1, 128, 3000))

    # Should not crash
    output, _, _ = encoder(input_features)

    assert output.ndim == 3
    assert output.shape[0] == 1  # batch size preserved


def test_encoder_batch_processing():
    """Test encoder with batch size > 1."""
    from mlx_voxtral.modeling_voxtral import VoxtralEncoder
    from mlx_voxtral.configuration_voxtral import VoxtralEncoderConfig

    config = VoxtralEncoderConfig()
    encoder = VoxtralEncoder(config)

    # Batch input: [batch=4, n_mels=128, seq_len=3000]
    input_features = mx.random.normal((4, 128, 3000))

    # Should not crash
    output, _, _ = encoder(input_features)

    assert output.ndim == 3
    assert output.shape[0] == 4  # batch size preserved
    assert output.shape[1] > 0  # sequence length
    assert output.shape[2] == config.hidden_size  # hidden size


def test_attention_mask_batching():
    """Test that attention masks work correctly with batches."""
    from mlx_voxtral.modeling_voxtral import VoxtralEncoder
    from mlx_voxtral.configuration_voxtral import VoxtralEncoderConfig

    config = VoxtralEncoderConfig()
    encoder = VoxtralEncoder(config)

    batch_size = 2
    seq_len = 1500  # After conv layers from 3000 input

    input_features = mx.random.normal((batch_size, 128, 3000))

    # Create attention mask with some padding
    attention_mask = mx.ones((batch_size, seq_len))
    attention_mask[1, 1000:] = 0  # Mask second half of second sample

    output, _, _ = encoder(input_features, attention_mask=attention_mask)

    assert output.shape[0] == batch_size
    # Verify output is not NaN
    assert not mx.any(mx.isnan(output))
