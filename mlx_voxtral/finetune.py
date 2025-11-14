#!/usr/bin/env python3
"""
Fine-tune Voxtral models with LoRA using MLX.

Usage:
    python -m mlx_voxtral.finetune \
        --model mistralai/Voxtral-Mini-3B \
        --dataset dataset.json \
        --output lora_adapters/ \
        --epochs 3

Note: Actuellement seul --batch-size 1 est supporté (valeur par défaut).
"""
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
import mlx.core as mx
import mlx.optimizers as optim
from tqdm import tqdm

from mlx_voxtral import VoxtralForConditionalGeneration, VoxtralProcessor
from mlx_voxtral.lora import (
    inject_lora_layers,
    train_step,
    save_lora_adapters,
)


def load_dataset(dataset_path: Path) -> List[Dict[str, Any]]:
    """
    Load dataset from JSON file.

    Expected format:
    [
        {"audio_path": "path/to/audio1.wav", "text": "transcription"},
        {"audio_path": "path/to/audio2.wav", "text": "another transcription"},
        ...
    ]
    """
    with open(dataset_path) as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Dataset must be a JSON array")

    return data


def create_dataloader(
    dataset: List[Dict[str, Any]],
    processor: VoxtralProcessor,
    dataset_root: Path,
    batch_size: int = 1,
    shuffle: bool = True,
):
    """
    Lazy generator that yields batches for training one at a time.

    Args:
        dataset: Liste d'échantillons {audio_path, text}
        processor: VoxtralProcessor pour le traitement audio/texte
        dataset_root: Répertoire racine pour résoudre les chemins relatifs
        batch_size: Taille du batch (actuellement ignoré, fixé à 1)
        shuffle: Mélanger les échantillons

    Yields:
        Dict with input_features, input_ids, labels, attention_mask (all mx.array)
    """
    import numpy as np

    indices = list(range(len(dataset)))
    if shuffle:
        np.random.shuffle(indices)

    # Process one sample at a time (batch_size=1 for now due to variable audio lengths)
    for idx in indices:
        sample = dataset[idx]
        audio_path_str = sample["audio_path"]
        text = sample["text"]

        # Resolve paths: URLs unchanged, absolute paths unchanged, relative paths from dataset dir
        if audio_path_str.startswith(("http://", "https://")):
            # Keep URLs unchanged
            resolved_audio_path = audio_path_str
        else:
            # Handle local file paths
            audio_path = Path(audio_path_str)
            if audio_path.is_absolute():
                # Absolute paths used as-is
                resolved_audio_path = str(audio_path)
            else:
                # Try dataset_root first, fallback to current directory for compatibility
                dataset_relative = (dataset_root / audio_path).resolve()
                if dataset_relative.exists():
                    resolved_audio_path = str(dataset_relative)
                else:
                    # Fallback: interpret relative to CWD (backward compatibility)
                    resolved_audio_path = audio_path_str

        # Process audio to get prompt with audio tokens
        # Note: language=None omits lang: prefix for better alignment with inference
        transcription_inputs = processor.apply_transcrition_request(
            resolved_audio_path, language=None
        )

        input_features = transcription_inputs.input_features
        input_ids = transcription_inputs.input_ids  # [INST][BEGIN_AUDIO][AUDIO_TOKENS][/INST][TRANSCRIBE]

        # Tokenize target transcription text
        text_encoding = processor.tokenizer(
            text, return_tensors="np", padding=False, add_special_tokens=False
        )
        text_tokens = text_encoding["input_ids"][0]  # Remove batch dim

        # Build full sequence: prompt + transcription + EOS
        # This is teacher forcing: model sees ground truth tokens during training
        eos_token = processor._special_token_ids.get('eos', 2)
        prompt_tokens = input_ids[0].tolist() if hasattr(input_ids[0], 'tolist') else np.array(input_ids[0])

        full_sequence = np.concatenate([
            prompt_tokens,
            text_tokens,
            [eos_token]
        ])

        # input_ids = full sequence (teacher forcing)
        input_ids_full = mx.array([full_sequence], dtype=mx.int32)

        # labels = full sequence, but we'll mask the prompt part during loss computation
        # Use -100 for tokens we don't want to compute loss on (prompt tokens)
        labels_sequence = np.array(full_sequence, dtype=np.int32)

        # Mask prompt tokens: set to -100 to ignore in loss
        # Only compute loss on transcription tokens (after [TRANSCRIBE] token)
        transcribe_token = processor._special_token_ids.get('transcribe', 34)

        # Find position of [TRANSCRIBE] token
        transcribe_pos = -1
        for i, token in enumerate(prompt_tokens):
            if token == transcribe_token:
                transcribe_pos = i
                break

        # Mask all tokens up to and including [TRANSCRIBE]
        if transcribe_pos >= 0:
            labels_sequence[:transcribe_pos + 1] = -100
        else:
            # Fallback: mask entire prompt
            labels_sequence[:len(prompt_tokens)] = -100

        labels = mx.array([labels_sequence], dtype=mx.int32)

        yield {
            "input_features": input_features,
            "input_ids": input_ids_full,
            "labels": labels,
            # Note: attention_mask=None uses default causal mask in model
        }


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Voxtral with LoRA"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Voxtral-Mini-3B",
        help="Model name or path",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to dataset JSON file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for LoRA adapters",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size (currently supports 1)"
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument(
        "--lora-rank", type=int, default=8, help="LoRA rank"
    )
    parser.add_argument(
        "--lora-alpha", type=float, default=16.0, help="LoRA alpha"
    )
    args = parser.parse_args()

    # Fail fast: batch size > 1 not yet supported
    if args.batch_size != 1:
        raise ValueError(
            f"Batch size {args.batch_size} not supported. "
            f"Current implementation only supports --batch-size 1 due to variable-length audio chunks. "
            f"Use gradient accumulation for larger effective batch sizes."
        )

    print("🚀 Loading model...")
    model = VoxtralForConditionalGeneration.from_pretrained(args.model)
    processor = VoxtralProcessor.from_pretrained(args.model)

    print("🔧 Injecting LoRA adapters...")
    model, num_params = inject_lora_layers(
        model, rank=args.lora_rank, alpha=args.lora_alpha
    )
    print(f"✓ {num_params:,} trainable parameters")

    print("📚 Loading dataset...")
    dataset = load_dataset(args.dataset)
    print(f"✓ {len(dataset)} samples")

    optimizer = optim.AdamW(learning_rate=args.lr)

    print(f"\n🏋️ Starting fine-tuning ({args.epochs} epochs)...")
    for epoch in range(args.epochs):
        total_loss = 0
        num_batches = 0

        # Create lazy dataloader (generator, not materialized list)
        dataloader = create_dataloader(
            dataset, processor, args.dataset.parent, args.batch_size, shuffle=True
        )

        # Use tqdm directly on generator (no list() materialization)
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}", total=len(dataset))
        for batch in pbar:
            loss = train_step(model, optimizer, batch)
            total_loss += loss
            num_batches += 1

            pbar.set_postfix({"loss": f"{loss:.4f}"})

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")

    print(f"\n💾 Saving LoRA adapters...")
    save_lora_adapters(
        model,
        args.output,
        metadata={
            "base_model": args.model,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "epochs": args.epochs,
            "final_loss": avg_loss,
        },
    )

    print("✅ Fine-tuning complete!")


if __name__ == "__main__":
    main()
