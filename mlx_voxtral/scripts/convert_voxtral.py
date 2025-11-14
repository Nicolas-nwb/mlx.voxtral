#!/usr/bin/env python3
"""Conversion de checkpoints Voxtral vers le format safetensors MLX attendu."""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import mlx.core as mx

from mlx_voxtral import load_voxtral_model
from mlx_voxtral.quantization import (
    compute_bits_per_weight,
    quantize_model,
    save_config,
    save_model,
    voxtral_mixed_quantization_predicate,
)

from .common import copy_support_files, resolve_model_assets_path


def parse_args() -> argparse.Namespace:
    """Construit l'interface CLI pour la conversion."""
    parser = argparse.ArgumentParser(
        description=(
            "Convertit un checkpoint Voxtral Hugging Face vers un dossier MLX"
            " (float16/bfloat16) avec option de quantification 4/8 bits."
        )
    )
    parser.add_argument("model", type=str, help="Identifiant HF ou chemin local")
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        required=True,
        help="Dossier cible pour le checkpoint MLX",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float16", "bfloat16"],
        default="bfloat16",
        help="Précision des poids sauvegardés",
    )
    parser.add_argument(
        "--bits",
        type=int,
        choices=[4, 8],
        help="Active une quantification MLX (4 ou 8 bits)",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=64,
        help="Taille de groupe pour la quantification (64 recommandé)",
    )
    parser.add_argument(
        "--mixed",
        action="store_true",
        help="Active la stratégie de quantification mixte Voxtral",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Supprime le dossier de sortie s'il existe déjà",
    )
    return parser.parse_args()


def build_quant_predicate(args, config):
    """Retourne le prédicat de sélection des couches à quantifier."""

    if args.mixed:
        def predicate(path, module):
            return voxtral_mixed_quantization_predicate(
                path,
                module,
                config,
                default_bits=args.bits,
            )

        return predicate

    def predicate(path, module):
        if "embed_positions" in path or "pos_emb" in path:
            return False
        if not hasattr(module, "to_quantized"):
            return False
        return True

    return predicate


def main() -> None:
    args = parse_args()
    if args.mixed and args.bits is None:
        raise ValueError("L'option --mixed requiert --bits 4 ou 8")

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    output_path = Path(args.output_dir)
    if output_path.exists():
        if not args.overwrite:
            raise ValueError(f"Output directory {output_path} already exists")
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    dtype_map = {
        "float16": mx.float16,
        "bfloat16": mx.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    logger.info("Loading model from %s", args.model)
    model, config = load_voxtral_model(args.model, dtype=dtype, lazy=True)

    quantized_model = model
    target_config = dict(config)

    if args.bits is not None:
        logger.info(
            "Applying MLX quantization (%s bits, group_size=%s)",
            args.bits,
            args.group_size,
        )
        predicate = build_quant_predicate(args, target_config)
        quantized_model, target_config = quantize_model(
            model,
            target_config,
            group_size=args.group_size,
            bits=args.bits,
            quant_predicate=predicate,
        )
    else:
        target_config.pop("quantization", None)

    logger.info("Saving MLX checkpoint to %s", output_path)
    save_model(output_path, quantized_model, donate_model=True)
    save_config(target_config, output_path / "config.json")

    model_path = resolve_model_assets_path(args.model)
    logger.info("Copying tokenizer and processor files...")
    copy_support_files(model_path, output_path, logger)

    if args.bits is not None:
        try:
            bits_per_weight = compute_bits_per_weight(quantized_model)
            logger.info("Average bits/weight: %.3f", bits_per_weight)
        except Exception as exc:  # pragma: no cover - indicatif seulement
            logger.info("Could not compute bits/weight: %s", exc)

    logger.info("✅ Conversion complete")


if __name__ == "__main__":
    main()

