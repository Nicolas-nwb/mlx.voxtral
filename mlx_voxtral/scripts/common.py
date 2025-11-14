"""Fonctions utilitaires partagées par les scripts CLI."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download


ESSENTIAL_FILES = [
    "generation_config.json",
    "preprocessor_config.json",
    "tekken.json",
    "params.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.model",
    "*.tiktoken",
]


def resolve_model_assets_path(model_identifier: str) -> Path:
    """Retourne le dossier local contenant les poids/tokenizers d'un modèle."""
    candidate = Path(model_identifier)
    if candidate.exists():
        return candidate
    return Path(snapshot_download(repo_id=model_identifier))


def copy_support_files(model_path: Path, output_path: Path, logger: logging.Logger) -> None:
    """Copie les fichiers processeur/tokenizer requis vers le dossier cible."""
    for pattern in ESSENTIAL_FILES:
        for file in model_path.glob(pattern):
            if not file.is_file():
                continue
            shutil.copy2(file, output_path / file.name)
            logger.info("Copied %s", file.name)
    for file in model_path.glob("*.py"):
        if file.is_file():
            shutil.copy2(file, output_path / file.name)
            logger.info("Copied %s", file.name)

