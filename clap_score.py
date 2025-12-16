"""
LAION-CLAP utilities for scoring and embedding extraction.

This file is intentionally based on the existing `clap_score.py` workflow used in the project:
- Audio is loaded at 48kHz mono (librosa), peak-normalized, then quantized via int16 round-trip.
- Text embeddings and audio embeddings are extracted with LAION-CLAP.
- CLAP score is cosine similarity between paired (audio, text) embeddings.

Default model: **630k-audioset-fusion-best.pt** (fusion-best).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import numpy as np
import requests
import torch
from tqdm import tqdm

import laion_clap
import librosa
import pyloudnorm as pyln


CLAP_MODEL_NAME = Literal[
    "music_speech_audioset_epoch_15_esc_89.98.pt",
    "music_audioset_epoch_15_esc_90.14.pt",
    "music_speech_epoch_15_esc_89.25.pt",
    "630k-audioset-fusion-best.pt",
]


def int16_to_float32(x: np.ndarray) -> np.ndarray:
    return (x / 32767.0).astype(np.float32)


def float32_to_int16(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, a_min=-1.0, a_max=1.0)
    return (x * 32767.0).astype(np.int16)


def _download_if_needed(url: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading CLAP checkpoint to {dest} ...")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))
    with dest.open("wb") as f, tqdm(total=total_size, unit="B", unit_scale=True, desc=dest.name) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if not chunk:
                continue
            f.write(chunk)
            pbar.update(len(chunk))


def load_clap_model(
    clap_model_name: CLAP_MODEL_NAME = "630k-audioset-fusion-best.pt",
    *,
    device: str = "cuda",
    cache_dir: str | Path = "load/clap_score",
) -> laion_clap.CLAP_Module:
    """
    Load a LAION-CLAP model checkpoint (with caching).
    Default is fusion-best: 630k-audioset-fusion-best.pt
    """
    cache_dir = Path(cache_dir)

    if clap_model_name == "music_speech_audioset_epoch_15_esc_89.98.pt":
        url = "https://huggingface.co/lukewys/laion_clap/resolve/main/music_speech_audioset_epoch_15_esc_89.98.pt"
        ckpt_path = cache_dir / clap_model_name
        model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base", device=device)
    elif clap_model_name == "music_audioset_epoch_15_esc_90.14.pt":
        url = "https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt"
        ckpt_path = cache_dir / clap_model_name
        model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base", device=device)
    elif clap_model_name == "music_speech_epoch_15_esc_89.25.pt":
        url = "https://huggingface.co/lukewys/laion_clap/resolve/main/music_speech_epoch_15_esc_89.25.pt"
        ckpt_path = cache_dir / clap_model_name
        model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base", device=device)
    elif clap_model_name == "630k-audioset-fusion-best.pt":
        url = "https://huggingface.co/lukewys/laion_clap/resolve/main/630k-audioset-fusion-best.pt"
        ckpt_path = cache_dir / clap_model_name
        model = laion_clap.CLAP_Module(enable_fusion=True, device=device)
    else:
        raise ValueError(f"Unsupported clap_model_name: {clap_model_name}")

    _download_if_needed(url, ckpt_path)

    # Patch position_ids issue for older checkpoints / transformer versions.
    # laion-clap provides `clap_module.factory.load_state_dict`.
    from clap_module.factory import load_state_dict

    pkg = load_state_dict(str(ckpt_path))
    pkg.pop("text_branch.embeddings.position_ids", None)
    model.model.load_state_dict(pkg)

    model.eval()
    model.to(device)
    return model


def compute_clap_embeddings(
    model: laion_clap.CLAP_Module,
    *,
    text_list: list[str],
    audio_path_list: list[str],
    batch_size: int = 32,
    device: str = "cuda",
    audio_sr: int = 48000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute aligned CLAP embeddings.

    Returns:
    - audio_emb: [N, D] torch.Tensor on CPU
    - text_emb : [N, D] torch.Tensor on CPU
    """
    if len(text_list) != len(audio_path_list):
        raise ValueError(f"text_list length ({len(text_list)}) != audio_path_list length ({len(audio_path_list)})")
    if len(text_list) == 0:
        raise ValueError("Empty input: text_list/audio_path_list are empty.")

    model.eval()
    model.to(device)

    # --- Text embeddings ---
    text_emb_batches: list[torch.Tensor] = []
    for i in tqdm(range(0, len(text_list), batch_size), desc="CLAP text embeddings"):
        batch_texts = text_list[i : i + batch_size]
        with torch.no_grad():
            emb = model.get_text_embedding(batch_texts, use_tensor=True).to(device)
        text_emb_batches.append(emb)
    text_emb = torch.cat(text_emb_batches, dim=0)  # [N, D]

    # --- Audio embeddings ---
    audio_emb_batches: list[torch.Tensor] = []
    for i in tqdm(range(0, len(audio_path_list), batch_size), desc="CLAP audio embeddings"):
        batch_paths = audio_path_list[i : i + batch_size]

        wav_tensors: list[torch.Tensor] = []
        max_len = 0
        for p in batch_paths:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Audio file not found: {p}")

            audio, _ = librosa.load(p, sr=audio_sr, mono=True)
            audio = pyln.normalize.peak(audio, -1.0)
            audio = int16_to_float32(float32_to_int16(audio))  # quantization round-trip

            t = torch.from_numpy(audio).float().to(device)  # [T]
            max_len = max(max_len, int(t.shape[0]))
            wav_tensors.append(t)

        # Pad to the max length and stack to [B, T]
        padded: list[torch.Tensor] = []
        for t in wav_tensors:
            if t.shape[0] < max_len:
                t = torch.nn.functional.pad(t, (0, max_len - t.shape[0]))
            padded.append(t)
        audio_batch = torch.stack(padded, dim=0)

        with torch.no_grad():
            emb = model.get_audio_embedding_from_data(x=audio_batch, use_tensor=True).to(device)  # [B, D]
        audio_emb_batches.append(emb)

    audio_emb = torch.cat(audio_emb_batches, dim=0)  # [N, D]

    return audio_emb.detach().cpu(), text_emb.detach().cpu()


def compute_clap_score(
    model: laion_clap.CLAP_Module,
    text_list: list[str],
    audio_path_list: list[str],
    *,
    batch_size: int = 32,
    device: str = "cuda",
    audio_sr: int = 48000,
) -> list[float]:
    """
    Compute CLAP cosine similarity scores for paired (text, audio).
    """
    audio_emb, text_emb = compute_clap_embeddings(
        model,
        text_list=text_list,
        audio_path_list=audio_path_list,
        batch_size=batch_size,
        device=device,
        audio_sr=audio_sr,
    )
    sim = torch.nn.functional.cosine_similarity(audio_emb, text_emb, dim=1, eps=1e-8)
    return sim.tolist()


