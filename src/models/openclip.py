"""OpenCLIP model loaders."""

from __future__ import annotations

from typing import Tuple

import open_clip
import torch

BIOMEDCLIP_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


def _freeze(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def load_clip_openai_vitb32(device: torch.device) -> Tuple[torch.nn.Module, callable, callable]:
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name="ViT-B-32",
        pretrained="openai",
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.to(device)
    model.eval()
    _freeze(model)
    return model, preprocess, tokenizer


def load_biomedclip_hf(device: torch.device) -> Tuple[torch.nn.Module, callable, callable]:
    model, preprocess = open_clip.create_model_from_pretrained(BIOMEDCLIP_ID)
    tokenizer = open_clip.get_tokenizer(BIOMEDCLIP_ID)
    model.to(device)
    model.eval()
    _freeze(model)
    return model, preprocess, tokenizer


def encode_text(model: torch.nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        features = model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
    return features


def encode_image(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        features = model.encode_image(images)
        features = features / features.norm(dim=-1, keepdim=True)
    return features
