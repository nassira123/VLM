from typing import Tuple

import open_clip
import torch


def load_openclip(model_name: str, pretrained: str, device: torch.device) -> Tuple[torch.nn.Module, callable, callable]:
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=model_name,
        pretrained=pretrained,
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device)
    model.eval()
    return model, preprocess, tokenizer


def load_openclip_hf(model_id: str, device: torch.device) -> Tuple[torch.nn.Module, callable, callable]:
    model, preprocess = open_clip.create_model_from_pretrained(model_id)
    tokenizer = open_clip.get_tokenizer(model_id)
    model = model.to(device)
    model.eval()
    return model, preprocess, tokenizer
