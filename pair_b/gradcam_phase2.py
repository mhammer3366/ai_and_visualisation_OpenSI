"""Grad-CAM helpers for pair B, Phase II.

Grad-CAM needs a model that maps an image to one score per class. StreetCLIP gives image
and text embeddings, so Scorer puts both together: score = similarity(image, prompt).
"""

from pathlib import Path

import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


class Scorer(torch.nn.Module):
    """Image -> scores for all candidate countries (like the logits of a classifier)."""

    def __init__(self, geo):
        super().__init__()
        self.clip = geo.model
        self.text = geo.text_embeds                      # (n_countries, 768), normalized

    def forward(self, x):
        pooled = self.clip.vision_model(pixel_values=x).pooler_output
        img = self.clip.visual_projection(pooled)
        img = img / img.norm(dim=-1, keepdim=True)
        scale = self.clip.logit_scale.exp()
        return scale * img @ self.text.T


def vit_reshape(t):
    """ViT tokens (batch, 1 + 24*24, 1024) -> (batch, 1024, 24, 24), CLS token dropped."""
    t = t[:, 1:, :]
    side = int(t.shape[1] ** 0.5)
    return t.reshape(t.shape[0], side, side, t.shape[2]).permute(0, 3, 1, 2)


def make_cam(geo):
    """Grad-CAM on the last vision layer (standard choice for ViT models).

    Speed-up: Grad-CAM only needs the gradient at the target layer, not the gradients of the
    model weights. So all weights are frozen except the tiny target layer itself. Then PyTorch
    only keeps the graph for the last layer instead of all 24, which saves most of the GPU memory
    and time. The heatmap stays exactly the same.
    """
    layer = geo.model.vision_model.encoder.layers[-1].layer_norm1
    for p in geo.model.parameters():
        p.requires_grad_(False)
    for p in layer.parameters():
        p.requires_grad_(True)
    return GradCAM(model=Scorer(geo), target_layers=[layer], reshape_transform=vit_reshape)


def heatmaps(cam, geo, images, countries):
    """List of PIL images + list of country ids -> array (n, 336, 336), each map scaled to 0..1."""
    x = geo.prepare(images).to(geo.device, geo.dtype)
    targets = []
    for c in countries:
        targets.append(ClassifierOutputTarget(geo.labels.index(c)))
    return cam(input_tensor=x, targets=targets)


def cached_text_mask(image_id, image_size, data_dir="data"):
    """Text mask from the Phase I cache (EasyOCR), so EasyOCR does not have to be loaded again.
    image_size is (width, height) of the original image. No cache -> empty mask."""
    path = Path(data_dir) / "masks" / f"{image_id}_text.npz"
    if path.exists():
        return np.load(path)["mask"]
    return np.zeros((image_size[1], image_size[0]), dtype=bool)
