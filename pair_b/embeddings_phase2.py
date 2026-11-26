"""Embedding helpers for pair B, Phase II (text side of StreetCLIP)."""

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def embed_texts(geo, prompts):
    """List of texts -> normalized text embeddings (numpy, n x 768).
    Same steps as GeoCLIPModel.set_candidates, but without changing the model's candidates."""
    tokens = geo.tokenizer(list(prompts), padding=True, return_tensors="pt").to(geo.device)
    pooled = geo.model.text_model(**tokens).pooler_output
    emb = geo.model.text_projection(pooled)
    return F.normalize(emb.float(), dim=-1).cpu().numpy()


def normalize(x):
    """Rows of x scaled to length 1, so a dot product is the cosine similarity."""
    return x / np.linalg.norm(x, axis=1, keepdims=True)


# Two-letter codes for the plotly US map
US_STATE_CODES = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
    "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "Florida": "FL", "Georgia": "GA",
    "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA",
    "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH",
    "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY", "North Carolina": "NC",
    "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA",
    "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN",
    "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY", "District of Columbia": "DC",
}
