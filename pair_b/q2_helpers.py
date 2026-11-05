"""Helper functions for Q2 (pair B). The notebook only calls these."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from shared.models import fit_image

# Prompt names that also mean something else (US state, part of another country, sibling country)
NAME_CLASHES = {("US", "GE"), ("US", "MX"), ("GB", "IE"),
                ("GN", "GW"), ("GN", "GQ"), ("GN", "PG"), ("GW", "GN"), ("GQ", "GN"),
                ("NE", "NG"), ("NG", "NE"), ("DM", "DO"), ("DO", "DM"),
                ("CG", "CD"), ("CD", "CG"), ("SD", "SS"), ("SS", "SD"), ("KR", "KP"), ("KP", "KR")}


def model_view(dataset, image_id, crop_mode):
    """The image as the model sees it (fitted to 336x336)."""
    return fit_image(Image.open(dataset.image_paths[image_id]).convert("RGB"), crop_mode)


def distance_km(attrs, a, b):
    """Great-circle distance between the centres of two countries."""
    lat = pd.to_numeric(attrs["center_lat"], errors="coerce")
    lng = pd.to_numeric(attrs["center_lng"], errors="coerce")
    lat1, lng1, lat2, lng2 = map(np.radians, [lat[a], lng[a], lat[b], lng[b]])
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(h))


def relation_features(work, attrs):
    """Relation between the true country and the 'other' country (wrong guess, or 2nd guess if correct)."""
    work = work.copy()
    work["other"] = np.where(work["is_error"], work["pred_1"], work["pred_2"])
    pairs = list(zip(work["country_id"], work["other"]))
    work["is_neighbor"] = [o in attrs.loc[t, "neighbors"].split(";") for t, o in pairs]
    work["same_subregion"] = [attrs.loc[t, "world_subregion"] == attrs.loc[o, "world_subregion"] for t, o in pairs]
    work["distance_km"] = [distance_km(attrs, t, o) for t, o in pairs]
    work["name_clash"] = [(t, o) in NAME_CLASHES for t, o in pairs]
    return work


def image_stats(img):
    """Simple pixel statistics of one model-view image."""
    rgb = np.asarray(img).astype(float)
    gray = rgb.mean(axis=2)
    h = gray.shape[0] // 3
    return {
        "brightness": gray.mean(),                              # low = dark
        "contrast": gray.std(),                                 # low = fog / haze / flat light
        "bright_share": (gray > 240).mean(),                    # glare, overexposed sky
        "top_minus_bottom": rgb[:h].mean() - rgb[-h:].mean(),   # normally > 0, strongly < 0 = upside down
    }


def ego_share(segmenter, img, image_id, ego_ids):
    """Share of the view that Mask2Former labels as the own car (hood / dashboard)."""
    label_map = segmenter.label_map(img, image_id=f"q2crop_{image_id}")   # cached in data/masks/
    return np.isin(label_map, ego_ids).mean()


def seq_share(data, col):
    """Share of `col` per group, each drive (sequence) counted once, with a simple 95% interval."""
    per_seq = data.groupby(["group", "sequence"])[col].mean()
    g = per_seq.groupby("group")
    return pd.DataFrame({"share": g.mean(), "ci": 1.96 * g.std() / np.sqrt(g.count())})


def gallery(dataset, rows, crop_mode, title, label_col=None, ncols=10):
    """Show images with 'true -> predicted, p' (and optionally one feature value)."""
    rows = list(rows.itertuples())
    nrows = max(1, int(np.ceil(len(rows) / ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 3 * nrows))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, r in zip(np.ravel(axes), rows):
        ax.imshow(model_view(dataset, r.id, crop_mode))
        text = f"{r.country_id}->{r.pred_1} p={r.prob_1:.2f}"
        if label_col:
            text += f"\n{label_col}={getattr(r, label_col):.2f}"
        ax.set_title(text, fontsize=8)
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()