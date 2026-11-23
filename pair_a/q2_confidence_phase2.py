"""Q2: Overconfidence and training coverage (pair_a, human-only, Phase II).

Embeddings (kNN) for confident errors and the training list, Grad-CAM for US -> Canada.
"""
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from scipy.stats import mannwhitneyu
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from .q2_confidence import wilson_ci, CONF_HIGH, MIN_N
from shared.models import fit_mask

K = 10
SEARCH = 5000


def load_embeddings(run_dir):
    emb = np.load(f"{run_dir}/embeddings.npy").astype(np.float32)
    return emb / np.linalg.norm(emb, axis=1, keepdims=True)


def add_group(p):
    conf = np.where(p["prob_1"] >= CONF_HIGH, "confident", "unsure")
    right = np.where(p["correct"] == 1, "correct", "wrong")
    p["group"] = pd.Series(right, index=p.index) + " / " + conf
    return p


def km_between(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def neighbours(emb, ref, queries, k=K, min_km=0, chunk=500):
    """Countries of the k nearest neighbours of each query.
    Skipped: the same camera sequence and images closer than min_km to the query."""
    ref_emb = emb[ref.index]
    ref_country = ref["country_id"].to_numpy()
    ref_seq = ref["sequence"].to_numpy()
    ref_lat = ref["latitude"].to_numpy()
    ref_lon = ref["longitude"].to_numpy()
    result = []
    for start in range(0, len(queries), chunk):
        q = queries.iloc[start:start + chunk]
        sims = emb[q.index] @ ref_emb.T
        top = np.argpartition(-sims, SEARCH, axis=1)[:, :SEARCH]
        for i in range(len(q)):
            order = top[i][np.argsort(-sims[i, top[i]])]
            km = km_between(q["latitude"].iloc[i], q["longitude"].iloc[i], ref_lat[order], ref_lon[order])
            keep = (ref_seq[order] != q["sequence"].iloc[i]) & (km >= min_km)
            result.append(ref_country[order[keep][:k]])
    return result


def knn_columns(queries, neigh):
    q = queries.copy()
    q["n_neigh"] = [len(n) for n in neigh]
    q["knn_vote"] = [Counter(n).most_common(1)[0][0] if len(n) > 0 else None for n in neigh]
    q["share_true"] = [np.mean(n == c) for n, c in zip(neigh, q["country_id"])]
    q["share_pred"] = [np.mean(n == c) for n, c in zip(neigh, q["pred_1"])]
    return q


# --- Q2.1 -------------------------------------------------------------------

def group_table(q):
    q = q.copy()
    q["vote_true"] = (q["knn_vote"] == q["country_id"]).astype(int)
    q["vote_pred"] = (q["knn_vote"] == q["pred_1"]).astype(int)
    t = q.groupby("group").agg(n=("id", "size"), countries=("country_id", "nunique"),
                               share_true=("share_true", "mean"), share_pred=("share_pred", "mean"),
                               vote_true=("vote_true", "mean"), vote_pred=("vote_pred", "mean"))
    t["vote_pred_low"], t["vote_pred_high"] = wilson_ci(t["vote_pred"] * t["n"], t["n"])
    return t


def plot_groups(t):
    wrong = t.loc[["wrong / confident", "wrong / unsure"]]
    x = np.arange(len(wrong))
    plt.figure(figsize=(6, 4))
    plt.bar(x - 0.2, wrong["share_true"], 0.4, label="neighbours from true country")
    plt.bar(x + 0.2, wrong["share_pred"], 0.4, label="neighbours from predicted country")
    plt.xticks(x, wrong.index)
    plt.ylabel(f"share of the {K} nearest neighbours")
    plt.title("Wrong images: whose neighbours are they?")
    plt.legend()
    plt.show()


# --- Q2.2 -------------------------------------------------------------------

def country_table(q_street, q_clip, mapper):
    rows = []
    for c, g in q_street.groupby("country_id"):
        g_clip = q_clip.loc[g.index]
        rows.append({"country_id": c, "n": len(g),
                     "zeroshot_street": g["correct"].mean(), "zeroshot_clip": g_clip["correct"].mean(),
                     "knn_street": (g["knn_vote"] == c).mean(), "knn_clip": (g_clip["knn_vote"] == c).mean()})
    t = pd.DataFrame(rows).set_index("country_id")
    t["knn_gain"] = t["knn_street"] - t["knn_clip"]
    t["zeroshot_gain"] = t["zeroshot_street"] - t["zeroshot_clip"]
    t["on_list"] = [mapper.in_training_list(c) for c in t.index]
    t["low_n"] = t["n"] < MIN_N
    return t


def compare_lists(t):
    t = t[~t["low_n"]]
    rows = []
    for col in ["knn_street", "knn_clip", "knn_gain", "zeroshot_gain"]:
        on, off = t.loc[t["on_list"], col], t.loc[~t["on_list"], col]
        rows.append({"value": col, "on_list_mean": on.mean(), "off_list_mean": off.mean(),
                     "on_list_median": on.median(), "off_list_median": off.median(),
                     "p_value": mannwhitneyu(on, off).pvalue})
    print(f"countries: {t['on_list'].sum()} on the list, {(~t['on_list']).sum()} not on the list")
    return pd.DataFrame(rows).set_index("value")


def plot_gain(t):
    t = t[~t["low_n"]]
    plt.figure(figsize=(5, 4))
    plt.boxplot([t.loc[t["on_list"], "knn_gain"], t.loc[~t["on_list"], "knn_gain"]])
    plt.xticks([1, 2], ["on training list", "not on list"])
    plt.axhline(0, color="gray", lw=1)
    plt.ylabel("kNN accuracy StreetCLIP - CLIP")
    plt.title("Does fine-tuning separate list countries more?")
    plt.show()


# --- Q2.3 -------------------------------------------------------------------

class Scorer(torch.nn.Module):
    """Image -> scores for all candidate countries, so Grad-CAM can pick one class."""

    def __init__(self, geo):
        super().__init__()
        self.clip = geo.model
        self.text = geo.text_embeds

    def forward(self, x):
        pooled = self.clip.vision_model(pixel_values=x).pooler_output
        img = self.clip.visual_projection(pooled)
        img = img / img.norm(dim=-1, keepdim=True)
        return 100 * img @ self.text.T


def vit_reshape(t):
    t = t[:, 1:, :]
    side = int(t.shape[1] ** 0.5)
    return t.reshape(t.shape[0], side, side, t.shape[2]).permute(0, 3, 1, 2)


def make_cam(geo):
    layer = geo.model.vision_model.encoder.layers[-1].layer_norm1
    return GradCAM(model=Scorer(geo), target_layers=[layer], reshape_transform=vit_reshape)


def heat_per_element(cam, geo, segmenter, dataset, rows, crop_mode, batch=4):
    """Share of the Grad-CAM heat and of the area for each element, one row per image."""
    out = []
    for start in range(0, len(rows), batch):
        part = rows.iloc[start:start + batch]
        images = [dataset.get_image(i) for i in part["id"]]
        x = geo.prepare(images).to(geo.device, geo.dtype)
        targets = [ClassifierOutputTarget(geo.labels.index(c)) for c in part["pred_1"]]
        heats = cam(input_tensor=x, targets=targets)
        for img, image_id, heat in zip(images, part["id"], heats):
            row = {"id": image_id}
            masks = segmenter.element_masks(img, image_id)
            for element, mask in masks.items():
                m = fit_mask(mask, crop_mode)
                row[f"heat_{element}"] = heat[m].sum() / heat.sum()
                row[f"area_{element}"] = m.mean()
            out.append(row)
    return pd.DataFrame(out)


def element_table(heat, elements):
    rows = []
    for e in elements:
        a = heat.loc[heat["group"] == "US -> Canada", f"heat_{e}"]
        b = heat.loc[heat["group"] == "US correct", f"heat_{e}"]
        rows.append({"element": e, "heat_wrong": a.mean(), "heat_correct": b.mean(),
                     "area_wrong": heat.loc[heat["group"] == "US -> Canada", f"area_{e}"].mean(),
                     "area_correct": heat.loc[heat["group"] == "US correct", f"area_{e}"].mean(),
                     "p_value": mannwhitneyu(a, b).pvalue})
    return pd.DataFrame(rows).set_index("element")


def plot_elements(t):
    x = np.arange(len(t))
    plt.figure(figsize=(7, 4))
    plt.bar(x - 0.2, t["heat_correct"], 0.4, label="US correct (conf. >= 0.8)")
    plt.bar(x + 0.2, t["heat_wrong"], 0.4, label="US -> Canada (conf. >= 0.8)")
    plt.xticks(x, t.index, rotation=30)
    plt.ylabel("share of Grad-CAM heat")
    plt.title("Where does the model look for the predicted country?")
    plt.legend()
    plt.tight_layout()
    plt.show()
