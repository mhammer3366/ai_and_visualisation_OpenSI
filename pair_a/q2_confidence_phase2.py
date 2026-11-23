"""Q2: Overconfidence and training coverage (pair_a, human-only, Phase II).

Embeddings (kNN) for confident errors and the training list, Grad-CAM for US -> Canada.
"""
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from scipy.stats import mannwhitneyu
from sklearn.decomposition import PCA
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from .q2_confidence import wilson_ci, CONF_HIGH, MIN_N
from shared.models import fit_image, fit_mask

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


def usable_refs(ref, queries, min_km=50):
    """For each query: how many references of its own country the kNN is allowed to use."""
    out = pd.Series(0, index=queries.index)
    for c, q in queries.groupby("country_id"):
        r = ref[ref["country_id"] == c]
        for i, row in q.iterrows():
            km = km_between(row["latitude"], row["longitude"], r["latitude"].to_numpy(), r["longitude"].to_numpy())
            out[i] = ((km >= min_km) & (r["sequence"].to_numpy() != row["sequence"])).sum()
    return out


def keep_reachable(ref, queries, min_km=50, k=K):
    """Drop queries with fewer than k usable references of their own country (e.g. small islands)."""
    refs = usable_refs(ref, queries, min_km)
    kept = queries[refs >= k]
    print(f"kept {len(kept):,} of {len(queries):,} queries, "
          f"{queries['country_id'].nunique() - kept['country_id'].nunique()} countries dropped completely")
    return kept


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


def plot_pca_pair(emb, p, a="US", b="CA", n=2000, seed=42):
    """PCA of correctly guessed images of two countries, with the confident a -> b errors on top."""
    base = p[p["country_id"].isin([a, b]) & (p["correct"] == 1)]
    base = base.sample(frac=1, random_state=seed).groupby("country_id").head(n)
    err = p[(p["country_id"] == a) & (p["pred_1"] == b) & (p["prob_1"] >= CONF_HIGH)]
    pca = PCA(n_components=2).fit(emb[base.index])
    xb = pca.transform(emb[base.index])
    xe = pca.transform(emb[err.index])
    plt.figure(figsize=(6, 6))
    for c, color in [(a, "tab:blue"), (b, "tab:red")]:
        m = (base["country_id"] == c).to_numpy()
        plt.scatter(xb[m, 0], xb[m, 1], s=5, alpha=0.3, color=color, label=f"{c} correct")
    plt.scatter(xe[:, 0], xe[:, 1], s=10, color="black", label=f"{a} -> {b}, conf. >= {CONF_HIGH} (n = {len(err)})")
    plt.xlabel(f"PC 1 ({pca.explained_variance_ratio_[0]:.0%} of variance)")
    plt.ylabel(f"PC 2 ({pca.explained_variance_ratio_[1]:.0%} of variance)")
    plt.title(f"{a} and {b} in the embedding space")
    plt.legend(markerscale=2)
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


def plot_text_gap(t, labels=("GE", "CD", "LT", "US", "CA")):
    t = t[~t["low_n"]]
    plt.figure(figsize=(6, 6))
    for on, color, name in [(True, "tab:blue", "on training list"), (False, "tab:orange", "not on list")]:
        s = t[t["on_list"] == on]
        plt.scatter(s["zeroshot_street"], s["knn_street"], color=color, alpha=0.7, label=name)
    plt.plot([0, 1], [0, 1], color="gray", lw=1, ls="--")
    for c in labels:
        plt.annotate(c, (t.loc[c, "zeroshot_street"], t.loc[c, "knn_street"]),
                     xytext=(4, 4), textcoords="offset points")
    plt.xlabel("zero-shot accuracy (prompts)")
    plt.ylabel("kNN accuracy (image neighbours)")
    plt.title("Above the line: the images know more than the prompts")
    plt.legend()
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


def heat_per_element(cam, geo, segmenter, dataset, rows, crop_mode, target="pred_1", text=None, batch=4):
    """Share of the Grad-CAM heat and of the area for each element, one row per image.
    target: column with the country Grad-CAM explains. text: a TextDetector adds a "text" element."""
    out = []
    for start in range(0, len(rows), batch):
        part = rows.iloc[start:start + batch]
        images = [dataset.get_image(i) for i in part["id"]]
        x = geo.prepare(images).to(geo.device, geo.dtype)
        targets = [ClassifierOutputTarget(geo.labels.index(c)) for c in part[target]]
        heats = cam(input_tensor=x, targets=targets)
        for img, image_id, heat in zip(images, part["id"], heats):
            row = {"id": image_id}
            masks = segmenter.element_masks(img, image_id)
            if text is not None:
                masks["text"] = text.text_mask(img, image_id)
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


def mean_heatmap(cam, geo, dataset, rows, target="pred_1", batch=2):
    """Average Grad-CAM map over all images in rows."""
    total = 0
    for start in range(0, len(rows), batch):
        part = rows.iloc[start:start + batch]
        x = geo.prepare([dataset.get_image(i) for i in part["id"]]).to(geo.device, geo.dtype)
        targets = [ClassifierOutputTarget(geo.labels.index(c)) for c in part[target]]
        total = total + cam(input_tensor=x, targets=targets).sum(axis=0)
    return total / len(rows)


def plot_mean_maps(a, b, name_a, name_b):
    diff = b - a
    lim = np.abs(diff).max()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, m, title in [(axes[0], a, name_a), (axes[1], b, name_b)]:
        im = ax.imshow(m, cmap="jet", vmin=0, vmax=max(a.max(), b.max()))
        ax.set_title(title)
    fig.colorbar(im, ax=axes[:2], shrink=0.8, label="mean Grad-CAM")
    im = axes[2].imshow(diff, cmap="RdBu_r", vmin=-lim, vmax=lim)
    axes[2].set_title(f"{name_b}\nminus {name_a}")
    fig.colorbar(im, ax=axes[2], shrink=0.8)
    for ax in axes:
        ax.axis("off")
    plt.show()


def plot_examples(cam, geo, dataset, rows, crop_mode, n=4, target="pred_1"):
    """n random images per group with their Grad-CAM map on top."""
    groups = rows["group"].unique()
    fig, axes = plt.subplots(len(groups), n, figsize=(3.2 * n, 3.4 * len(groups)))
    for r, g in enumerate(groups):
        for c, (_, row) in enumerate(rows[rows["group"] == g].head(n).iterrows()):
            img = dataset.get_image(row["id"])
            x = geo.prepare([img]).to(geo.device, geo.dtype)
            heat = cam(input_tensor=x, targets=[ClassifierOutputTarget(geo.labels.index(row[target]))])[0]
            axes[r, c].imshow(fit_image(img, crop_mode))
            axes[r, c].imshow(heat, cmap="jet", alpha=0.45)
            axes[r, c].set_title(f"{g} ({row['prob_1']:.2f})", fontsize=9)
            axes[r, c].axis("off")
    plt.tight_layout()
    plt.show()
