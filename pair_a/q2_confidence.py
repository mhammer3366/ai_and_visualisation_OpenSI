"""Q2: Overconfidence and training coverage (pair_a, AI-aided, Phase I).

When is StreetCLIP confidently wrong, and can we infer which countries were in its
Street View training? Uses only the prediction tables in results/shared/ (black box).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

CONF_HIGH = 0.8   # "confident": top-1 probability >= 0.8
MIN_N = 30        # countries with fewer images: flagged as low_n


def usable(preds):
    """Only images whose country is a candidate; correct as 0/1."""
    p = preds[preds["representable"]].copy()
    p["correct"] = p["correct"].astype(float)
    return p


def wilson_ci(k, n, z=1.96):
    """95 % Wilson interval for k correct out of n."""
    p = k / n
    center = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return np.clip(center - half, 0, 1), np.clip(center + half, 0, 1)


# --- 1. Calibration ---------------------------------------------------------

def calibration_table(preds, n_bins=10):
    """Per confidence bin: n, mean confidence, accuracy with CI."""
    p = usable(preds)
    bins = pd.cut(p["prob_1"], np.linspace(0, 1, n_bins + 1), include_lowest=True)
    t = p.groupby(bins, observed=True).agg(n=("correct", "size"), conf=("prob_1", "mean"),
                                           acc=("correct", "mean"))
    t["ci_low"], t["ci_high"] = wilson_ci(t["acc"] * t["n"], t["n"])
    return t


def ece(table):
    """Expected calibration error: |conf - acc| per bin, weighted by bin size."""
    return (table["n"] * (table["conf"] - table["acc"]).abs()).sum() / table["n"].sum()


def plot_reliability(tables):
    """Reliability diagram, one line per model. tables: {model name: calibration_table}."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    for name, t in tables.items():
        ax.errorbar(t["conf"], t["acc"], yerr=[t["acc"] - t["ci_low"], t["ci_high"] - t["acc"]],
                    marker="o", capsize=3, label=f"{name} (ECE {ece(t):.3f})")
    ax.set(xlabel="mean confidence (top-1 probability)", ylabel="accuracy",
           xlim=(0, 1), ylim=(0, 1), title="Below the diagonal = overconfident")
    ax.legend()
    return fig


# --- 2. Where is the model confidently wrong? -------------------------------

def country_table(preds, countries):
    """Per true country: n, accuracy with CI, mean confidence, gap, share of confident errors."""
    p = usable(preds)
    p["conf_wrong"] = (p["prob_1"] >= CONF_HIGH) & (p["correct"] == 0)
    t = p.groupby("country_id").agg(n=("correct", "size"), acc=("correct", "mean"),
                                    conf=("prob_1", "mean"), conf_wrong=("conf_wrong", "mean"))
    t["ci_low"], t["ci_high"] = wilson_ci(t["acc"] * t["n"], t["n"])
    t["gap"] = t["conf"] - t["acc"]   # > 0: overconfident
    t["name"] = countries.loc[t.index, "prompt_name"]
    t["iso3"] = countries.loc[t.index, "iso3"]
    t["in_train_list"] = countries.loc[t.index, "in_paper_training_list"] == 1
    t["low_n"] = t["n"] < MIN_N
    return t


def plot_gap_map(table):
    """World map: confidence minus accuracy per country (red = overconfident)."""
    t = table[~table["low_n"]].reset_index()
    fig = px.choropleth(t, locations="iso3", color="gap", hover_name="name",
                        hover_data={"n": True, "acc": ":.2f", "conf": ":.2f", "iso3": False},
                        color_continuous_scale="RdBu_r", range_color=(-0.4, 0.4),
                        title=f"Overconfidence per country: mean confidence - accuracy (n >= {MIN_N})")
    return fig


def confident_errors(preds, countries, top=15):
    """Most common (true, predicted) pairs among errors with prob_1 >= CONF_HIGH."""
    p = usable(preds)
    errors = p[p["correct"] == 0]
    wrong = errors[errors["prob_1"] >= CONF_HIGH]
    t = wrong.groupby(["country_id", "pred_1"]).agg(n=("prob_1", "size"), conf=("prob_1", "mean"))
    t = t.reset_index().sort_values("n", ascending=False).head(top)
    t["share_of_errors"] = t["n"] / t["country_id"].map(errors["country_id"].value_counts())
    t["true_name"] = t["country_id"].map(countries["prompt_name"])
    t["pred_name"] = t["pred_1"].map(countries["prompt_name"])
    return t


def predicted_table(preds, countries):
    """Per predicted country: how often predicted, how often right (precision), real images (n_true)."""
    p = usable(preds)
    t = p.groupby("pred_1").agg(n_pred=("correct", "size"), precision=("correct", "mean"),
                                conf=("prob_1", "mean"))
    t["n_true"] = p["country_id"].value_counts().reindex(t.index, fill_value=0)   # 0: never right by design
    t["name"] = countries.loc[t.index, "prompt_name"]
    return t.sort_values("n_pred", ascending=False)


# --- 3. Training coverage ---------------------------------------------------

def training_groups(table):
    """Macro accuracy, confidence and gap for countries in / not in the paper's training list."""
    t = table[~table["low_n"]]
    g = t.groupby("in_train_list").agg(countries=("n", "size"), images=("n", "sum"),
                                       macro_acc=("acc", "mean"), acc_sem=("acc", "sem"),
                                       macro_conf=("conf", "mean"), mean_gap=("gap", "mean"))
    g["acc_ci"] = 1.96 * g.pop("acc_sem")   # 95 % CI half-width over countries
    return g


def gain_table(sc_table, clip_table):
    """Per country: StreetCLIP accuracy minus original CLIP accuracy (= gain from fine-tuning)."""
    t = sc_table[["name", "n", "acc", "in_train_list", "low_n"]].copy()
    t["acc_clip"] = clip_table["acc"]
    t["gain"] = t["acc"] - t["acc_clip"]
    return t


def plot_gain(gain):
    """CLIP vs StreetCLIP accuracy per country. Above the diagonal = fine-tuning helped."""
    t = gain[~gain["low_n"]].reset_index()
    fig = px.scatter(t, x="acc_clip", y="acc", color="in_train_list", hover_name="name",
                     hover_data={"n": True, "gain": ":.2f", "in_train_list": False},
                     labels={"acc_clip": "accuracy CLIP (before fine-tuning)",
                             "acc": "accuracy StreetCLIP (after fine-tuning)",
                             "in_train_list": "in paper's training list"},
                     title=f"Gain from Street View fine-tuning per country (n >= {MIN_N})")
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="gray"))
    fig.update_layout(width=700, height=650)
    return fig
