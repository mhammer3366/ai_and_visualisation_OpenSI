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
    edges = np.linspace(0, 1, n_bins + 1)
    labels = [f"{a:.0%} to {b:.0%}" for a, b in zip(edges[:-1], edges[1:])]   # e.g. "90% to 100%"
    bins = pd.cut(p["prob_1"], edges, labels=labels, include_lowest=True)
    t = p.groupby(bins, observed=True).agg(n=("correct", "size"), conf=("prob_1", "mean"),
                                           acc=("correct", "mean"))
    t["ci_low"], t["ci_high"] = wilson_ci(t["acc"] * t["n"], t["n"])
    return t


def ece(table):
    """Expected calibration error: |conf - acc| per bin, weighted by bin size."""
    return (table["n"] * (table["conf"] - table["acc"]).abs()).sum() / table["n"].sum()


def plot_reliability(tables, title="Below the diagonal = overconfident"):
    """Reliability diagram, one line per model, and below it how many images are in each bin.
    tables: {model name: calibration_table}."""
    fig, (ax, ax_n) = plt.subplots(2, 1, figsize=(5.5, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    for k, (name, t) in enumerate(tables.items()):
        ax.errorbar(t["conf"], t["acc"], yerr=[t["acc"] - t["ci_low"], t["ci_high"] - t["acc"]],
                    marker="o", capsize=3, label=f"{name} (ECE {ece(t):.3f})")
        ax_n.bar(t["conf"] + (k - 0.5) * 0.035, t["n"] / 1000, width=0.035, label=name)
    ax.set(ylabel="accuracy", xlim=(0, 1), ylim=(0, 1), title=title)
    ax.legend()
    ax_n.set(xlabel="mean confidence (top-1 probability)", ylabel="images (thousands)")
    plt.tight_layout()
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
                        labels={"gap": "overconf."},
                        color_continuous_scale="RdBu_r", range_color=(-0.4, 0.4),
                        projection="natural earth",
                        title=f"Overconfidence per country: mean confidence - accuracy (n >= {MIN_N})")
    fig.update_geos(projection_rotation_lon=0)
    fig.update_layout(width=900, height=520, margin=dict(l=10, r=10, t=60, b=10))
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


def plot_confident_errors(table, title):
    """Bar per (true -> guessed) pair: number of confident errors. Hover: share of that country's errors."""
    t = table.iloc[::-1].copy()   # biggest bar at the top
    t["pair"] = t["true_name"] + " -> " + t["pred_name"]
    fig = px.bar(t, x="n", y="pair", orientation="h", text="n",
                 hover_data={"conf": ":.0%", "share_of_errors": ":.0%", "pair": False},
                 labels={"n": f"images wrong with confidence >= {CONF_HIGH:.0%}", "pair": "",
                         "conf": "mean confidence", "share_of_errors": "share of this country's errors"})
    fig.update_layout(title=title, width=900, height=140 + 30 * len(t))
    return fig


def confident_examples(preds, true, pred, n=8, seed=42):
    """Random (not hand-picked) confident errors of one pair, e.g. true US, guessed GE."""
    p = usable(preds)
    rows = p[(p["country_id"] == true) & (p["pred_1"] == pred) & (p["prob_1"] >= CONF_HIGH)]
    return rows.sample(min(n, len(rows)), random_state=seed)


US_STATES = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
    "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "District of Columbia": "DC",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN",
    "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO",
    "Montana": "MT", "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ",
    "New Mexico": "NM", "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH",
    "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT",
    "Virginia": "VA", "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}


def georgia_states(preds):
    """Per US state: how many of its images StreetCLIP calls "Georgia" (the country GE)."""
    us = usable(preds)
    us = us[us["country_id"] == "US"]
    t = us.groupby("region").agg(n=("id", "size"), to_georgia=("pred_1", lambda s: (s == "GE").sum()))
    t["share"] = t["to_georgia"] / t["n"]
    t["code"] = t.index.map(US_STATES)
    return t.sort_values("share", ascending=False)


def georgia_compare(models):
    """Georgia confusion per model, e.g. {"CLIP": clip_preds, "StreetCLIP": preds}.
    Same prompt for both, so a difference comes from the fine-tuning."""
    rows = {}
    for name, preds in models.items():
        p = usable(preds)
        us_ge = (p["country_id"] == "US") & (p["pred_1"] == "GE")
        rows[name] = {"us_to_ge": us_ge.sum(),
                      "us_share": us_ge.sum() / (p["country_id"] == "US").sum(),
                      "us_to_ge_conf": (us_ge & (p["prob_1"] >= CONF_HIGH)).sum(),
                      "ge_guess_us": (p.loc[p["pred_1"] == "GE", "country_id"] == "US").mean(),
                      "ge_right": ((p["country_id"] == "GE") & (p["pred_1"] == "GE")).sum(),
                      "ge_n": (p["country_id"] == "GE").sum()}
    return pd.DataFrame.from_dict(rows, orient="index")

def plot_georgia_map(table, title):
    """US map: share of each state's images that are guessed as the country Georgia."""
    t = table[table["code"].notna()].reset_index()
    fig = px.choropleth(t, locations="code", locationmode="USA-states", scope="usa", color="share",
                        hover_name="region", hover_data={"n": True, "to_georgia": True, "code": False,
                                                         "share": ":.0%"},
                        color_continuous_scale="Reds", range_color=(0, 1),
                        labels={"share": "share", "to_georgia": "images guessed Georgia"})
    fig.update_layout(title=title, width=900, height=520, coloraxis_colorbar_tickformat=".0%",
                      margin=dict(l=10, r=10, t=60, b=10))
    return fig


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


def plot_gain_by_group(gain, title):
    """The direct test: gain per country, split by the paper's training list (one dot = one country)."""
    t = gain[~gain["low_n"]].reset_index()
    t["group"] = np.where(t["in_train_list"], "on the paper's training list", "not on the list")
    fig = px.box(t, x="group", y="gain", points="all", hover_name="name", color="group",
                 hover_data={"n": True, "acc_clip": ":.0%", "acc": ":.0%", "group": False},
                 category_orders={"group": ["on the paper's training list", "not on the list"]},
                 labels={"gain": "accuracy gain CLIP -> StreetCLIP", "group": "",
                         "acc_clip": "CLIP", "acc": "StreetCLIP"})
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(title=title, yaxis_tickformat="+.0%", showlegend=False, width=700, height=550)
    return fig
