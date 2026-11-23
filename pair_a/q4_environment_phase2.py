"""Q4: Environment (pair_a, human-only, Phase II).

Is the environment visible in the embeddings (kNN), and where does Grad-CAM look in built-up vs. rural images?
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.stats import wilcoxon

from .q4_environment import MIN_CELL, MIN_COUNTRIES

URBAN = "urban (built-up)"
RURAL = "rural / other"


# --- Q4.1 -------------------------------------------------------------------

def urban_rural(q, cols=("share_true", "knn_true", "correct")):
    """Per country: built-up minus rural for each column (countries with enough images in both)."""
    t = q.pivot_table(index="country_id", columns="urban", values=list(cols), aggfunc="mean")
    n = q.pivot_table(index="country_id", columns="urban", values=cols[0], aggfunc="size")
    keep = (n[URBAN] >= MIN_CELL) & (n[RURAL] >= MIN_CELL)
    out = pd.DataFrame({f"d_{c}": t[(c, URBAN)] - t[(c, RURAL)] for c in cols})[keep]
    out["n_urban"] = n.loc[keep, URBAN]
    out["n_rural"] = n.loc[keep, RURAL]
    return out


def summary(diffs):
    rows = []
    for col in diffs.columns:
        d = diffs[col].dropna()
        rows.append({"value": col, "countries": len(d), "mean": d.mean(), "ci": 1.96 * d.sem(),
                     "share_above_0": (d > 0).mean(), "p_value": wilcoxon(d).pvalue})
    return pd.DataFrame(rows).set_index("value")


def plot_urban(diffs):
    plt.figure(figsize=(5, 5))
    plt.scatter(diffs["d_correct"], diffs["d_share_true"], alpha=0.7)
    plt.axhline(0, color="gray", lw=1)
    plt.axvline(0, color="gray", lw=1)
    plt.xlabel("zero-shot accuracy: built-up - rural")
    plt.ylabel("neighbours from own country: built-up - rural")
    plt.title("One dot = one country")
    plt.show()


def plot_urban_maps(diffs, countries):
    names = {"d_correct": "zero-shot accuracy", "d_share_true": "neighbours from own country (kNN)"}
    m = diffs[list(names)].rename(columns=names).reset_index()
    m = m.melt(id_vars="country_id", var_name="value", value_name="built-up minus rural")
    m["iso3"] = countries.loc[m["country_id"], "iso3"].to_numpy()
    m["name"] = countries.loc[m["country_id"], "prompt_name"].to_numpy()
    fig = px.choropleth(m, locations="iso3", color="built-up minus rural", facet_col="value",
                        hover_name="name", color_continuous_scale="RdBu", range_color=(-0.3, 0.3),
                        projection="natural earth")
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
    fig.update_geos(projection_rotation_lon=0, showland=True, landcolor="#dddddd")
    fig.update_layout(title="Built-up minus rural per country (blue = higher in built-up areas)",
                      width=1100, height=420, margin=dict(l=10, r=10, t=70, b=10))
    return fig


# --- Q4.2 -------------------------------------------------------------------

def within_country(q, var, col):
    """Per category: mean over countries of (category value - country value)."""
    country = q.groupby("country_id")[col].mean()
    cells = q.groupby(["country_id", var])[col].agg(["size", "mean"]).reset_index()
    cells = cells[cells["size"] >= MIN_CELL]
    cells["diff"] = cells["mean"] - cells["country_id"].map(country)
    t = cells.groupby(var)["diff"].agg(["size", "mean", "sem"])
    t.columns = ["countries", f"w_{col}", "sem"]
    t[f"w_{col}_ci"] = 1.96 * t.pop("sem")
    t["images"] = q.groupby(var).size()
    return t[t["countries"] >= MIN_COUNTRIES]


def env_table(q, var):
    a = within_country(q, var, "share_true")
    b = within_country(q, var, "correct")[["w_correct", "w_correct_ci"]]
    return a.join(b).sort_values("w_share_true")


def plot_env(t, title):
    y = np.arange(len(t))
    plt.figure(figsize=(7, 0.5 * len(t) + 1.5))
    plt.errorbar(t["w_share_true"], y - 0.15, xerr=t["w_share_true_ci"], fmt="o",
                 label="neighbours from own country (kNN)")
    plt.errorbar(t["w_correct"], y + 0.15, xerr=t["w_correct_ci"], fmt="s",
                 label="zero-shot accuracy")
    plt.axvline(0, color="gray", lw=1)
    plt.yticks(y, t.index)
    plt.xlabel("within-country difference")
    plt.title(title)
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.show()


# --- Q4.3 -------------------------------------------------------------------

def gradcam_sample(env, per_group=10, seed=42):
    """Countries with enough built-up and rural images: per_group of each, drawn at random."""
    env = env[env["urban"].notna()]
    n = env.groupby(["country_id", "urban"]).size().unstack()
    countries = n[(n[URBAN] >= MIN_CELL) & (n[RURAL] >= MIN_CELL)].index
    s = env[env["country_id"].isin(countries)]
    return s.sample(frac=1, random_state=seed).groupby(["country_id", "urban"]).head(per_group)


def heat_urban_rural(heat, elements):
    """Per element: built-up minus rural, first inside each country, then averaged over countries."""
    for e in elements:
        heat[f"extra_{e}"] = heat[f"heat_{e}"] - heat[f"area_{e}"]
    cols = [f"{kind}_{e}" for e in elements for kind in ["heat", "area", "extra"]]
    m = heat.groupby(["country_id", "urban"])[cols].mean().unstack()
    diffs = pd.DataFrame({c: m[(c, URBAN)] - m[(c, RURAL)] for c in cols})
    rows = []
    for e in elements:
        rows.append({"element": e,
                     "heat_built_up": heat.loc[heat["urban"] == URBAN, f"heat_{e}"].mean(),
                     "heat_rural": heat.loc[heat["urban"] == RURAL, f"heat_{e}"].mean(),
                     "d_heat": diffs[f"heat_{e}"].mean(), "d_area": diffs[f"area_{e}"].mean(),
                     "d_extra": diffs[f"extra_{e}"].mean(), "d_extra_ci": 1.96 * diffs[f"extra_{e}"].sem(),
                     "p_value": wilcoxon(diffs[f"extra_{e}"].dropna()).pvalue})
    return pd.DataFrame(rows).set_index("element")


def plot_heat(t):
    x = np.arange(len(t))
    plt.figure(figsize=(7, 4))
    plt.bar(x - 0.2, t["d_area"], 0.4, label="area")
    plt.bar(x + 0.2, t["d_heat"], 0.4, label="Grad-CAM heat (true country)")
    plt.axhline(0, color="gray", lw=1)
    plt.xticks(x, t.index, rotation=30)
    plt.ylabel("built-up - rural (within country)")
    plt.title("Does the heat follow the area?")
    plt.legend()
    plt.tight_layout()
    plt.show()
