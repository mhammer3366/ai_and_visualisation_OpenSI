"""Q4: Environment (pair_a, AI-aided, Phase I).

Do accuracy and confidence vary with climate, land cover, distance to the sea, or
urban vs. rural setting? Uses only the prediction tables in results/shared/ (black box).

Main idea: environment and country are mixed up (e.g. half of "sparse vegetation" is
Australia). So every effect is shown twice:
    raw             pooled accuracy per category (what a simple table shows)
    within-country  accuracy of the category minus the accuracy of its own country,
                    averaged over countries (the country mix is removed)
"""
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots

from .q2_confidence import wilson_ci

MIN_N = 30      # categories / countries with fewer images: flagged as low_n
MIN_CELL = 20   # a country needs this many images in a category for the within-country comparison
MIN_COUNTRIES = 5   # within-country value only shown if at least this many countries support it
MIN_SEQ = 10    # urban map: a country needs this many built-up camera sequences

CLIMATE_GROUPS = {"A": "A tropical", "B": "B arid", "C": "C temperate",
                  "D": "D continental", "E": "E polar"}
SEA_BINS = [0, 0.05, 0.25, 0.5, 0.9, 0.99, 1.0]   # dist_sea: 0 = at the coast, 1 = far inland (capped)
SEA_LABELS = ["0-0.05 (coast)", "0.05-0.25", "0.25-0.5", "0.5-0.9", "0.9-0.99", "0.99-1 (inland)"]


def add_env_columns(preds):
    """Only candidate countries; adds climate_group, urban and sea_bin. preds must be decoded."""
    p = preds[preds["representable"]].copy()
    p["correct"] = p["correct"].astype(float)
    first = p["climate_name"].str[0]
    p["climate_group"] = first.map(CLIMATE_GROUPS)   # code 0 (water / no data) -> NaN
    p["urban"] = np.where(p["land_cover_name"] == "artificial surfaces", "urban (built-up)", "rural / other")
    p.loc[p["land_cover_name"].isin(["no data"]), "urban"] = np.nan
    p["sea_bin"] = pd.cut(p["dist_sea"], SEA_BINS, labels=SEA_LABELS, include_lowest=True).astype(str)
    return p


def effect_table(env, var):
    """Per category of var: raw accuracy and confidence (pooled), and the within-country difference."""
    p = env[env[var].notna()]
    t = p.groupby(var).agg(n=("correct", "size"), countries=("country_id", "nunique"),
                           acc=("correct", "mean"), conf=("prob_1", "mean"))
    t["ci_low"], t["ci_high"] = wilson_ci(t["acc"] * t["n"], t["n"])
    t["gap"] = t["conf"] - t["acc"]   # > 0: overconfident (see Q2)
    t["low_n"] = t["n"] < MIN_N

    # Within-country: compare each (country, category) cell with the country's own average
    country = p.groupby("country_id").agg(c_acc=("correct", "mean"), c_conf=("prob_1", "mean"))
    cells = p.groupby(["country_id", var]).agg(n=("correct", "size"), acc=("correct", "mean"),
                                               conf=("prob_1", "mean")).reset_index()
    cells = cells[cells["n"] >= MIN_CELL].join(country, on="country_id")
    cells["d_acc"] = cells["acc"] - cells["c_acc"]
    cells["d_conf"] = cells["conf"] - cells["c_conf"]
    w = cells.groupby(var).agg(w_countries=("d_acc", "size"), w_acc=("d_acc", "mean"),
                               w_acc_sem=("d_acc", "sem"), w_conf=("d_conf", "mean"),
                               share_better=("d_acc", lambda d: (d > 0).mean()))
    w["w_acc_ci"] = 1.96 * w.pop("w_acc_sem")   # 95 % CI half-width over countries
    w.loc[w["w_countries"] < MIN_COUNTRIES, ["w_acc", "w_acc_ci", "w_conf", "share_better"]] = np.nan
    return t.join(w)


def per_sequence(env, var):
    """One row per (country, camera sequence, category): images of one drive count only once.
    Use as input for effect_table to check that a result is not driven by a few sequences."""
    return env.groupby(["country_id", "sequence", var]).agg(
        correct=("correct", "mean"), prob_1=("prob_1", "mean")).reset_index()


def plot_effects(table, title, order=None):
    """Left: raw accuracy with CI. Right: within-country difference with CI. Hover shows n and countries."""
    t = table[~table["low_n"]].reset_index()
    name = t.columns[0]
    t = t.set_index(name).loc[order or t.sort_values("acc")[name].tolist()].reset_index()
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.04,
                        subplot_titles=("raw accuracy (pooled)",
                                        f"within-country difference (countries with >= {MIN_CELL} images)"))
    hover = "%{y}<br>n = %{customdata[0]:,}<br>countries = %{customdata[1]}<extra></extra>"
    fig.add_scatter(x=t["acc"], y=t[name], mode="markers", row=1, col=1, name="raw",
                    error_x=dict(type="data", symmetric=False, array=t["ci_high"] - t["acc"],
                                 arrayminus=t["acc"] - t["ci_low"]),
                    customdata=t[["n", "countries"]], hovertemplate=hover)
    fig.add_scatter(x=t["w_acc"], y=t[name], mode="markers", row=1, col=2, name="within-country",
                    error_x=dict(type="data", array=t["w_acc_ci"]),
                    customdata=t[["n", "w_countries"]], hovertemplate=hover)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", row=1, col=2)
    fig.update_layout(title=title, showlegend=False, height=120 + 40 * len(t), width=950)
    return fig


def urban_table(env, countries):
    """Per country: accuracy in built-up vs. other areas (only countries with enough images in both)."""
    t = env.pivot_table(index="country_id", columns="urban", values="correct", aggfunc=["mean", "size"])
    t.columns = [f"{a}_{'urban' if 'urban' in b else 'rural'}" for a, b in t.columns]
    t = t[(t["size_urban"] >= MIN_CELL) & (t["size_rural"] >= MIN_CELL)].copy()
    t["diff"] = t["mean_urban"] - t["mean_rural"]   # > 0: better in built-up areas
    t["seq_urban"] = env[env["urban"] == "urban (built-up)"].groupby("country_id")["sequence"].nunique()
    t["name"] = countries.loc[t.index, "prompt_name"]
    t["iso3"] = countries.loc[t.index, "iso3"]
    return t.sort_values("diff")


def plot_urban_map(table):
    """World map: accuracy in built-up areas minus accuracy elsewhere, per country.
    Countries with fewer than MIN_SEQ built-up sequences are left out (too few independent drives)."""
    t = table[table["seq_urban"] >= MIN_SEQ].reset_index()
    fig = px.choropleth(t, locations="iso3", color="diff", hover_name="name",
                        hover_data={"size_urban": True, "seq_urban": True, "size_rural": True,
                                    "mean_urban": ":.2f", "mean_rural": ":.2f", "iso3": False},
                        color_continuous_scale="RdBu", range_color=(-0.4, 0.4),
                        projection="natural earth",
                        title=f"Urban minus rural accuracy (blue = better in built-up areas, "
                              f">= {MIN_SEQ} built-up sequences)")
    fig.update_geos(projection_rotation_lon=0)
    fig.update_layout(width=950, height=520, margin=dict(l=10, r=10, t=60, b=10))
    return fig
