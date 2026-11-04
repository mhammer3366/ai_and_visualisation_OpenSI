"""Q4: Environment (pair_a, AI-aided, Phase I).

Do accuracy and confidence vary with climate, land cover, distance to the sea, or
urban vs. rural setting? Uses only the prediction tables in results/shared/ (black box).

Main idea: environment and country are mixed up (e.g. half of "sparse vegetation" is
Australia). So every category gets:
    actual          pooled accuracy of the category (what a simple table shows)
    expected        accuracy we would expect from its country mix alone:
                    sum over countries (share of the country in the category * accuracy of the country)
    within-country  accuracy of the category minus the accuracy of its own country,
                    averaged over countries (every country counts once)
If actual = expected, the country mix explains the difference, not the environment.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .q2_confidence import wilson_ci
from .viz import show_table

MIN_N = 30      # categories / countries with fewer images: flagged as low_n
MIN_CELL = 20   # a country needs this many images in a category for the within-country comparison
MIN_COUNTRIES = 5   # within-country value only shown if at least this many countries support it
MIN_SEQ = 10    # urban map: a country needs this many built-up camera sequences
MIN_PLOT = 500  # plots: only categories with at least this many images (small ones stay in the tables)

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

    # Expected from the country mix: mean over the images of their country's accuracy
    country = p.groupby("country_id").agg(c_acc=("correct", "mean"), c_conf=("prob_1", "mean"))
    t["expected"] = p["country_id"].map(country["c_acc"]).groupby(p[var]).mean()
    t["effect"] = t["acc"] - t["expected"]   # > 0: better than its country mix predicts
    by_cat = p.groupby(var)["country_id"]
    t["top_country"] = by_cat.agg(lambda s: s.value_counts().index[0])   # biggest country in the category
    t["top_share"] = by_cat.agg(lambda s: s.value_counts(normalize=True).iloc[0])

    # Within-country: compare each (country, category) cell with the country's own average
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


def show_effects(table):
    """Readable version of effect_table (grey rows: fewer than MIN_N images)."""
    cols = {"n": "Images", "countries": "Countries", "acc": "Accuracy", "expected": "Expected from country mix",
            "effect": "Actual - expected", "w_acc": "Within-country difference",
            "share_better": "Countries above own average"}
    return show_table(table, cols, pct=["acc", "expected", "share_better"], pts=["effect", "w_acc"],
                      ci=("acc", "ci_low", "ci_high"), grey=table["low_n"])


def per_sequence(env, var):
    """One row per (country, camera sequence, category): images of one drive count only once.
    Use as input for effect_table to check that a result is not driven by a few sequences."""
    return env.groupby(["country_id", "sequence", var]).agg(
        correct=("correct", "mean"), prob_1=("prob_1", "mean")).reset_index()


def plot_expected(table, title, order=None):
    """Dumbbell: grey ring = expected from the country mix, blue dot = actual accuracy (with CI).
    Short line: the country mix explains the category. Long line: the environment itself matters."""
    t = table[table["n"] >= MIN_PLOT]
    t = t.loc[order[::-1] if order else t.sort_values("acc").index]   # first row of order at the top
    y = t.index.tolist()
    lines_x, lines_y = [], []
    for cat, r in t.iterrows():
        lines_x += [r["expected"], r["acc"], None]
        lines_y += [cat, cat, None]
    hover = ("%{y}<br>actual %{customdata[0]:.1%} | expected %{customdata[1]:.1%}<br>"
             "images %{customdata[2]:,} | countries %{customdata[3]}<br>"
             "biggest country %{customdata[4]} (%{customdata[5]:.0%})<extra></extra>")
    data = t[["acc", "expected", "n", "countries", "top_country", "top_share"]]
    fig = go.Figure()
    fig.add_scatter(x=lines_x, y=lines_y, mode="lines", line=dict(color="#bbb", width=3), hoverinfo="skip",
                    showlegend=False)
    fig.add_scatter(x=t["expected"], y=y, mode="markers", name="expected from country mix",
                    marker=dict(size=11, color="white", line=dict(color="#777", width=2)),
                    customdata=data, hovertemplate=hover)
    fig.add_scatter(x=t["acc"], y=y, mode="markers", name="actual accuracy",
                    marker=dict(size=11, color="#2171b5"),
                    error_x=dict(type="data", symmetric=False, array=t["ci_high"] - t["acc"],
                                 arrayminus=t["acc"] - t["ci_low"], color="#2171b5"),
                    customdata=data, hovertemplate=hover)
    # x axis spans at least 20 points, so small differences also look small
    lo = min(t["ci_low"].min(), t["expected"].min())
    hi = max(t["ci_high"].max(), t["expected"].max())
    mid, half = (lo + hi) / 2, max(hi - lo, 0.2) / 2 + 0.01
    fig.update_xaxes(range=[mid - half, mid + half])
    fig.update_layout(title=title, xaxis_title=f"accuracy (categories with >= {MIN_PLOT} images)",
                      xaxis_tickformat=".0%", width=900, height=170 + 38 * len(t),
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0))
    return fig


def plot_country_mix(env, var, title, order=None, top=3):
    """Who is in each category? Share of the 3 biggest countries (names in the bars) and the rest."""
    p = env[env[var].notna()]
    counts = p[var].value_counts()
    cats = order or counts[counts >= MIN_PLOT].index.tolist()
    rows = []
    for cat in cats:
        shares = p.loc[p[var] == cat, "country_id"].value_counts(normalize=True)
        for rank, (c, share) in enumerate(shares.head(top).items(), 1):
            rows.append({var: cat, "part": f"country {rank}", "country": c, "share": share})
        rows.append({var: cat, "part": "all other", "country": "", "share": 1 - shares.head(top).sum()})
    m = pd.DataFrame(rows)
    colors = {"country 1": "#08519c", "country 2": "#4292c6", "country 3": "#9ecae1", "all other": "#e5e5e5"}
    fig = px.bar(m, x="share", y=var, color="part", orientation="h", text="country",
                 color_discrete_map=colors, category_orders={var: cats},
                 hover_data={"share": ":.0%", "part": False})
    fig.update_traces(textposition="inside", insidetextanchor="middle")
    fig.update_layout(title=title, xaxis_tickformat=".0%", xaxis_title="share of images", yaxis_title="",
                      legend_title_text="", width=900, height=170 + 38 * len(cats),
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0))
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


def urban_examples(env, country, n=4, seed=42):
    """Random (not hand-picked) images of one country: n built-up, then n other."""
    q = env[env["country_id"] == country]
    return pd.concat([q[q["urban"] == u].sample(n, random_state=seed)
                      for u in ["urban (built-up)", "rural / other"]])


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
    fig.update_layout(width=900, height=520, margin=dict(l=10, r=10, t=60, b=10))
    return fig
