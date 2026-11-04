"""Small display helpers for pair_a (readable tables, example images)."""
import numpy as np
import matplotlib.pyplot as plt

from shared.models import fit_image


def show_table(df, cols, pct=(), pts=(), ci=None, bar=None, grey=None, index=True, digits=1):
    """Readable table for the notebook.

    cols  {old column: new name}, in the order to show (other columns are dropped)
    pct   old column names shown as percent (0.605 -> 60.5 %)
    pts   old column names that are differences, shown in points (0.067 -> +6.7 pts)
    ci    (value, low, high) old names: adds a "95 % CI" column as text
    bar   old column name that gets a small bar inside the cell
    grey  boolean Series (same index): rows shown in grey, e.g. too few images
    index False: hide the row labels (when a name column is shown)
    """
    t = df.copy()
    if ci:   # CI as text, placed right after its value column
        value, low, high = ci
        t["95 % CI"] = [f"{a * 100:.{digits}f} to {b * 100:.{digits}f} %" if np.isfinite(a) else ""
                        for a, b in zip(t[low], t[high])]
        keys = list(cols)
        keys.insert(keys.index(value) + 1, "95 % CI")
        cols = {k: cols.get(k, k) for k in keys}
    t = t[list(cols)].rename(columns=cols)
    t.index.name = None
    fmt = {cols[c]: (lambda v: "" if v != v else f"{v * 100:.{digits}f} %") for c in pct}
    fmt.update({cols[c]: (lambda v: "" if v != v else f"{v * 100:+.{digits}f} pts") for c in pts})
    fmt.update({c: (lambda v: "" if v != v else f"{v:,.0f}") for c in t.columns
                if c not in fmt and t[c].dtype.kind in "if" and (t[c].dropna() % 1 == 0).all()})
    style = t.style.format(fmt, na_rep="")
    if bar:
        style = style.bar(subset=[cols[bar]], color="#c6dbef", vmin=0, height=60, width=90)
    if grey is not None:
        rows = grey.reindex(t.index).fillna(False)
        style = style.apply(lambda r: ["color: #aaa" if rows[r.name] else "" for _ in r], axis=1)
    if not index:
        style = style.hide(axis="index")
    return style


def image_grid(dataset, rows, title, note=None, crop_mode="center_crop", cols=4, size=3.0):
    """Example images as the model sees them (after crop_mode), with true country, guess and confidence.
    rows: DataFrame with id, country_id, pred_1, prob_1. note: extra column for a second line (e.g. region)."""
    n = len(rows)
    fig, axes = plt.subplots((n + cols - 1) // cols, cols, figsize=(cols * size, ((n + cols - 1) // cols) * size + 0.6))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, (_, r) in zip(np.atleast_1d(axes).ravel(), rows.iterrows()):
        ax.imshow(fit_image(dataset.get_image(r["id"]), crop_mode))
        mark = "right" if r["pred_1"] == r["country_id"] else "wrong"
        extra = f"\n{r[note]}" if note else ""
        ax.set_title(f"true {r['country_id']} | guess {r['pred_1']} ({r['prob_1']:.0%}) {mark}{extra}", fontsize=9)
    fig.suptitle(title)
    plt.tight_layout()
    return fig
