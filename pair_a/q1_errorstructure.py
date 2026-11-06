"""
q1_errorstructure.py

Q1 - Error structure and geography
Which countries are confused, what do confused pairs share, are there attractor
countries, and do errors cluster geographically?

Classes
-------
Q1Config       all column names and analysis settings in one place
Q1DataBuilder  load -> validate -> filter -> build analysis tables
Q1Data         bundle returned by the builder (image table, country_stats,
               confusion_pairs, metadata, pair matrices, validation report)
Q1Analyser     rankings, continent comparison, attractors, random baselines,
               geographic statistics (returns DataFrames / dicts, never prints)
Q1Reporter     basic tables and plots
"""
from __future__ import annotations

import ast
import warnings
from collections import deque
from dataclasses import dataclass, field
from statistics import NormalDist

import numpy as np
import pandas as pd


# =============================================================================
# Config
# =============================================================================
@dataclass
class Q1Config:
    # --- image-level prediction table ----------------------------------------
    true_col: str = "true_country"
    pred_col: str = "pred_country"
    topk_col: str | None = None          # one column: list or delimited string, ordered by rank
    topk_cols: list | None = None        # or one column per rank, e.g. ["pred_1", ..., "pred_5"]
    topk_sep: str = "|"
    image_id_col: str | None = None
    lat_col: str | None = None           # image GPS, kept in the image table
    lon_col: str | None = None

    # --- country metadata -------------------------------------------------------
    meta_country_col: str = "country"
    meta_code_col: str | None = None     # set if neighbours are given as codes (e.g. ISO3)
    continent_col: str = "continent"
    subregion_col: str | None = "subregion"
    neighbors_col: str | None = "neighbors"
    meta_lat_col: str | None = None      # country centroid -> km distance (only if available)
    meta_lon_col: str | None = None
    scalar_attrs: tuple = ("driving_side", "landlocked")   # -> "same_<attr>" features
    list_attrs: tuple = ("languages",)                     # -> "shares_<attr>" features
    list_sep: str = ","
    name_map: dict = field(default_factory=dict)           # alias -> metadata country name

    # --- analysis settings ------------------------------------------------------
    ks: tuple = (1, 3, 5)
    min_n: int = 20
    ci_level: float = 0.95
    n_permutations: int = 1000
    baseline_pool: str = "observed"      # "observed" (labels seen as true/pred) or "metadata"
    seed: int = 42


# =============================================================================
# Helpers
# =============================================================================
def _is_missing(x) -> bool:
    if isinstance(x, (list, tuple, set, np.ndarray)):
        return False
    try:
        return bool(pd.isna(x))
    except (TypeError, ValueError):
        return False


def _to_list(x, sep: str = ",") -> list:
    """Parse a list, a stringified list "['a', 'b']" or a delimited string "a, b"."""
    if isinstance(x, (list, tuple, set, np.ndarray)):
        items = list(x)
    else:
        if _is_missing(x):
            return []
        s = str(x).strip()
        if s.lower() in ("", "nan", "none", "[]"):
            return []
        items = None
        if s[0] in "[(" and s[-1] in "])":
            try:
                v = ast.literal_eval(s)
                if isinstance(v, (list, tuple, set)):
                    items = list(v)
            except (ValueError, SyntaxError):
                s = s[1:-1]
        if items is None:
            items = s.split(sep)
    out = []
    for v in items:
        t = str(v).strip().strip("'\"").strip()
        if t and t.lower() != "nan":
            out.append(t)
    return out


def wilson_ci(k, n, level: float = 0.95):
    """Wilson score interval for a binomial proportion k/n (vectorised)."""
    z = NormalDist().inv_cdf(1 - (1 - level) / 2)
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = k / n
        denom = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def _entropy_bits(counts) -> float:
    c = np.asarray(counts, dtype=float)
    c = c[c > 0]
    if c.sum() == 0:
        return np.nan
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def _haversine_matrix(lat, lon) -> np.ndarray:
    lat = np.radians(np.asarray(lat, dtype=float))
    lon = np.radians(np.asarray(lon, dtype=float))
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _nanq(x, q) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return float(np.nanquantile(x, q))


def _nanmean(x) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return float(np.nanmean(x)) if len(x) else np.nan


# =============================================================================
# Data bundle
# =============================================================================
@dataclass
class Q1Data:
    images: pd.DataFrame            # one row per image (kept for GPS / image-level analyses)
    country_stats: pd.DataFrame     # one row per country
    confusion_pairs: pd.DataFrame   # one row per directed pair A -> B (errors only)
    meta: pd.DataFrame              # cleaned metadata, index = country
    countries: list                 # row/column order of the pair matrices
    country_index: dict             # country -> matrix index
    pair_matrices: dict             # feature -> (m x m) float array, NaN = unknown
    numeric_features: list          # distance features (smaller = closer)
    ks: list                        # top-k values actually available
    config: Q1Config
    report: dict                    # validation / filtering report


# =============================================================================
# 1) Builder
# =============================================================================
class Q1DataBuilder:
    """Load, validate, filter and build the Q1 analysis tables.

    Steps
    -----
    1. Prepare metadata (clean names, parse neighbours / list attributes).
    2. Validate and filter the image-level predictions.
    3. Build the image table (correct@k, regions, pair features of each error).
    4. Build country_stats (one row per country).
    5. Build confusion_pairs (one row per A -> B).
    6. Return everything as one Q1Data bundle (image table included).
    """

    def __init__(self, predictions: pd.DataFrame, metadata: pd.DataFrame,
                 config: Q1Config | None = None):
        self.cfg = config or Q1Config()
        self.predictions = predictions
        self.metadata = metadata
        self.report: dict = {}

    # ------------------------------------------------------------------ public
    def build(self) -> Q1Data:
        meta = self._prepare_metadata()
        images, ks = self._prepare_images(meta)
        countries = list(meta.index)
        index = {c: i for i, c in enumerate(countries)}
        mats, numeric = self._pair_matrices(meta, index)
        images = self._add_pair_features(images, index, mats)
        country_stats = self._country_stats(images, meta, ks)
        pairs = self._confusion_pairs(images, country_stats, meta, index, mats)
        return Q1Data(images=images, country_stats=country_stats, confusion_pairs=pairs,
                      meta=meta, countries=countries, country_index=index,
                      pair_matrices=mats, numeric_features=numeric, ks=ks,
                      config=self.cfg, report=self.report)

    # ------------------------------------------------------------------ helpers
    def _clean_name(self, x):
        if _is_missing(x):
            return np.nan
        t = str(x).strip()
        if not t or t.lower() == "nan":
            return np.nan
        return self.cfg.name_map.get(t, t)

    # ------------------------------------------------------------------ step 1
    def _prepare_metadata(self) -> pd.DataFrame:
        cfg = self.cfg
        meta = self.metadata.copy()

        required = [cfg.meta_country_col, cfg.continent_col]
        missing = [c for c in required if c not in meta.columns]
        if missing:
            raise KeyError(f"Metadata is missing required columns: {missing}")

        optional = [cfg.subregion_col, cfg.neighbors_col, cfg.meta_code_col,
                    cfg.meta_lat_col, cfg.meta_lon_col]
        self.report["meta_missing_optional_cols"] = [c for c in optional if c and c not in meta.columns]
        self.scalar_attrs = [a for a in cfg.scalar_attrs if a in meta.columns]
        self.list_attrs = [a for a in cfg.list_attrs if a in meta.columns]
        self.report["meta_missing_attr_cols"] = [a for a in (*cfg.scalar_attrs, *cfg.list_attrs)
                                                 if a not in meta.columns]

        meta["_country"] = meta[cfg.meta_country_col].map(self._clean_name)
        n0 = len(meta)
        meta = meta.dropna(subset=["_country"])
        dup = meta["_country"].duplicated()
        self.report["meta_duplicate_countries"] = meta.loc[dup, "_country"].tolist()
        meta = meta[~dup].set_index("_country")
        meta.index.name = "country"
        self.report["meta_rows_dropped"] = n0 - len(meta)

        meta["continent"] = meta[cfg.continent_col]
        meta["subregion"] = meta[cfg.subregion_col] if (cfg.subregion_col and cfg.subregion_col in meta.columns) else np.nan

        # neighbours -> list of metadata country names
        self.has_neighbors = bool(cfg.neighbors_col and cfg.neighbors_col in meta.columns)
        if self.has_neighbors:
            code_map = {}
            if cfg.meta_code_col and cfg.meta_code_col in meta.columns:
                code_map = {str(c).strip(): name for name, c in meta[cfg.meta_code_col].items()
                            if not _is_missing(c)}
            known = set(meta.index)
            unknown: set = set()

            def parse(x):
                out = []
                for t in _to_list(x, cfg.list_sep):
                    t = code_map.get(t, cfg.name_map.get(t, t))
                    if t in known:
                        out.append(t)
                    else:
                        unknown.add(t)
                return out

            meta["_neighbors"] = [parse(x) for x in meta[cfg.neighbors_col]]
            self.report["unknown_neighbor_names"] = sorted(unknown)
        else:
            meta["_neighbors"] = [[] for _ in range(len(meta))]

        for a in self.list_attrs:
            meta[f"_set_{a}"] = [set(_to_list(x, cfg.list_sep)) for x in meta[a]]

        self.has_coords = bool(cfg.meta_lat_col and cfg.meta_lon_col
                               and cfg.meta_lat_col in meta.columns and cfg.meta_lon_col in meta.columns)
        return meta

    # ------------------------------------------------------------------ step 2+3
    def _prepare_images(self, meta: pd.DataFrame):
        cfg = self.cfg
        df = self.predictions.copy()
        missing = [c for c in (cfg.true_col, cfg.pred_col) if c not in df.columns]
        if missing:
            raise KeyError(f"Prediction table is missing required columns: {missing}")
        self.report["n_rows_input"] = len(df)

        df["true_country"] = df[cfg.true_col].map(self._clean_name)
        df["pred_country"] = df[cfg.pred_col].map(self._clean_name)

        na = df["true_country"].isna() | df["pred_country"].isna()
        self.report["rows_missing_labels"] = int(na.sum())
        df = df[~na]

        if cfg.image_id_col and cfg.image_id_col in df.columns:
            dup = df[cfg.image_id_col].duplicated()
            self.report["rows_duplicate_id"] = int(dup.sum())
            df = df[~dup]

        known = set(meta.index)
        unk_true = ~df["true_country"].isin(known)
        self.report["unknown_true_countries"] = sorted(df.loc[unk_true, "true_country"].unique())
        self.report["rows_unknown_true_dropped"] = int(unk_true.sum())
        df = df[~unk_true].copy()

        unk_pred = ~df["pred_country"].isin(known)
        self.report["unknown_pred_countries"] = sorted(df.loc[unk_pred, "pred_country"].unique())
        self.report["rows_unknown_pred_kept"] = int(unk_pred.sum())
        df["pred_in_meta"] = ~unk_pred

        # --- top-k list ---------------------------------------------------------
        if cfg.topk_cols:
            cols = [c for c in cfg.topk_cols if c in df.columns]
            raw = [list(r) for r in df[cols].itertuples(index=False)]
        elif cfg.topk_col and cfg.topk_col in df.columns:
            raw = [_to_list(x, cfg.topk_sep) for x in df[cfg.topk_col]]
        else:
            raw = None

        if raw is None:
            df["_topk"] = [[p] for p in df["pred_country"]]
        else:
            df["_topk"] = [[c for c in (self._clean_name(v) for v in lst) if isinstance(c, str)]
                           for lst in raw]
            self.report["rows_topk_first_differs_from_pred"] = int(sum(
                (len(l) == 0 or l[0] != p) for l, p in zip(df["_topk"], df["pred_country"])))

        lens = df["_topk"].map(len)
        typical_len = int(lens.median()) if len(df) else 1
        ks = sorted({1, *[k for k in cfg.ks if k <= max(1, typical_len)]})
        self.report["ks_used"] = ks
        self.report["ks_dropped"] = [k for k in cfg.ks if k not in ks]
        self.report["rows_topk_shorter_than_max_k"] = int((lens < max(ks)).sum()) if raw is not None else 0

        df["correct_top1"] = df["true_country"] == df["pred_country"]
        for k in ks:
            if k > 1:
                df[f"correct_top{k}"] = df["correct_top1"].to_numpy() | np.array(
                    [t in l[:k] for t, l in zip(df["true_country"], df["_topk"])], dtype=bool)
        df["is_error"] = ~df["correct_top1"]

        for side in ("true", "pred"):
            df[f"{side}_continent"] = df[f"{side}_country"].map(meta["continent"])
            df[f"{side}_subregion"] = df[f"{side}_country"].map(meta["subregion"])

        self.report["n_images_used"] = len(df)
        return df.reset_index(drop=True), ks

    # ------------------------------------------------------------------ pair features
    def _pair_matrices(self, meta: pd.DataFrame, index: dict):
        m = len(meta)
        mats: dict = {}

        def same(values) -> np.ndarray:
            v = np.asarray(values, dtype=object)
            known = np.array([not _is_missing(x) for x in v])
            eq = (v[:, None] == v[None, :]).astype(float)
            eq[~(known[:, None] & known[None, :])] = np.nan
            return eq

        mats["same_continent"] = same(meta["continent"])
        if meta["subregion"].notna().any():
            mats["same_subregion"] = same(meta["subregion"])

        if self.has_neighbors:
            adj = np.zeros((m, m), dtype=bool)
            for c, nbs in meta["_neighbors"].items():
                for nb in nbs:
                    adj[index[c], index[nb]] = adj[index[nb], index[c]] = True
            np.fill_diagonal(adj, False)
            mats["is_neighbor"] = adj.astype(float)
            mats["hop_distance"] = self._hops(adj)

        for a in self.scalar_attrs:
            mats[f"same_{a}"] = same(meta[a])

        for a in self.list_attrs:
            sets = meta[f"_set_{a}"].tolist()
            vocab = sorted(set().union(*sets)) if sets else []
            pos = {v: i for i, v in enumerate(vocab)}
            M = np.zeros((m, len(vocab)))
            for i, s in enumerate(sets):
                for v in s:
                    M[i, pos[v]] = 1
            shared = ((M @ M.T) > 0).astype(float)
            empty = M.sum(axis=1) == 0
            shared[empty, :] = np.nan
            shared[:, empty] = np.nan
            mats[f"shares_{a}"] = shared

        if self.has_coords:
            lat = pd.to_numeric(meta[self.cfg.meta_lat_col], errors="coerce")
            lon = pd.to_numeric(meta[self.cfg.meta_lon_col], errors="coerce")
            mats["km_distance"] = _haversine_matrix(lat, lon)

        numeric = [f for f in ("hop_distance", "km_distance") if f in mats]
        return mats, numeric

    @staticmethod
    def _hops(adj: np.ndarray) -> np.ndarray:
        """Shortest border path between all countries (NaN = unreachable, e.g. islands)."""
        m = adj.shape[0]
        nbrs = [np.flatnonzero(adj[i]) for i in range(m)]
        H = np.full((m, m), np.nan)
        for s in range(m):
            H[s, s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                for v in nbrs[u]:
                    if np.isnan(H[s, v]):
                        H[s, v] = H[s, u] + 1
                        q.append(v)
        return H

    def _add_pair_features(self, images, index, mats):
        """Attach pair features of (true, pred) to every error row; NaN for correct rows."""
        ok = (images["is_error"] & images["pred_in_meta"]).to_numpy()
        ti = images.loc[ok, "true_country"].map(index).to_numpy(dtype=int)
        pi = images.loc[ok, "pred_country"].map(index).to_numpy(dtype=int)
        for f, M in mats.items():
            col = np.full(len(images), np.nan)
            col[ok] = M[ti, pi]
            images[f] = col
        return images

    # ------------------------------------------------------------------ step 4
    def _country_stats(self, images, meta, ks) -> pd.DataFrame:
        cfg = self.cfg
        g = images.groupby("true_country")
        cs = pd.DataFrame({"n": g.size()})
        for k in ks:
            cs[f"errors_top{k}"] = cs["n"] - g[f"correct_top{k}"].sum()
            cs[f"error_rate_top{k}"] = cs[f"errors_top{k}"] / cs["n"]
        lo, hi = wilson_ci(cs["errors_top1"], cs["n"], cfg.ci_level)
        cs["ci_low_top1"], cs["ci_high_top1"] = lo, hi

        pg = images.groupby("pred_country")
        ps = pd.DataFrame({"n_predicted": pg.size(), "tp": pg["correct_top1"].sum()})
        ps["fp"] = ps["n_predicted"] - ps["tp"]

        src = images[images["is_error"]].groupby(["pred_country", "true_country"]).size()
        rows = {}
        for b, s in src.groupby(level=0):
            s = s.droplevel(0).sort_values(ascending=False)
            rows[b] = {"fp_distinct_origins": len(s),
                       "fp_top_source": s.index[0],
                       "fp_top_source_share": s.iloc[0] / s.sum(),
                       "fp_source_entropy_bits": _entropy_bits(s.values)}
        src_df = pd.DataFrame.from_dict(rows, orient="index",
                                        columns=["fp_distinct_origins", "fp_top_source",
                                                 "fp_top_source_share", "fp_source_entropy_bits"])
        ps = ps.join(src_df)

        cs = cs.join(ps, how="outer")
        cs.index.name = "country"
        for c in ("n", "n_predicted", "tp", "fp", "fp_distinct_origins"):
            cs[c] = cs[c].fillna(0).astype(int)
        cs["in_test"] = cs["n"] > 0
        cs["below_min_n"] = cs["n"] < cfg.min_n
        cs["precision"] = cs["tp"] / cs["n_predicted"].replace(0, np.nan)
        cs["prediction_ratio"] = cs["n_predicted"] / cs["n"].replace(0, np.nan)
        cs["continent"] = cs.index.map(meta["continent"])
        cs["subregion"] = cs.index.map(meta["subregion"])

        front = ["continent", "subregion", "n", "in_test", "below_min_n"]
        cs = cs[front + [c for c in cs.columns if c not in front]]
        return cs.reset_index().sort_values("n", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------ step 5
    def _confusion_pairs(self, images, country_stats, meta, index, mats) -> pd.DataFrame:
        err = images[images["is_error"]]
        p = err.groupby(["true_country", "pred_country"]).size().rename("count").reset_index()
        cs = country_stats.set_index("country")

        p["n_true"] = p["true_country"].map(cs["n"])
        p["errors_true"] = p["true_country"].map(cs["errors_top1"])
        p["fp_pred"] = p["pred_country"].map(cs["fp"])
        p["p_pred_given_true"] = p["count"] / p["n_true"]               # P(pred B | true A)
        p["p_pred_given_true_error"] = p["count"] / p["errors_true"]    # P(pred B | A wrong)
        p["p_true_given_pred_error"] = p["count"] / p["fp_pred"]        # P(true A | pred B, wrong)

        rev = dict(zip(zip(p["true_country"], p["pred_country"]), p["count"]))
        p["reverse_count"] = [rev.get((b, a), 0) for a, b in zip(p["true_country"], p["pred_country"])]
        p["asymmetry"] = (p["count"] - p["reverse_count"]) / (p["count"] + p["reverse_count"])

        p["true_continent"] = p["true_country"].map(meta["continent"])
        p["pred_continent"] = p["pred_country"].map(meta["continent"])

        ok = p["pred_country"].isin(index).to_numpy()
        ti = p.loc[ok, "true_country"].map(index).to_numpy(dtype=int)
        pi = p.loc[ok, "pred_country"].map(index).to_numpy(dtype=int)
        for f, M in mats.items():
            col = np.full(len(p), np.nan)
            col[ok] = M[ti, pi]
            p[f] = col

        return p.sort_values("count", ascending=False).reset_index(drop=True)


# =============================================================================
# 2) Analyser
# =============================================================================
class Q1Analyser:
    """Statistics for Q1. All methods return DataFrames or dicts (no printing).

    a) difficulty_ranking / topk_comparison / region_comparison
    b) top_confusion_pairs / per_country_confusions
    c) attractors / attractor_sources / phantom_predictions
    d) feature_baseline (shared attributes, global or within-continent)
    e) region_locality / hop_distribution / morans_i (geography)
    """

    def __init__(self, data: Q1Data):
        self.data = data
        self.cfg = data.config
        self.cs = data.country_stats
        self.pairs = data.confusion_pairs
        self.images = data.images

    # ------------------------------------------------------------------ helpers
    def _min_n(self, min_n):
        return max(1, self.cfg.min_n if min_n is None else min_n)

    def _check_k(self, k):
        if k not in self.data.ks:
            raise ValueError(f"k={k} not available. Available: {self.data.ks}")

    def _eligible(self, min_n=None) -> pd.DataFrame:
        return self.cs[self.cs["n"] >= self._min_n(min_n)].copy()

    # ------------------------------------------------------------------ a) difficulty
    def difficulty_ranking(self, k: int = 1, min_n: int | None = None) -> pd.DataFrame:
        """Countries ranked by top-k error rate (only countries with n >= min_n)."""
        self._check_k(k)
        cols = ["country", "continent", "n", f"errors_top{k}", f"error_rate_top{k}"]
        if k == 1:
            cols += ["ci_low_top1", "ci_high_top1"]
        df = self._eligible(min_n)[cols].sort_values(f"error_rate_top{k}", ascending=False)
        df = df.reset_index(drop=True)
        df.index = df.index + 1
        df.index.name = "rank"
        return df

    def topk_comparison(self, min_n: int | None = None) -> pd.DataFrame:
        """Error rate and rank at every k. recovered_in_topK = share of top-1 errors
        that are inside the top-K (near misses). Spearman rho in df.attrs."""
        ks = self.data.ks
        df = self._eligible(min_n)[["country", "continent", "n"] + [f"error_rate_top{k}" for k in ks]]
        for k in ks:
            df[f"rank_top{k}"] = df[f"error_rate_top{k}"].rank(ascending=False, method="min")
        kmax = max(ks)
        if kmax > 1:
            df[f"rank_change_top1_to_top{kmax}"] = df[f"rank_top{kmax}"] - df["rank_top1"]
            with np.errstate(divide="ignore", invalid="ignore"):
                df[f"recovered_in_top{kmax}"] = 1 - df[f"error_rate_top{kmax}"] / df["error_rate_top1"].replace(0, np.nan)
            df.attrs["spearman"] = {f"top1_vs_top{k}": df["error_rate_top1"].rank().corr(df[f"error_rate_top{k}"].rank())
                                    for k in ks if k > 1}
        return df.sort_values("error_rate_top1", ascending=False).reset_index(drop=True)

    def region_comparison(self, level: str = "continent", k: int = 1,
                          eligible_only: bool = True, min_n: int | None = None) -> pd.DataFrame:
        """Macro (mean of country error rates), micro (pooled) and spread per region."""
        self._check_k(k)
        cs = self._eligible(min_n) if eligible_only else self.cs[self.cs["in_test"]].copy()
        cs = cs.dropna(subset=[level]).reset_index(drop=True)
        rate, errs = f"error_rate_top{k}", f"errors_top{k}"
        g = cs.groupby(level)
        out = pd.DataFrame({
            "n_countries": g.size(),
            "n_images": g["n"].sum(),
            "macro_error_rate": g[rate].mean(),
            "macro_std": g[rate].std(),
            "micro_error_rate": g[errs].sum() / g["n"].sum(),
            "min_error_rate": g[rate].min(),
            "max_error_rate": g[rate].max(),
        })
        out["easiest_country"] = cs.loc[g[rate].idxmin().values, "country"].values
        out["hardest_country"] = cs.loc[g[rate].idxmax().values, "country"].values
        return out.sort_values("macro_error_rate", ascending=False)

    # ------------------------------------------------------------------ b) confusions
    def top_confusion_pairs(self, n: int = 20, min_count: int = 1,
                            sort_by: str = "count") -> pd.DataFrame:
        """Strongest directed pairs A -> B."""
        p = self.pairs[self.pairs["count"] >= min_count]
        return p.sort_values(sort_by, ascending=False).head(n).reset_index(drop=True)

    def per_country_confusions(self, top: int = 3, min_n: int | None = None) -> pd.DataFrame:
        """For each true country: its top-N wrong predictions."""
        p = self.pairs[self.pairs["n_true"] >= self._min_n(min_n)]
        p = p.sort_values(["true_country", "count"], ascending=[True, False])
        p = p.groupby("true_country").head(top).copy()
        p["rank"] = p.groupby("true_country").cumcount() + 1
        return p[["true_country", "rank", "pred_country", "count",
                  "p_pred_given_true", "p_pred_given_true_error"]].reset_index(drop=True)

    # ------------------------------------------------------------------ c) attractors
    def attractors(self, min_n: int | None = None, sort_by: str = "prediction_ratio") -> pd.DataFrame:
        """prediction_ratio > 1 = predicted more often than it truly occurs."""
        cols = ["country", "continent", "n", "n_predicted", "tp", "fp", "precision",
                "prediction_ratio", "fp_distinct_origins", "fp_top_source",
                "fp_top_source_share", "fp_source_entropy_bits"]
        return self._eligible(min_n)[cols].sort_values(sort_by, ascending=False).reset_index(drop=True)

    def attractor_sources(self, country: str, top: int = 10) -> pd.DataFrame:
        """Where the false positives of one country come from."""
        p = self.pairs[self.pairs["pred_country"] == country]
        return p[["true_country", "count", "p_true_given_pred_error", "p_pred_given_true"]] \
            .head(top).reset_index(drop=True)

    def phantom_predictions(self) -> pd.DataFrame:
        """Countries the model predicts that have no test images (pure false positives)."""
        cs = self.cs[~self.cs["in_test"] & (self.cs["n_predicted"] > 0)]
        return cs[["country", "continent", "n_predicted", "fp_distinct_origins", "fp_top_source"]] \
            .sort_values("n_predicted", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------ baseline machinery
    def _pool(self) -> np.ndarray:
        d = self.data
        if self.cfg.baseline_pool == "metadata":
            names = d.countries
        else:
            obs = set(self.images["true_country"]) | set(self.images.loc[self.images["pred_in_meta"], "pred_country"])
            names = [c for c in d.countries if c in obs]
        return np.array([d.country_index[c] for c in names], dtype=int)

    def _error_pairs(self, weighting: str = "error"):
        """(a, b) matrix indices of errors. weighting='error' counts every image,
        'pair' counts every distinct A -> B pair once."""
        e = self.images[self.images["is_error"] & self.images["pred_in_meta"]]
        if weighting == "pair":
            e = e.drop_duplicates(["true_country", "pred_country"])
        elif weighting != "error":
            raise ValueError("weighting must be 'error' or 'pair'")
        idx = self.data.country_index
        return (e["true_country"].map(idx).to_numpy(dtype=int),
                e["pred_country"].map(idx).to_numpy(dtype=int))

    def _make_sampler(self, a: np.ndarray, within_continent: bool):
        """Return (sample(rng) -> random b != a, valid mask)."""
        pool = self._pool()
        if not within_continent:
            pos_map = {x: i for i, x in enumerate(pool)}
            valid = np.array([x in pos_map for x in a], dtype=bool)
            pos = np.array([pos_map.get(x, 0) for x in a], dtype=int)
            P = len(pool)

            def sample(rng):
                r = rng.integers(0, P - 1, size=len(a))
                return pool[r + (r >= pos)]
            return sample, valid & (P > 1)

        cont = self.data.meta["continent"].to_numpy(dtype=object)
        valid = np.zeros(len(a), dtype=bool)
        plan = []
        for c in pd.unique(cont[pool]):
            if _is_missing(c):
                continue
            members = pool[cont[pool] == c]
            if len(members) < 2:
                continue
            sel = np.flatnonzero(cont[a] == c)
            sel = sel[np.isin(a[sel], members)]
            if len(sel) == 0:
                continue
            pos_map = {x: i for i, x in enumerate(members)}
            pos = np.array([pos_map[x] for x in a[sel]], dtype=int)
            valid[sel] = True
            plan.append((sel, members, pos))

        def sample(rng):
            b = np.full(len(a), -1, dtype=int)
            for sel, members, pos in plan:
                r = rng.integers(0, len(members) - 1, size=len(sel))
                b[sel] = members[r + (r >= pos)]
            return b
        return sample, valid

    # ------------------------------------------------------------------ d) shared attributes
    def feature_baseline(self, features: list | None = None, within_continent: bool = False,
                         weighting: str = "error", n_permutations: int | None = None,
                         seed: int | None = None) -> pd.DataFrame:
        """Do confused pairs share attribute X more often than random pairs?

        For each error (true A, predicted B), the baseline replaces B by a random
        country != A (from the whole pool, or from A's continent if within_continent).
        within_continent=True only uses errors where A and B are on the same continent,
        which controls for continent as a confounder.

        Returns observed share (or mean distance), bootstrap CI, baseline mean and
        null interval, lift = observed / baseline (with CI) and a one-sided p-value
        (shares: observed larger than random; distances: observed smaller).
        """
        mats = self.data.pair_matrices
        numeric = set(self.data.numeric_features)
        if features is None:
            features = [f for f in mats if not (within_continent and f == "same_continent")]
        n_perm = n_permutations or self.cfg.n_permutations
        rng = np.random.default_rng(self.cfg.seed if seed is None else seed)
        q_lo, q_hi = (1 - self.cfg.ci_level) / 2, 1 - (1 - self.cfg.ci_level) / 2

        a, b = self._error_pairs(weighting)
        if within_continent:
            keep = mats["same_continent"][a, b] == 1
            a, b = a[keep], b[keep]
        _, valid = self._make_sampler(a, within_continent)
        a, b = a[valid], b[valid]
        if len(a) == 0:
            return pd.DataFrame()
        sample, _ = self._make_sampler(a, within_continent)

        vals = {f: mats[f][a, b] for f in features}
        obs = np.array([_nanmean(vals[f]) for f in features])
        base = np.empty((n_perm, len(features)))
        boot = np.empty((n_perm, len(features)))
        for i in range(n_perm):
            br = sample(rng)
            bi = rng.integers(0, len(a), size=len(a))
            for j, f in enumerate(features):
                base[i, j] = _nanmean(mats[f][a, br])
                boot[i, j] = _nanmean(vals[f][bi])

        rows = []
        for j, f in enumerate(features):
            bj, oj = base[:, j], boot[:, j]
            bm = np.float64(_nanmean(bj))   # numpy float -> x/0 gives inf/nan, not an error
            is_dist = f in numeric
            hits = np.sum(bj <= obs[j]) if is_dist else np.sum(bj >= obs[j])
            with np.errstate(divide="ignore", invalid="ignore"):
                rows.append({
                    "feature": f,
                    "kind": "mean distance" if is_dist else "share",
                    "n_errors": len(a),
                    "n_known": int(np.sum(~np.isnan(vals[f]))),
                    "observed": obs[j],
                    "observed_ci_low": _nanq(oj, q_lo),
                    "observed_ci_high": _nanq(oj, q_hi),
                    "baseline": bm,
                    "baseline_null_low": _nanq(bj, q_lo),
                    "baseline_null_high": _nanq(bj, q_hi),
                    "lift": obs[j] / bm,
                    "lift_ci_low": _nanq(oj, q_lo) / bm,
                    "lift_ci_high": _nanq(oj, q_hi) / bm,
                    "p_value": np.nan if np.isnan(obs[j]) else (hits + 1) / (n_perm + 1),
                    "test": "observed < random" if is_dist else "observed > random",
                })
        out = pd.DataFrame(rows)
        out.attrs.update(within_continent=within_continent, weighting=weighting,
                         n_permutations=n_perm, baseline_pool=self.cfg.baseline_pool)
        return out

    # ------------------------------------------------------------------ e) geography
    def _category_shares(self, catmat: np.ndarray, labels: list, weighting: str) -> pd.DataFrame:
        """Observed share of errors per category vs. the exact expected share for random
        b != a drawn from the pool (same a-distribution as the errors)."""
        a, b = self._error_pairs(weighting)
        pool = self._pool()
        L = len(labels)
        codes = catmat[a, b]
        known = codes >= 0
        obs = np.bincount(codes[known], minlength=L) / max(known.sum(), 1)

        base, total = np.zeros(L), 0
        uniq, cnt = np.unique(a, return_counts=True)
        for x, w in zip(uniq, cnt):
            c = catmat[x, pool[pool != x]]
            c = c[c >= 0]
            if len(c):
                base += w * np.bincount(c, minlength=L) / len(c)
                total += w
        base = base / total if total else base * np.nan
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = obs / base
        return pd.DataFrame({"category": labels, "observed_share": obs,
                             "baseline_share": base, "ratio": ratio,
                             "n_errors": int(known.sum())})

    def region_locality(self, weighting: str = "error") -> pd.DataFrame:
        """Share of errors in the same subregion / same continent / other continent vs. random."""
        mats = self.data.pair_matrices
        sc = mats["same_continent"]
        cat = np.where(np.isnan(sc), -1, np.where(sc == 1, 1, 2)).astype(int)
        if "same_subregion" in mats:
            ss = mats["same_subregion"]
            cat = np.where((sc == 1) & (ss == 1), 0, cat)
            labels = ["same subregion", "same continent, other subregion", "other continent"]
        else:
            cat = cat - 1
            cat[cat < 0] = -1
            labels = ["same continent", "other continent"]
        np.fill_diagonal(cat, -1)
        return self._category_shares(cat, labels, weighting)

    def hop_distribution(self, max_hops: int = 5, weighting: str = "error") -> pd.DataFrame:
        """Border hops between true and predicted country vs. random pairs.
        'no land path' covers islands / countries not connected by borders."""
        if "hop_distance" not in self.data.pair_matrices:
            raise ValueError("No neighbour data -> hop distance not available.")
        H = self.data.pair_matrices["hop_distance"]
        cat = np.where(np.isnan(H), max_hops + 1,
                       np.where(H > max_hops, max_hops, H - 1)).astype(int)
        np.fill_diagonal(cat, -1)
        labels = [str(h) for h in range(1, max_hops + 1)] + [f">{max_hops}", "no land path"]
        return self._category_shares(cat, labels, weighting)

    def morans_i(self, k: int = 1, min_n: int | None = None,
                 n_permutations: int | None = None, seed: int | None = None) -> dict:
        """Spatial autocorrelation of country error rates on the neighbour graph.
        I > expected and small p = hard countries tend to border hard countries."""
        self._check_k(k)
        mats = self.data.pair_matrices
        if "is_neighbor" not in mats:
            raise ValueError("No neighbour data -> Moran's I not available.")
        cs = self._eligible(min_n)
        cs = cs[cs["country"].isin(self.data.country_index)]
        ids = cs["country"].map(self.data.country_index).to_numpy(dtype=int)
        x = cs[f"error_rate_top{k}"].to_numpy(dtype=float)
        W = mats["is_neighbor"][np.ix_(ids, ids)]
        has = W.sum(axis=1) > 0
        W, x = W[np.ix_(has, has)], x[has]
        N, S0 = len(x), W.sum()
        if N < 3 or S0 == 0:
            return {"morans_i": np.nan, "note": "too few connected countries"}

        def stat(v):
            z = v - v.mean()
            return (N / S0) * (z @ W @ z) / (z @ z)

        n_perm = n_permutations or self.cfg.n_permutations
        rng = np.random.default_rng(self.cfg.seed if seed is None else seed)
        obs = stat(x)
        perm = np.array([stat(rng.permutation(x)) for _ in range(n_perm)])
        return {"morans_i": obs, "expected_i": -1 / (N - 1),
                "p_value": (np.sum(perm >= obs) + 1) / (n_perm + 1),
                "n_countries": N, "n_excluded_no_eligible_neighbour": int((~has).sum()),
                "k": k, "min_n": self._min_n(min_n)}

    # ------------------------------------------------------------------ all at once
    def run_all(self) -> dict:
        has_nb = "is_neighbor" in self.data.pair_matrices
        return {
            "difficulty": self.difficulty_ranking(),
            "topk_comparison": self.topk_comparison(),
            "continents": self.region_comparison("continent"),
            "top_pairs": self.top_confusion_pairs(),
            "per_country_confusions": self.per_country_confusions(),
            "attractors": self.attractors(),
            "phantom_predictions": self.phantom_predictions(),
            "features_global": self.feature_baseline(),
            "features_within_continent": self.feature_baseline(within_continent=True),
            "region_locality": self.region_locality(),
            "hop_distribution": self.hop_distribution() if has_nb else None,
            "morans_i": self.morans_i() if has_nb else None,
        }


# =============================================================================
# 3) Reporter
# =============================================================================
class Q1Reporter:
    """Basic tables and plots on top of Q1Analyser."""

    def __init__(self, analyser: Q1Analyser):
        self.a = analyser
        self.d = analyser.data
        self.country_names = self.d.meta["prompt_name"].to_dict()

    # ------------------------------------------------------------------ tables
    def show(self, obj, title: str | None = None, n: int | None = 15):
        if title:
            print(f"\n=== {title} ===")

        if isinstance(obj, pd.DataFrame):
            obj = obj.copy()

            # Convert country codes to readable names (display only)
            country_cols = [
                "country", "true_country", "pred_country",
                "fp_top_source", "easiest_country", "hardest_country"
            ]

            for col in country_cols:
                if col in obj.columns:
                    obj[col] = obj[col].map(
                        lambda c: f"{self.country_names.get(c, c)} ({c})"
                        if pd.notna(c) else c
                    )

            if n:
                obj = obj.head(n)

        try:
            from IPython.display import display
            display(obj)
        except ImportError:
            print(obj.to_string() if hasattr(obj, "to_string") else obj)

    def summary(self):
        r, im, cs = self.d.report, self.d.images, self.d.country_stats
        print("Q1 data summary")
        print(f"  images used:          {len(im):,} (input rows {r.get('n_rows_input', 0):,})")
        print(f"  countries in test:    {int(cs['in_test'].sum())} "
              f"({int((cs['in_test'] & ~cs['below_min_n']).sum())} with n >= {self.d.config.min_n})")
        print(f"  top-k available:      {self.d.ks}")
        for k in self.d.ks:
            print(f"  overall error top-{k}:  {1 - im[f'correct_top{k}'].mean():.3f}")
        flags = {k: v for k, v in r.items()
                 if k not in ("n_rows_input", "n_images_used", "ks_used") and v not in (0, [], None)}
        if flags:
            print("  validation notes:")
            for k, v in flags.items():
                print(f"    - {k}: {v}")

    # ------------------------------------------------------------------ plots
    def plot_difficulty(self, k: int = 1, top: int = 20, min_n: int | None = None):
        import matplotlib.pyplot as plt
        df = self.a.difficulty_ranking(k, min_n).head(top).iloc[::-1]
        fig, ax = plt.subplots(figsize=(7, 0.3 * len(df) + 1.5))
        rate = df[f"error_rate_top{k}"]
        xerr = None
        if k == 1:
            xerr = [rate - df["ci_low_top1"], df["ci_high_top1"] - rate]
        labels = [
            f"{self.country_names.get(c, c)} ({c}, n={n})"
            for c, n in zip(df["country"], df["n"])
        ]
        ax.barh(labels, rate, xerr=xerr, capsize=2)
        ax.set_xlabel(f"Top-{k} error rate")
        ax.set_title(f"Hardest countries (top-{k}, n >= {self.a._min_n(min_n)})")
        fig.tight_layout()
        return fig

    def plot_regions(self, level: str = "continent", k: int = 1):
        import matplotlib.pyplot as plt
        df = self.a.region_comparison(level, k)
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(df))
        ax.bar(x - 0.2, df["macro_error_rate"], 0.4, yerr=df["macro_std"], capsize=3, label="macro (mean of countries)")
        ax.bar(x + 0.2, df["micro_error_rate"], 0.4, label="micro (pooled)")
        ax.set_xticks(x, df.index, rotation=30, ha="right")
        ax.set_ylabel(f"Top-{k} error rate")
        ax.set_title(f"Error rate per {level}")
        ax.legend()
        fig.tight_layout()
        return fig

    def plot_feature_lift(self, result: pd.DataFrame | None = None, within_continent: bool = False):
        import matplotlib.pyplot as plt
        df = result if result is not None else self.a.feature_baseline(within_continent=within_continent)
        df = df[df["kind"] == "share"]
        fig, ax = plt.subplots(figsize=(7, 0.4 * len(df) + 1.5))
        err = [df["lift"] - df["lift_ci_low"], df["lift_ci_high"] - df["lift"]]
        ax.barh(df["feature"], df["lift"], xerr=err, capsize=3)
        ax.axvline(1, color="k", lw=1, ls="--")
        ax.set_xlabel("Lift (observed share / random-pair share)")
        scope = "within continent" if df.attrs.get("within_continent", within_continent) else "global"
        ax.set_title(f"Shared attributes of confused pairs ({scope} baseline)")
        fig.tight_layout()
        return fig

    def plot_hop_distribution(self, result: pd.DataFrame | None = None):
        import matplotlib.pyplot as plt
        df = result if result is not None else self.a.hop_distribution()
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(df))
        ax.bar(x - 0.2, df["observed_share"], 0.4, label="errors")
        ax.bar(x + 0.2, df["baseline_share"], 0.4, label="random pairs")
        ax.set_xticks(x, df["category"])
        ax.set_xlabel("Border hops between true and predicted country")
        ax.set_ylabel("Share")
        ax.legend()
        fig.tight_layout()
        return fig

    def plot_attractors(self, top: int = 15, min_n: int | None = None):
        import matplotlib.pyplot as plt
        df = self.a.attractors(min_n).head(top).iloc[::-1]
        fig, ax = plt.subplots(figsize=(7, 0.3 * len(df) + 1.5))
        labels = [f"{self.country_names.get(c, c)} ({c}, n={n})" for c, n in zip(df["country"], df["n"])]
        ax.barh(labels, df["prediction_ratio"])
        ax.axvline(1, color="k", lw=1, ls="--")
        ax.set_xlabel("Prediction ratio (# predicted / # true)")
        ax.set_title("Attractor countries")
        fig.tight_layout()
        return fig


# =============================================================================
# 4) Interactive dashboard (Dash + Plotly)
#    Needs: dash >= 2.11, plotly. Imported only when Q1Dashboard is used.
# =============================================================================
_CONTINENT_NAMES = {"AF": "Africa", "AN": "Antarctica", "AS": "Asia", "EU": "Europe",
                    "NA": "North America", "OC": "Oceania", "SA": "South America"}

_FEATURE_LABELS = {
    "same_continent": "Same continent",
    "same_subregion": "Same subregion",
    "is_neighbor": "Neighbours (share a border)",
    "hop_distance": "Border hops (mean, lower = closer)",
    "km_distance": "Distance in km (mean, lower = closer)",
    "same_driving_side": "Same driving side",
    "same_landlocked": "Same landlocked status",
    "shares_languages": "Share a language",
}

_COL_LABELS = {
    "name": "Country", "true_name": "True country", "pred_name": "Predicted country",
    "continent_name": "Continent", "n": "n images", "n_true": "n true images",
    "count": "Count", "p_pred_given_true": "% of origin images",
    "p_pred_given_true_error": "% of origin errors",
    "p_true_given_pred_error": "% of incoming FPs", "reverse_count": "Reverse count",
    "reverse_rate": "Reverse rate (% of B's images -> A)", "asymmetry": "Asymmetry",
    "n_predicted": "n predicted", "fp": "FP", "tp": "TP", "precision": "Precision",
    "prediction_ratio": "Prediction ratio", "fp_distinct_origins": "Distinct FP origins",
    "fp_top_source_name": "Top FP source", "fp_top_source_share": "Top source share",
    "fp_source_entropy_bits": "FP source entropy (bits)",
}

_GREY_SMALL = "#b3b3b3"     # n < min_n
_GREY_NODATA = "#e8e8e8"    # no test images
_PHANTOM = "#6a3d9a"        # predicted but never in the test set
_ORIGIN = "#1b9e77"         # selected true country (confusion mode)
_TARGET = "#d95f02"         # selected predicted country (attractor mode)
_SEQ_ERR = "OrRd"           # error rates
_SEQ_CONF = "Purples"       # confusion strength
_SEQ_ATTR = "Blues"         # attractor / FP strength


def _pct(x, d=1):
    return "–" if _is_missing(x) else f"{x:.{d}%}"


def _num(x, d=2):
    return "–" if _is_missing(x) else f"{x:,.{d}f}"


class Q1Dashboard:
    """Interactive Q1 dashboard on top of Q1Analyser (no new statistics, only views).

    Five modes: 1 country difficulty, 2 confusion pairs, 3 attractors,
    4 shared characteristics, 5 geographic clustering.

    Internal keys are the metadata country codes (e.g. ISO2), maps use ISO3,
    the UI shows full names. Baselines are computed once and cached.

    Usage (Jupyter):  Q1Dashboard(ana).run()            # inline under the cell
                      Q1Dashboard(ana).run(mode="tab")  # new browser tab
    """

    MODES = [("difficulty", "1 · Country difficulty"),
             ("confusion", "2 · Confusion pairs"),
             ("attractors", "3 · Attractor countries"),
             ("shared", "4 · Shared characteristics"),
             ("geography", "5 · Geographic clustering")]

    CONF_METRICS = {"count": ("Count (images)", False),
                    "p_pred_given_true": ("% of origin images", True),
                    "p_pred_given_true_error": ("% of origin errors", True)}
    ATTR_OVERVIEW = {"ratio": "Prediction ratio (log scale)",
                     "fp": "False positives (count)",
                     "precision": "Precision"}
    ATTR_SELECTED = {"count": ("FP count", False),
                     "p_true_given_pred_error": ("% of incoming FPs", True),
                     "p_pred_given_true": ("Origin confusion rate (% of origin images)", True)}

    # ------------------------------------------------------------------ setup
    def __init__(self, analyser: Q1Analyser, name_col: str = "prompt_name",
                 iso3_col: str = "iso3", n_permutations: int | None = None,
                 precompute: bool = True):
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        self.go, self.make_subplots = go, make_subplots

        self.a, self.d = analyser, analyser.data
        self.cfg = self.d.config
        self.n_perm = n_permutations
        meta = self.d.meta

        def col_map(col):
            if col not in meta.columns:
                return {}
            return {c: v for c, v in meta[col].items() if not _is_missing(v) and str(v).strip()}

        names = col_map(name_col)
        self.name = {c: str(names.get(c, c)) for c in meta.index}
        self.iso3 = {c: str(v).strip() for c, v in col_map(iso3_col).items()}
        if not self.iso3:   # fallback: keys already are ISO3
            self.iso3 = {c: c for c in meta.index if isinstance(c, str) and len(c) == 3}
        self.from_iso3 = {v: k for k, v in self.iso3.items()}

        self.coords = {}
        if self.cfg.meta_lat_col in meta.columns and self.cfg.meta_lon_col in meta.columns:
            lat = pd.to_numeric(meta[self.cfg.meta_lat_col], errors="coerce")
            lon = pd.to_numeric(meta[self.cfg.meta_lon_col], errors="coerce")
            self.coords = {c: (lat[c], lon[c]) for c in meta.index
                           if not (np.isnan(lat[c]) or np.isnan(lon[c]))}

        self.continent = {c: v for c, v in meta["continent"].items()}

        cs = self.d.country_stats.copy()
        cs["name"] = cs["country"].map(self.name).fillna(cs["country"])
        cs["continent_name"] = cs["continent"].map(self._cont_name)
        cs["fp_top_source_name"] = cs["fp_top_source"].map(lambda c: self.name.get(c, c) if isinstance(c, str) else c)
        self.cs = cs
        self.cs_idx = cs.set_index("country")

        p = self.d.confusion_pairs.copy()
        p["true_name"] = p["true_country"].map(lambda c: self.name.get(c, c))
        p["pred_name"] = p["pred_country"].map(lambda c: self.name.get(c, c))
        n_by = self.cs_idx["n"]
        p["reverse_rate"] = p["reverse_count"] / p["pred_country"].map(n_by).replace(0, np.nan)
        self.pairs = p

        missing_cont = int(cs.loc[cs["in_test"], "continent"].isna().sum())
        if missing_cont:
            warnings.warn(f"{missing_cont} test countries have no continent. If North America is "
                          f"missing: the code 'NA' was probably read as NaN (use keep_default_na=False).")
        no_iso3 = [c for c in cs.loc[cs["in_test"], "country"] if c not in self.iso3]
        if no_iso3:
            warnings.warn(f"No ISO3 code for {no_iso3}: these countries are not drawn on the map.")

        self._cache: dict = {}
        if precompute:
            self.precompute()

    def _cont_name(self, c):
        return "Unknown" if _is_missing(c) else _CONTINENT_NAMES.get(c, c)

    def label(self, code):
        return self.name.get(code, code)

    # ------------------------------------------------------------------ cached statistics
    def _cached(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    def _baseline(self, within, weighting):
        return self._cached(("fb", within, weighting), lambda: self.a.feature_baseline(
            within_continent=within, weighting=weighting, n_permutations=self.n_perm))

    def _locality(self, weighting):
        return self._cached(("loc", weighting), lambda: self.a.region_locality(weighting))

    def _hops(self, weighting):
        if "hop_distance" not in self.d.pair_matrices:
            return None
        return self._cached(("hop", weighting), lambda: self.a.hop_distribution(weighting=weighting))

    def _moran(self, k, min_n):
        if "is_neighbor" not in self.d.pair_matrices:
            return None
        return self._cached(("moran", k, min_n), lambda: self.a.morans_i(
            k=k, min_n=min_n, n_permutations=self.n_perm))

    def precompute(self):
        """Run all permutation-based statistics once (dashboard callbacks then only filter)."""
        for within in (False, True):
            for w in ("error", "pair"):
                self._baseline(within, w)
        for w in ("error", "pair"):
            self._locality(w)
            self._hops(w)
        for k in self.d.ks:
            self._moran(k, self.cfg.min_n)

    # ------------------------------------------------------------------ small helpers
    def _filter_cont(self, df, conts, col="continent"):
        return df[df[col].isin(conts)] if conts else df

    def _nodata_layer(self, conts):
        in_test = set(self.cs.loc[self.cs["in_test"], "country"])
        codes = [c for c in self.d.meta.index if c not in in_test
                 and (not conts or self.continent.get(c) in conts)]
        return self._grey(codes, [f"<b>{self.label(c)}</b><br>No test images" for c in codes],
                          _GREY_NODATA)

    def _grey(self, codes, text, color):
        return dict(kind="grey", codes=list(codes), text=list(text), color=color)

    def _values(self, codes, z, text, colorscale, zmin, zmax, cbar, tickvals=None,
                ticktext=None, tickformat=None):
        return dict(kind="value", codes=list(codes), z=list(z), text=list(text),
                    colorscale=colorscale, zmin=zmin, zmax=zmax,
                    cbar=dict(title=cbar, tickvals=tickvals, ticktext=ticktext, tickformat=tickformat))

    def _empty(self, msg, height=300):
        fig = self.go.Figure()
        fig.add_annotation(text=msg, showarrow=False, font=dict(size=14, color="#666"),
                           x=0.5, y=0.5, xref="paper", yref="paper")
        fig.update_layout(xaxis_visible=False, yaxis_visible=False, height=height,
                          margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="white")
        return fig

    def _map(self, layers, title, selected=None, fill=None, dots=False, note=None):
        """layers: value / grey layers. fill=(code, colour, label) colours one country.
        selected: outlined + star at its centre."""
        go = self.go
        fig = go.Figure()
        for L in layers:
            keep = [i for i, c in enumerate(L["codes"]) if c in self.iso3]
            if not keep:
                continue
            codes = [L["codes"][i] for i in keep]
            locs = [self.iso3[c] for c in codes]
            text = [L["text"][i] for i in keep]
            common = dict(locations=locs, locationmode="ISO-3", text=text, customdata=codes,
                          hovertemplate="%{text}<extra></extra>",
                          marker_line_color="white", marker_line_width=0.4)
            if L["kind"] == "grey":
                fig.add_trace(go.Choropleth(z=[0] * len(locs), showscale=False,
                                            colorscale=[[0, L["color"]], [1, L["color"]]], **common))
                continue
            z = [L["z"][i] for i in keep]
            cb = {k: v for k, v in L["cbar"].items() if v is not None and k != "title"}
            cb.update(title=dict(text=L["cbar"]["title"]), thickness=12, len=0.75)
            fig.add_trace(go.Choropleth(z=z, zmin=L["zmin"], zmax=L["zmax"],
                                        colorscale=L["colorscale"], colorbar=cb, **common))
            if dots and self.coords:
                d = [(c, t, v) for c, t, v in zip(codes, text, z) if c in self.coords]
                if d:
                    fig.add_trace(go.Scattergeo(
                        lat=[self.coords[c][0] for c, _, _ in d], lon=[self.coords[c][1] for c, _, _ in d],
                        mode="markers", text=[t for _, t, _ in d], customdata=[c for c, _, _ in d],
                        hovertemplate="%{text}<extra></extra>", showlegend=False,
                        marker=dict(size=7, color=[v for _, _, v in d], colorscale=L["colorscale"],
                                    cmin=L["zmin"], cmax=L["zmax"], line=dict(width=0.6, color="#333"))))
        if fill and fill[0] in self.iso3:
            code, colour, lab = fill
            fig.add_trace(go.Choropleth(
                locations=[self.iso3[code]], locationmode="ISO-3", z=[0], showscale=False,
                colorscale=[[0, colour], [1, colour]], text=[lab], customdata=[code],
                hovertemplate="%{text}<extra></extra>", marker_line_color="white"))
        if selected and selected in self.iso3:
            fig.add_trace(go.Choropleth(
                locations=[self.iso3[selected]], locationmode="ISO-3", z=[0], showscale=False,
                colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
                marker_line_color="black", marker_line_width=2.5, hoverinfo="skip"))
        if selected and selected in self.coords:
            lat, lon = self.coords[selected]
            fig.add_trace(go.Scattergeo(lat=[lat], lon=[lon], mode="markers", hoverinfo="skip",
                                        showlegend=False,
                                        marker=dict(symbol="star", size=14, color="black")))
        fig.update_geos(projection_type="natural earth", showcountries=True, countrycolor="white",
                        showland=True, landcolor="#f5f5f5", showframe=False, showcoastlines=False)
        fig.update_layout(title=dict(text=title, x=0.01, font=dict(size=15)), height=470,
                          margin=dict(l=0, r=0, t=45, b=0), showlegend=False)
        if note:
            fig.add_annotation(text=note, showarrow=False, x=0, y=-0.02, xref="paper", yref="paper",
                               xanchor="left", font=dict(size=11, color="#555"))
        return fig

    def _hbar(self, labels, values, codes=None, title="", xtitle="", color="#555",
              selected=None, err=None, pct=False, log=False, xrange=None, vline=None,
              hover=None, height=300):
        go = self.go
        colors = [("#000000" if (codes is not None and c == selected) else color)
                  for c in (codes if codes is not None else [None] * len(labels))]
        bar = dict(x=list(values), y=list(labels), orientation="h", marker_color=colors,
                   customdata=list(codes) if codes is not None else None)
        if hover is not None:
            bar.update(text=list(hover), hovertemplate="%{text}<extra></extra>", textposition="none")
        if err is not None:
            bar["error_x"] = dict(type="data", symmetric=False, array=list(err[1]),
                                  arrayminus=list(err[0]), thickness=1, width=2)
        fig = go.Figure(go.Bar(**bar))
        fig.update_yaxes(autorange="reversed", automargin=True)
        fig.update_xaxes(title=xtitle, type="log" if log else "linear",
                         tickformat=".0%" if pct else None)
        if xrange is not None:
            fig.update_xaxes(range=xrange)
        if vline is not None and len(labels):
            self._ref_line(fig, vline, list(labels))
        fig.update_layout(title=dict(text=title, x=0.01, font=dict(size=13)),
                          height=max(height, 22 * len(labels) + 90),
                          margin=dict(l=10, r=10, t=40, b=40), plot_bgcolor="white")
        return fig

    def _ref_line(self, fig, x, labels):
        """Dashed vertical line at x across categorical y labels (safe on log axes)."""
        fig.add_trace(self.go.Scatter(x=[x, x], y=[labels[0], labels[-1]], mode="lines", hoverinfo="skip",
                                      showlegend=False, line=dict(color="black", dash="dash", width=1)))

    def _obs_vs_base(self, df, title, label_col="category", pct=True, height=300):
        go = self.go
        fig = go.Figure([
            go.Bar(name="Errors (observed)", x=df[label_col], y=df["observed_share"], marker_color="#d62728"),
            go.Bar(name="Random pairs (baseline)", x=df[label_col], y=df["baseline_share"], marker_color="#9e9e9e"),
        ])
        fig.update_layout(barmode="group", title=dict(text=title, x=0.01, font=dict(size=13)),
                          height=height, margin=dict(l=10, r=10, t=40, b=40), plot_bgcolor="white",
                          legend=dict(orientation="h", y=1.12, x=1, xanchor="right"))
        if pct:
            fig.update_yaxes(tickformat=".0%")
        return fig

    def _table(self, df, pct=(), dec=()):
        from dash.dash_table import FormatTemplate
        from dash.dash_table.Format import Format, Scheme
        df = df.replace([np.inf, -np.inf], np.nan)
        cols = []
        for c in df.columns:
            col = {"name": _COL_LABELS.get(c, c.replace("_", " ")), "id": c}
            if c in pct:
                col.update(type="numeric", format=FormatTemplate.percentage(1))
            elif c in dec:
                col.update(type="numeric", format=Format(precision=2, scheme=Scheme.fixed))
            elif pd.api.types.is_numeric_dtype(df[c]):
                col["type"] = "numeric"
            cols.append(col)
        data = df.astype(object).where(df.notna(), None).to_dict("records")
        return data, cols

    # ================================================================== views
    # each view returns: main fig, side1 fig, side2 fig or None, table df, (pct cols, dec cols), finding (markdown)

    # ------------------------------------------------------------------ 1 difficulty
    def view_difficulty(self, sel, k, conts, min_n, auto, dots):
        rate = f"error_rate_top{k}"
        cs = self._filter_cont(self.cs[self.cs["in_test"]], conts)
        elig, small = cs[cs["n"] >= min_n], cs[cs["n"] < min_n]
        level = int(round(self.cfg.ci_level * 100))

        def hov(r, flag=""):
            s = (f"<b>{r['name']}</b> ({r['country']})<br>{r['continent_name']}<br>"
                 f"n = {r['n']:,}<br>Top-{k} error: {_pct(r[rate])}")
            if k == 1:
                s += f"<br>{level}% CI: {_pct(r['ci_low_top1'])} – {_pct(r['ci_high_top1'])}"
            return s + flag

        layers = [self._nodata_layer(conts),
                  self._grey(small["country"], [hov(r, f"<br><i>n < {min_n}: not ranked</i>")
                                                for _, r in small.iterrows()], _GREY_SMALL),
                  self._values(elig["country"], elig[rate], [hov(r) for _, r in elig.iterrows()],
                               _SEQ_ERR, None if auto else 0, None if auto else 1,
                               f"Top-{k}<br>error rate", tickformat=".0%")]
        main = self._map(layers, f"Top-{k} error rate per country", selected=sel, dots=dots,
                         note=f"Dark grey: n < {min_n} (not ranked) · light grey: no test images")

        top = elig.sort_values(rate, ascending=False).head(20)
        err = None
        if k == 1:
            err = (top[rate] - top["ci_low_top1"], top["ci_high_top1"] - top[rate])
        side1 = self._hbar([f"{n} (n={m})" for n, m in zip(top["name"], top["n"])], top[rate],
                           top["country"], f"20 hardest countries (top-{k}, n ≥ {min_n})",
                           f"Top-{k} error rate", "#d6604d", sel, err, pct=True,
                           xrange=None if auto else [0, 1])

        rc = self.a.region_comparison("continent", k, eligible_only=True, min_n=min_n)
        rc.index = [self._cont_name(c) for c in rc.index]
        go = self.go
        side2 = go.Figure([
            go.Bar(name="Macro (mean of countries)", x=rc.index, y=rc["macro_error_rate"],
                   error_y=dict(type="data", array=rc["macro_std"].fillna(0)), marker_color="#d6604d"),
            go.Bar(name="Micro (all images pooled)", x=rc.index, y=rc["micro_error_rate"], marker_color="#f4a582"),
        ])
        side2.update_layout(barmode="group", height=300, plot_bgcolor="white",
                            title=dict(text=f"Continents (top-{k}, n ≥ {min_n})", x=0.01, font=dict(size=13)),
                            margin=dict(l=10, r=10, t=40, b=40),
                            legend=dict(orientation="h", y=1.15, x=1, xanchor="right"))
        side2.update_yaxes(tickformat=".0%", range=None if auto else [0, 1])

        tk = self.a.topk_comparison(min_n)
        tk = tk[tk["country"].isin(elig["country"])]
        ks = self.d.ks
        tk = tk.merge(self.cs[["country", "name", "continent_name", "ci_low_top1", "ci_high_top1"]], on="country")
        kmax = max(ks)
        cols = ["name", "country", "continent_name", "n", "error_rate_top1", "ci_low_top1", "ci_high_top1"] + \
               [f"error_rate_top{x}" for x in ks if x > 1] + \
               ([f"recovered_in_top{kmax}"] if kmax > 1 else [])
        table = tk[cols].sort_values(f"error_rate_top{k}", ascending=False)
        pct_cols = [c for c in cols if c.startswith(("error_rate", "ci_", "recovered"))]

        lines = []
        if len(elig):
            h, e = elig.loc[elig[rate].idxmax()], elig.loc[elig[rate].idxmin()]
            lines.append(f"**Hardest:** {h['name']} – {_pct(h[rate])} top-{k} error (n = {h['n']:,}). "
                         f"**Easiest:** {e['name']} – {_pct(e[rate])} (n = {e['n']:,}).")
        if len(rc) > 1:
            lines.append(f"**Continents (macro):** highest {rc['macro_error_rate'].idxmax()} "
                         f"({_pct(rc['macro_error_rate'].max())}), lowest {rc['macro_error_rate'].idxmin()} "
                         f"({_pct(rc['macro_error_rate'].min())}).")
        rho = self.a.topk_comparison(min_n).attrs.get("spearman", {})
        if rho:
            lines.append("**Top-1 vs top-k ranking (Spearman ρ):** " +
                         ", ".join(f"{key.replace('top1_vs_', '')} {v:.2f}" for key, v in rho.items()))
        if len(small):
            lines.append(f"{len(small)} {'country' if len(small) == 1 else 'countries'} with n < {min_n} "
                         f"{'is' if len(small) == 1 else 'are'} dark grey and not ranked.")
        if sel is not None and sel in self.cs_idx.index:
            r = self.cs_idx.loc[sel]
            if r["n"] >= min_n and sel in set(elig["country"]):
                rank = int((elig[rate] > r[rate]).sum()) + 1
                lines.append(f"**Selected {self.label(sel)}:** rank {rank} of {len(elig)}, "
                             f"top-{k} error {_pct(r[rate])} (n = {int(r['n']):,}).")
            elif r["n"] > 0:
                lines.append(f"**Selected {self.label(sel)}:** only {int(r['n'])} images (< {min_n}), "
                             f"top-{k} error {_pct(r[rate])} – not reliable.")
        return main, side1, side2, table, (pct_cols, []), "  \n".join(lines)

    # ------------------------------------------------------------------ 2 confusion pairs
    def view_confusion(self, sel, conts, min_n, metric, dots):
        mlabel, is_pct = self.CONF_METRICS[metric]
        fmt = (lambda v: _pct(v)) if is_pct else (lambda v: f"{int(v):,}")
        P = self.pairs
        tcols = ["true_name", "pred_name", "true_country", "pred_country", "count", "n_true",
                 "p_pred_given_true", "p_pred_given_true_error", "reverse_count", "reverse_rate",
                 "asymmetry"] + [f for f in self.d.pair_matrices if f in P.columns]
        pct_cols = ["p_pred_given_true", "p_pred_given_true_error", "p_true_given_pred_error", "reverse_rate"]
        dec_cols = ["asymmetry", "hop_distance", "km_distance"]

        if sel is None or sel not in set(P["true_country"]):
            origins = self._filter_cont(self.cs[self.cs["in_test"]], conts)
            p = P[P["true_country"].isin(origins["country"]) & (P["n_true"] >= min_n)]
            top = p.sort_values("count", ascending=False).drop_duplicates("true_country")
            small = origins[origins["n"] < min_n]
            text = [f"<b>{r.true_name}</b><br>Strongest confusion: {r.pred_name}<br>"
                    f"{int(r.count):,} images · {_pct(r.p_pred_given_true)} of its images · "
                    f"{_pct(r.p_pred_given_true_error)} of its errors" for r in top.itertuples()]
            layers = [self._nodata_layer(conts),
                      self._grey(small["country"], [f"<b>{n}</b><br>n = {m} < {min_n}" for n, m in
                                                    zip(small["name"], small["n"])], _GREY_SMALL),
                      self._values(top["true_country"], top[metric], text, _SEQ_CONF, 0, None,
                                   f"Strongest<br>destination<br>({mlabel})",
                                   tickformat=".0%" if is_pct else None)]
            main = self._map(layers, f"Strongest confusion per origin country – {mlabel}",
                             dots=dots, note="Click a country (or search above) to see where its images go")
            rk = p.sort_values(metric, ascending=False).head(15)
            side1 = self._hbar([f"{a} → {b}" for a, b in zip(rk["true_name"], rk["pred_name"])], rk[metric],
                               rk["true_country"], f"Strongest directional pairs ({mlabel})", mlabel,
                               "#756bb1", pct=is_pct,
                               hover=[f"{a} → {b}<br>{int(c):,} images · {_pct(x)} of {a}'s images · "
                                      f"reverse {int(r):,}" for a, b, c, x, r in
                                      zip(rk["true_name"], rk["pred_name"], rk["count"],
                                          rk["p_pred_given_true"], rk["reverse_count"])])
            table = p.sort_values(metric, ascending=False)[tcols].head(500)
            lines = []
            if len(p):
                s = p.iloc[0]
                lines.append(f"**Strongest pair (count):** {s['true_name']} → {s['pred_name']}: "
                             f"{int(s['count']):,} images ({_pct(s['p_pred_given_true'])} of all "
                             f"{s['true_name']} images, {_pct(s['p_pred_given_true_error'])} of its errors).")
                s2 = p.loc[p["p_pred_given_true"].idxmax()]
                lines.append(f"**Strongest pair (% of origin images):** {s2['true_name']} → {s2['pred_name']}: "
                             f"{_pct(s2['p_pred_given_true'])}.")
            if sel is not None:
                lines.append(f"{self.label(sel)} has no top-1 errors or no test images.")
            lines.append("Top-1 predictions only. Raw counts and percentages are separate metrics (switch above).")
            return main, side1, None, table, (pct_cols, dec_cols), "  \n".join(lines)

        p = P[P["true_country"] == sel].sort_values(metric, ascending=False)
        r0 = self.cs_idx.loc[sel]
        text = [f"<b>{self.label(sel)} → {r.pred_name}</b><br>{int(r.count):,} images<br>"
                f"{_pct(r.p_pred_given_true)} of {self.label(sel)}'s images<br>"
                f"{_pct(r.p_pred_given_true_error)} of its errors<br>"
                f"Reverse ({r.pred_name} → {self.label(sel)}): {int(r.reverse_count):,}"
                f" ({_pct(r.reverse_rate)} of {r.pred_name}'s images)" for r in p.itertuples()]
        layers = [self._values(p["pred_country"], p[metric], text, _SEQ_CONF, 0, None,
                               mlabel.replace(" ", "<br>", 1), tickformat=".0%" if is_pct else None)]
        main = self._map(layers, f"Where {self.label(sel)}'s images are wrongly predicted – {mlabel}",
                         selected=sel, dots=dots,
                         fill=(sel, _ORIGIN, f"<b>{self.label(sel)}</b> (origin)<br>n = {int(r0['n']):,}, "
                                             f"top-1 error {_pct(r0['error_rate_top1'])}"),
                         note="Green = selected true country · predictions outside its region are always shown")
        top = p.head(12)
        side1 = self._hbar(top["pred_name"], top[metric], top["pred_country"],
                           f"Top destinations of {self.label(sel)} ({mlabel})", mlabel, "#756bb1",
                           pct=is_pct, hover=[f"{b}: {fmt(v)}" for b, v in zip(top["pred_name"], top[metric])])
        go = self.go
        side2 = go.Figure([
            go.Bar(name=f"{self.label(sel)} → B", y=top["pred_name"], x=top["p_pred_given_true"],
                   orientation="h", marker_color="#756bb1"),
            go.Bar(name=f"B → {self.label(sel)}", y=top["pred_name"], x=top["reverse_rate"],
                   orientation="h", marker_color="#bcbddc"),
        ])
        side2.update_layout(barmode="group", height=max(300, 30 * len(top) + 90), plot_bgcolor="white",
                            title=dict(text="Both directions (% of the origin's images)", x=0.01, font=dict(size=13)),
                            margin=dict(l=10, r=10, t=40, b=40),
                            legend=dict(orientation="h", y=1.1, x=1, xanchor="right"))
        side2.update_yaxes(autorange="reversed", automargin=True)
        side2.update_xaxes(tickformat=".0%")

        lines = [f"**{self.label(sel)}:** {int(r0['n']):,} images, {int(r0['errors_top1']):,} top-1 errors "
                 f"({_pct(r0['error_rate_top1'])}), spread over {len(p)} predicted countries."]
        if r0["n"] < min_n:
            lines.append(f"⚠ Only {int(r0['n'])} images (< {min_n}): treat with caution.")
        if len(p):
            t = p.sort_values("count", ascending=False).iloc[0]
            lines.append(f"**Top destination:** {t['pred_name']} – {int(t['count']):,} images "
                         f"({_pct(t['p_pred_given_true'])} of all images, {_pct(t['p_pred_given_true_error'])} "
                         f"of errors). Reverse: {int(t['reverse_count']):,} ({_pct(t['reverse_rate'])} of "
                         f"{t['pred_name']}'s images).")
            top3 = p.sort_values("count", ascending=False).head(3)["p_pred_given_true_error"].sum()
            lines.append(f"Top 3 destinations take {_pct(top3)} of its errors.")
        return main, side1, side2, p[tcols], (pct_cols, dec_cols), "  \n".join(lines)

    # ------------------------------------------------------------------ 3 attractors
    def view_attractors(self, sel, conts, min_n, overview, selm, dots):
        cs, P = self.cs, self.pairs
        pct_cols = ["precision", "fp_top_source_share", "p_true_given_pred_error", "p_pred_given_true"]
        dec_cols = ["prediction_ratio", "fp_source_entropy_bits"]

        def hov(r):
            return (f"<b>{r['name']}</b> ({r['country']})<br>True images: {int(r['n']):,} · "
                    f"predicted: {int(r['n_predicted']):,}<br>FP: {int(r['fp']):,} · precision "
                    f"{_pct(r['precision'])}<br>Prediction ratio: {_num(r['prediction_ratio'])}<br>"
                    f"FP origins: {int(r['fp_distinct_origins'])} · top source "
                    f"{r['fp_top_source_name'] if isinstance(r['fp_top_source_name'], str) else '–'} "
                    f"({_pct(r['fp_top_source_share'])})")

        if sel is None or sel not in set(P["pred_country"]):
            pred = cs[cs["n_predicted"] > 0]
            elig = pred[pred["n"] >= min_n]
            small = pred[(pred["n"] > 0) & (pred["n"] < min_n)]
            phantom = pred[pred["n"] == 0]
            if overview == "ratio":
                z = np.log2(np.clip(elig["prediction_ratio"].astype(float), 1 / 8, 8))
                vl = self._values(elig["country"], z, [hov(r) for _, r in elig.iterrows()], "RdBu_r", -3, 3,
                                  "Prediction<br>ratio", tickvals=[-3, -2, -1, 0, 1, 2, 3],
                                  ticktext=["≤1/8", "1/4", "1/2", "1", "2", "4", "≥8"])
            elif overview == "fp":
                vl = self._values(elig["country"], elig["fp"], [hov(r) for _, r in elig.iterrows()],
                                  _SEQ_ATTR, 0, None, "False<br>positives")
            else:
                vl = self._values(elig["country"], elig["precision"], [hov(r) for _, r in elig.iterrows()],
                                  _SEQ_ATTR, 0, 1, "Precision", tickformat=".0%")
            layers = [self._grey(small["country"], [hov(r) + f"<br><i>n < {min_n}</i>" for _, r in small.iterrows()],
                                 _GREY_SMALL),
                      self._grey(phantom["country"], [hov(r) + "<br><i>No test images: never correct by design</i>"
                                                      for _, r in phantom.iterrows()], _PHANTOM),
                      vl]
            main = self._map(layers, f"Attractor strength – {self.ATTR_OVERVIEW[overview]}", selected=sel,
                             dots=dots, note="Ratio > 1 = predicted more often than it occurs · purple = predicted "
                                             "but no test images · click a country to see its FP sources")
            top = elig.sort_values("prediction_ratio", ascending=False).head(15)
            side1 = self._hbar(top["name"], top["prediction_ratio"], top["country"],
                               f"Strongest attractors (prediction ratio, n ≥ {min_n})", "Prediction ratio (log)",
                               "#3182bd", sel, log=True, vline=1, hover=[hov(r) for _, r in top.iterrows()])
            side2 = None
            if len(phantom):
                ph = phantom.sort_values("n_predicted", ascending=False).head(10)
                side2 = self._hbar(ph["name"], ph["n_predicted"], ph["country"],
                                   "Predicted but no test images", "Predictions (all FP)", _PHANTOM, sel,
                                   hover=[hov(r) for _, r in ph.iterrows()])
            table = self.a.attractors(min_n).merge(cs[["country", "name", "fp_top_source_name"]], on="country")
            table = table[["name", "country", "continent", "n", "n_predicted", "tp", "fp", "precision",
                           "prediction_ratio", "fp_distinct_origins", "fp_top_source_name",
                           "fp_top_source_share", "fp_source_entropy_bits"]]
            lines = []
            if len(elig):
                s = elig.loc[elig["prediction_ratio"].idxmax()]
                lines.append(f"**Strongest attractor:** {s['name']} – predicted {_num(s['prediction_ratio'])}× as "
                             f"often as it occurs (precision {_pct(s['precision'])}, {int(s['fp']):,} FP from "
                             f"{int(s['fp_distinct_origins'])} countries; top source {s['fp_top_source_name']} "
                             f"{_pct(s['fp_top_source_share'])}).")
                f = elig.loc[elig["fp"].idxmax()]
                lines.append(f"**Most false positives:** {f['name']} – {int(f['fp']):,}.")
            if len(phantom):
                lines.append(f"**{len(phantom)} {'country is' if len(phantom) == 1 else 'countries are'} predicted "
                         f"but {'has' if len(phantom) == 1 else 'have'} no test images** "
                             f"({int(phantom['n_predicted'].sum()):,} predictions, all wrong by design).")
            lines.append("Top-1 predictions only.")
            return main, side1, side2, table, (pct_cols, dec_cols), "  \n".join(lines)

        mlabel, is_pct = self.ATTR_SELECTED[selm]
        src = P[P["pred_country"] == sel]
        src = self._filter_cont(src, conts, "true_continent").sort_values(selm, ascending=False)
        r0 = self.cs_idx.loc[sel]
        text = [f"<b>{r.true_name} → {self.label(sel)}</b><br>{int(r.count):,} images<br>"
                f"{_pct(r.p_true_given_pred_error)} of {self.label(sel)}'s FPs<br>"
                f"{_pct(r.p_pred_given_true)} of {r.true_name}'s images" for r in src.itertuples()]
        layers = [self._values(src["true_country"], src[selm], text, _SEQ_ATTR, 0, None,
                               mlabel.split(" (")[0].replace(" ", "<br>", 1),
                               tickformat=".0%" if is_pct else None)]
        fill_txt = (f"<b>{self.label(sel)}</b> (predicted)<br>predicted {int(r0['n_predicted']):,}× · "
                    f"FP {int(r0['fp']):,} · precision {_pct(r0['precision'])}")
        main = self._map(layers, f"Where the false positives of {self.label(sel)} come from – {mlabel}",
                         selected=sel, fill=(sel, _TARGET, fill_txt), dots=dots,
                         note="Orange = selected predicted country" +
                              (" · origins filtered by continent" if conts else ""))
        top = src.head(12)
        side1 = self._hbar(top["true_name"], top[selm], top["true_country"],
                           f"Top FP sources of {self.label(sel)} ({mlabel})", mlabel, "#3182bd", pct=is_pct)
        by_c = src.groupby("true_continent")["count"].sum().sort_values(ascending=False)
        side2 = self._hbar([self._cont_name(c) for c in by_c.index], by_c / max(by_c.sum(), 1), None,
                           "FP sources by continent (% of shown FPs)", "% of FPs", "#9ecae1", pct=True)
        tcols = ["true_name", "true_country", "true_continent", "count", "p_true_given_pred_error",
                 "p_pred_given_true", "n_true", "reverse_count"]
        lines = [f"**{self.label(sel)}:** predicted {int(r0['n_predicted']):,}× for {int(r0['n']):,} true images "
                 f"(ratio {_num(r0['prediction_ratio'])}), precision {_pct(r0['precision'])}, "
                 f"{int(r0['fp']):,} FP from {int(r0['fp_distinct_origins'])} countries."]
        if r0["n"] == 0:
            lines.append("No test images: every prediction is wrong by design.")
        elif r0["n"] < min_n:
            lines.append(f"⚠ Only {int(r0['n'])} true images (< {min_n}): ratio and precision are unreliable.")
        if len(src):
            t3 = src.sort_values("count", ascending=False).head(3)
            lines.append("**Top sources:** " + ", ".join(
                f"{a} {int(c):,} ({_pct(s)})" for a, c, s in
                zip(t3["true_name"], t3["count"], t3["p_true_given_pred_error"])) + ".")
        return main, side1, side2, src[tcols], (pct_cols, dec_cols), "  \n".join(lines)

    # ------------------------------------------------------------------ 4 shared characteristics
    def view_shared(self, scope, weighting):
        within = scope == "within"
        fb = self._baseline(within, weighting)
        name = f"{'within-continent' if within else 'global'} baseline, {weighting}-weighted"
        if fb is None or fb.empty:
            e = self._empty("No errors available for this baseline")
            return e, e, None, pd.DataFrame(), ([], []), "No data."
        fb = fb.copy()
        fb["label"] = fb["feature"].map(lambda f: _FEATURE_LABELS.get(f, f.replace("_", " ")))
        fb["significant"] = fb["p_value"] < 0.05
        go = self.go
        hover = [f"<b>{r.label}</b><br>Observed: {_pct(r.observed) if r.kind == 'share' else _num(r.observed, 1)}"
                 f"<br>Random pairs: {_pct(r.baseline) if r.kind == 'share' else _num(r.baseline, 1)}"
                 f"<br>Lift {_num(r.lift)} ({_num(r.lift_ci_low)} – {_num(r.lift_ci_high)})"
                 f"<br>p = {r.p_value:.3f} · n known = {int(r.n_known):,}" for r in fb.itertuples()]
        main = go.Figure(go.Scatter(
            x=fb["lift"], y=fb["label"], mode="markers", text=hover, hovertemplate="%{text}<extra></extra>",
            error_x=dict(type="data", symmetric=False, array=fb["lift_ci_high"] - fb["lift"],
                         arrayminus=fb["lift"] - fb["lift_ci_low"], thickness=1.5, width=4),
            marker=dict(size=12, color=np.where(fb["significant"], "#d62728", "#9e9e9e"),
                        symbol=np.where(fb["significant"], "circle", "circle-open"), line=dict(width=2))))
        self._ref_line(main, 1, list(fb["label"]))
        main.update_xaxes(type="log", title="Lift = observed / random pairs (log scale)")
        main.update_yaxes(autorange="reversed", automargin=True)
        main.update_layout(title=dict(text=f"Do confused pairs share these characteristics? ({name})",
                                      x=0.01, font=dict(size=15)),
                           height=470, plot_bgcolor="white", margin=dict(l=10, r=10, t=45, b=60))
        main.add_annotation(text="Filled red = p < 0.05 · > 1: shared more often than by random pairs · "
                                 "for distances < 1 = closer than random",
                            showarrow=False, x=0, y=-0.16, xref="paper", yref="paper", xanchor="left",
                            font=dict(size=11, color="#555"))
        sh = fb[fb["kind"] == "share"].rename(columns={"observed": "observed_share", "baseline": "baseline_share"})
        side1 = self._obs_vs_base(sh, "Share of pairs with the characteristic", label_col="label")
        dist = fb[fb["kind"] != "share"]
        side2 = None
        if len(dist):
            side2 = self.make_subplots(rows=1, cols=len(dist), subplot_titles=list(dist["label"]))
            for i, r in enumerate(dist.itertuples(), start=1):
                side2.add_trace(go.Bar(x=["Errors", "Random"], y=[r.observed, r.baseline],
                                       marker_color=["#d62728", "#9e9e9e"], showlegend=False), row=1, col=i)
            side2.update_layout(height=280, plot_bgcolor="white", margin=dict(l=10, r=10, t=50, b=30),
                                title=dict(text="Mean distance: errors vs random pairs", x=0.01, font=dict(size=13)))
        table = fb[["label", "kind", "n_errors", "n_known", "observed", "observed_ci_low", "observed_ci_high",
                    "baseline", "lift", "lift_ci_low", "lift_ci_high", "p_value"]]
        sig = fb[fb["significant"]].sort_values("lift", ascending=False)
        lines = [f"**{name.capitalize()}**, {int(fb['n_errors'].iloc[0]):,} "
                 f"{'errors' if weighting == 'error' else 'distinct pairs'}"
                 + (" (only errors within the same continent)." if within else ".")]
        for r in sig.itertuples():
            if r.kind == "share":
                lines.append(f"**{r.label}:** {_pct(r.observed)} vs {_pct(r.baseline)} for random pairs "
                             f"(lift {_num(r.lift)}, p = {r.p_value:.3f}).")
            else:
                lines.append(f"**{r.label}:** {_num(r.observed, 1)} vs {_num(r.baseline, 1)} for random pairs "
                             f"(p = {r.p_value:.3f}).")
        if not len(sig):
            lines.append("No characteristic differs significantly from random pairs.")
        lines.append("Shared ≠ cause: these are associations. Compare global vs within-continent to separate "
                     "continent effects from the characteristic itself.")
        return main, side1, side2, table, (["observed", "observed_ci_low", "observed_ci_high", "baseline"],
                                           ["lift", "lift_ci_low", "lift_ci_high"]), "  \n".join(lines)

    # ------------------------------------------------------------------ 5 geography
    def view_geography(self, sel, k, min_n, weighting, auto, dots):
        rate = f"error_rate_top{k}"
        rc = self.a.region_comparison("subregion", k, eligible_only=True, min_n=min_n)
        cs = self.cs[self.cs["in_test"]]
        has = cs[cs["subregion"].isin(rc.index)]
        rest = cs[~cs["subregion"].isin(rc.index)]
        text = [f"<b>{r['name']}</b><br>Subregion: {r['subregion']}<br>"
                f"Subregion macro error: {_pct(rc.at[r['subregion'], 'macro_error_rate'])} "
                f"({int(rc.at[r['subregion'], 'n_countries'])} countries)<br>"
                f"Own top-{k} error: {_pct(r[rate])} (n = {int(r['n']):,})" +
                ("" if r["n"] >= min_n else f"<br><i>n < {min_n}: not in the average</i>")
                for _, r in has.iterrows()]
        layers = [self._nodata_layer(None),
                  self._grey(rest["country"], [f"<b>{n}</b><br>No subregion average (too few countries)"
                                               for n in rest["name"]], _GREY_SMALL),
                  self._values(has["country"], has["subregion"].map(rc["macro_error_rate"]), text, _SEQ_ERR,
                               None if auto else 0, None if auto else 1, f"Subregion<br>top-{k} error",
                               tickformat=".0%")]
        main = self._map(layers, f"Top-{k} error rate per subregion (mean of countries with n ≥ {min_n})",
                         selected=sel, dots=dots)
        loc = self._locality(weighting)
        side1 = self._obs_vs_base(loc, "Where do wrong predictions land?")
        hops = self._hops(weighting)
        side2 = self._obs_vs_base(hops, "Border hops between true and predicted country") if hops is not None else None
        table = rc.reset_index().rename(columns={"index": "subregion"})
        lines = []
        mi = self._moran(k, min_n)
        if mi and not _is_missing(mi.get("morans_i")):
            clustered = mi["p_value"] < 0.05 and mi["morans_i"] > mi["expected_i"]
            lines.append(f"**Moran's I (top-{k} error, neighbour graph):** {mi['morans_i']:.2f} "
                         f"(random ≈ {mi['expected_i']:.2f}, p = {mi['p_value']:.3f}, {mi['n_countries']} countries). "
                         + ("Hard countries tend to border hard countries." if clustered
                            else "No evidence that hard countries cluster."))
        elif mi is not None:
            lines.append("Moran's I: too few connected countries.")
        if len(loc):
            r = loc.iloc[0]
            lines.append(f"**{r['category'].capitalize()}:** {_pct(r['observed_share'])} of errors vs "
                         f"{_pct(r['baseline_share'])} for random pairs ({_num(r['ratio'])}×).")
        if hops is not None and len(hops):
            h = hops.iloc[0]
            lines.append(f"**Direct neighbour predicted:** {_pct(h['observed_share'])} of errors vs "
                         f"{_pct(h['baseline_share'])} for random pairs.")
        lines.append(f"Locality and hops use top-1 errors ({weighting}-weighted). Islands count as 'no land path'.")
        return main, side1, side2, table, (["macro_error_rate", "macro_std", "micro_error_rate",
                                            "min_error_rate", "max_error_rate"], []), "  \n".join(lines)

    # ------------------------------------------------------------------ dispatcher
    def render(self, mode="difficulty", selected=None, k=1, continents=None, min_n=None,
               conf_metric="count", attr_overview="ratio", attr_selected="count",
               scope="global", weighting="error", options=()):
        min_n = self.cfg.min_n if min_n in (None, "") else max(1, int(min_n))
        conts = list(continents or [])
        auto, dots = "auto" in (options or []), "dots" in (options or [])
        if mode == "difficulty":
            return self.view_difficulty(selected, k, conts, min_n, auto, dots)
        if mode == "confusion":
            return self.view_confusion(selected, conts, min_n, conf_metric, dots)
        if mode == "attractors":
            return self.view_attractors(selected, conts, min_n, attr_overview, attr_selected, dots)
        if mode == "shared":
            return self.view_shared(scope, weighting)
        return self.view_geography(selected, k, min_n, weighting, auto, dots)

    # ================================================================== Dash app
    def _code_from_click(self, click):
        if not click or not click.get("points"):
            return None
        pt = click["points"][0]
        cd = pt.get("customdata")
        if isinstance(cd, list):
            cd = cd[0] if cd else None
        if isinstance(cd, str) and cd in self.name:
            return cd
        loc = pt.get("location")
        return self.from_iso3.get(loc)

    def build_app(self):
        from dash import Dash, Input, Output, State, ctx, dash_table, dcc, html, no_update

        cfg = self.cfg
        countries = self.cs[(self.cs["n"] > 0) | (self.cs["n_predicted"] > 0)]
        country_opts = [{"label": f"{n} ({c})", "value": c}
                        for n, c in sorted(zip(countries["name"], countries["country"]))]
        cont_codes = sorted({c for c in self.cs.loc[self.cs["in_test"], "continent"] if not _is_missing(c)},
                            key=self._cont_name)
        cont_opts = [{"label": self._cont_name(c), "value": c} for c in cont_codes]
        graph_cfg = {"displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 3}}
        box = {"display": "inline-block", "marginRight": "22px", "verticalAlign": "top"}
        lab = {"fontSize": "12px", "color": "#555", "fontWeight": "600", "marginBottom": "2px"}

        base_style = {}

        def ctrl(cid, label, comp, width=None, label_id=None):
            style = dict(box, **({"width": width} if width else {}))
            base_style[cid] = style
            return html.Div([html.Div(label, id=label_id, style=lab) if label_id else html.Div(label, style=lab),
                             comp], id=cid, style=style)

        controls = [
            ctrl("c-k", "Top-k", dcc.RadioItems(id="k", value=1, inline=True,
                                                options=[{"label": f" Top-{x} ", "value": x} for x in self.d.ks])),
            ctrl("c-cont", "Continent", dcc.Dropdown(id="continents", options=cont_opts, value=[], multi=True,
                                                     placeholder="All continents"), "250px", "cont-label"),
            ctrl("c-minn", "Min. images per country", dcc.Input(id="min-n", type="number", min=1, step=1,
                                                               value=cfg.min_n, debounce=True,
                                                               style={"width": "80px"})),
            ctrl("c-country", "Country", dcc.Dropdown(id="country", options=country_opts, value=None,
                                                      placeholder="Search a country…", clearable=True),
                 "240px", "country-label"),
            ctrl("c-conf", "Colour / rank by", dcc.RadioItems(
                id="conf-metric", value="count", inline=True,
                options=[{"label": f" {v[0]} ", "value": k} for k, v in self.CONF_METRICS.items()])),
            ctrl("c-attr-o", "Map colour (overview)", dcc.RadioItems(
                id="attr-over", value="ratio", inline=True,
                options=[{"label": f" {v} ", "value": k} for k, v in self.ATTR_OVERVIEW.items()])),
            ctrl("c-attr-s", "Map colour (selected country)", dcc.RadioItems(
                id="attr-sel", value="count", inline=True,
                options=[{"label": f" {v[0].split(' (')[0]} ", "value": k} for k, v in self.ATTR_SELECTED.items()])),
            ctrl("c-scope", "Baseline", dcc.RadioItems(
                id="scope", value="global", inline=True,
                options=[{"label": " Global ", "value": "global"},
                         {"label": " Within continent ", "value": "within"}])),
            ctrl("c-weight", "Weighting", dcc.RadioItems(
                id="weight", value="error", inline=True,
                options=[{"label": " Every error ", "value": "error"},
                         {"label": " Every distinct pair ", "value": "pair"}])),
            ctrl("c-opts", "Display", dcc.Checklist(
                id="options", value=[], inline=True,
                options=[{"label": " Auto colour range ", "value": "auto"},
                         {"label": " Dots for small countries ", "value": "dots"}])),
        ]
        ctrl_ids = ["c-k", "c-cont", "c-minn", "c-country", "c-conf", "c-attr-o", "c-attr-s",
                    "c-scope", "c-weight", "c-opts"]
        visible = {"difficulty": {"c-k", "c-cont", "c-minn", "c-country", "c-opts"},
                   "confusion": {"c-cont", "c-minn", "c-country", "c-conf", "c-opts"},
                   "attractors": {"c-cont", "c-minn", "c-country", "c-attr-o", "c-attr-s", "c-opts"},
                   "shared": {"c-scope", "c-weight"},
                   "geography": {"c-k", "c-minn", "c-weight", "c-opts"}}
        help_text = {
            "difficulty": "How often is each country wrong? Hover for n and CI, click a country to select it.",
            "confusion": "Where do a country's images go when they are wrong? Select an origin country.",
            "attractors": "Which countries does the model over-predict, and where do their false positives come from?",
            "shared": "Do confused pairs share characteristics more often than random country pairs?",
            "geography": "Do errors stay nearby, and do hard countries border each other?",
        }

        app = Dash(__name__)
        app.title = "Q1 – Error structure and geography"
        app.layout = html.Div(style={"fontFamily": "system-ui, sans-serif", "padding": "8px 14px",
                                     "background": "white"}, children=[
            dcc.Store(id="selected"),
            dcc.Tabs(id="mode", value="difficulty",
                     children=[dcc.Tab(label=lab_, value=val) for val, lab_ in self.MODES]),
            html.Div(id="help", style={"margin": "8px 0", "color": "#444", "fontSize": "14px"}),
            html.Div(controls, style={"padding": "6px 0"}),
            html.Div([html.Span(id="chip", style={"marginRight": "10px"}),
                      html.Button("Clear selection", id="clear", n_clicks=0,
                                  style={"fontSize": "12px"})],
                     style={"margin": "4px 0 6px 0", "fontSize": "14px"}),
            dcc.Markdown(id="finding", style={"background": "#f6f6f6", "padding": "8px 12px",
                                              "borderLeft": "4px solid #444", "fontSize": "14px"}),
            html.Div(style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}, children=[
                html.Div(dcc.Graph(id="main", config=graph_cfg), style={"flex": "3 1 620px", "minWidth": "0"}),
                html.Div([dcc.Graph(id="side1", config=graph_cfg), dcc.Graph(id="side2", config=graph_cfg)],
                         style={"flex": "2 1 380px", "minWidth": "0"}),
            ]),
            html.H4("Details", style={"margin": "10px 0 4px 0"}),
            dash_table.DataTable(id="table", sort_action="native", filter_action="native", page_size=15,
                                 export_format="csv", style_table={"overflowX": "auto"},
                                 style_cell={"fontSize": "12px", "padding": "4px",
                                             "fontFamily": "system-ui, sans-serif", "maxWidth": "220px",
                                             "overflow": "hidden", "textOverflow": "ellipsis"},
                                 style_header={"fontWeight": "600", "background": "#f0f0f0"}),
        ])

        @app.callback([Output(i, "style") for i in ctrl_ids] +
                      [Output("country-label", "children"), Output("cont-label", "children"),
                       Output("help", "children")],
                      Input("mode", "value"))
        def show_controls(mode):
            styles = [base_style[i] if i in visible[mode] else {"display": "none"} for i in ctrl_ids]
            country_lab = "Predicted country" if mode == "attractors" else \
                          "True country (origin)" if mode == "confusion" else "Country"
            cont_lab = "Origin (true-country) continent" if mode in ("confusion", "attractors") else "Continent"
            return styles + [country_lab, cont_lab, help_text[mode]]

        @app.callback(Output("selected", "data"), Output("country", "value"),
                      Input("main", "clickData"), Input("side1", "clickData"), Input("side2", "clickData"),
                      Input("country", "value"), Input("clear", "n_clicks"),
                      prevent_initial_call=True)
        def select(main_click, side1_click, side2_click, dd, _clear):
            trig = ctx.triggered_id
            if trig == "clear":
                return None, None
            if trig == "country":
                return dd, no_update
            click = {"main": main_click, "side1": side1_click, "side2": side2_click}.get(trig)
            code = self._code_from_click(click)
            if code is None:
                return no_update, no_update
            return code, code

        @app.callback(Output("main", "figure"), Output("side1", "figure"), Output("side2", "figure"),
                      Output("side2", "style"), Output("table", "data"), Output("table", "columns"),
                      Output("finding", "children"), Output("chip", "children"),
                      Input("mode", "value"), Input("selected", "data"), Input("k", "value"),
                      Input("continents", "value"), Input("min-n", "value"), Input("conf-metric", "value"),
                      Input("attr-over", "value"), Input("attr-sel", "value"), Input("scope", "value"),
                      Input("weight", "value"), Input("options", "value"))
        def update(mode, sel, k, conts, min_n, conf_m, attr_o, attr_s, scope, weight, opts):
            try:
                main, s1, s2, table, (pct, dec), finding = self.render(
                    mode, sel, k, conts, min_n, conf_m, attr_o, attr_s, scope, weight, opts)
            except Exception as exc:   # show the error in the dashboard instead of failing silently
                e = self._empty(f"Error: {exc}")
                return e, e, e, {"display": "none"}, [], [], f"**Error:** `{exc!r}`", ""
            data, cols = self._table(table, pct, dec) if len(table) else ([], [])
            chip = (f"Selected: **{self.label(sel)}** ({sel})" if sel else
                    "No country selected – click the map / a bar, or search above.")
            return (main, s1, s2 if s2 is not None else self._empty(""),
                    {} if s2 is not None else {"display": "none"},
                    data, cols, finding, dcc.Markdown(chip, style={"display": "inline-block"}))

        return app

    def run(self, mode: str | None = "inline", port: int = 8050, height: int = 1250,
            host: str = "127.0.0.1", proxy: bool = False, debug: bool = False):
        """mode: 'inline' (under the cell), 'tab' (new browser tab), 'external' (prints a URL),
        None (plain server outside Jupyter). proxy=True: route through the Jupyter server
        (use if the inline view stays blank on a remote JupyterLab)."""
        app = self.build_app()
        if proxy:
            from dash import jupyter_dash
            jupyter_dash.infer_jupyter_proxy_config()
        if mode is None:
            app.run(host=host, port=port, debug=debug)
        else:
            app.run(jupyter_mode=mode, jupyter_height=height, host=host, port=port, debug=debug)
        return app