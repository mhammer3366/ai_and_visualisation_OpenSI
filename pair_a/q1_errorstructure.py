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

    # ------------------------------------------------------------------ tables
    @staticmethod
    def show(obj, title: str | None = None, n: int | None = 15):
        if title:
            print(f"\n=== {title} ===")
        if isinstance(obj, pd.DataFrame) and n:
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
        ax.barh([f"{c} (n={n})" for c, n in zip(df["country"], df["n"])], rate, xerr=xerr, capsize=2)
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
        ax.barh(df["country"], df["prediction_ratio"])
        ax.axvline(1, color="k", lw=1, ls="--")
        ax.set_xlabel("Prediction ratio (# predicted / # true)")
        ax.set_title("Attractor countries")
        fig.tight_layout()
        return fig
