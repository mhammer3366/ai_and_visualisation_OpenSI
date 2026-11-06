"""Q3: Visual clues (pair_a, AI-aided, Phase I).

Which image elements (road, lane markings, vegetation, signs, text, ...) change
StreetCLIP's prediction when they are hidden, and does this differ between countries?
Black box only: we change the input image and compare the outputs.

Steps (every step saves its result, so nothing is computed twice)
    Q3Config       all Q3 settings: sample size, element groups, combinations, thresholds
    Q3Sampler      stratified sample: every candidate country, at most `cap` images per
                   country, same right/wrong share as on the full test set, spread over
                   as many camera sequences as possible
    Q3MaskBuilder  segmentation + text detection once per image (cached in data/masks/),
                   then a presence table: which elements each image shows, and how much
    Q3Runner       StreetCLIP on the original, on every hidden element / combination and
                   on random same-size masks (controls); logits saved in resumable chunks
    Q3Analyser     effect = real mask minus random control; flips; neighbour confusion;
                   do two cues compensate each other; country profiles; environment
    Q3Reporter     readable tables, plots and example images

Files (both folders are already in .gitignore)
    data/masks/<id>_seg.npz, <id>_text.npz          label maps / text masks (shared cache)
    results/pair_a/q3/sample.csv                    the sample
    results/pair_a/q3/presence.parquet              elements per image
    results/pair_a/q3/runs/<name>/chunk_*.parquet   one row per (image, version)
    results/pair_a/q3/runs/<name>/chunk_*.npy       logits of these rows (float32)
    results/pair_a/q3/runs/<name>/settings.json     settings and candidate labels of the run

Masked images are NOT saved: they are rebuilt from the cached masks in milliseconds
(Q3Runner.variant_images), and saving ~500k JPEGs would take ~20 GB and change pixels.
"""
from __future__ import annotations

import json
import os
import time
import zlib
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

# Element -> Mapillary Vistas classes. Elements may overlap: "road" is the old shared
# element, "road_surface" and "lane_markings" are its two halves. Names are matched
# ignoring case, spaces and punctuation (same rule as shared.models.Segmenter).
ELEMENT_CLASSES = {
    "road": ["Road", "Service Lane", "Bike Lane", "Parking", "Crosswalk - Plain", "Manhole",
             "Pothole", "Lane Marking - General", "Lane Marking - Crosswalk"],
    "road_surface": ["Road", "Service Lane", "Bike Lane", "Parking", "Crosswalk - Plain",
                     "Manhole", "Pothole"],
    "lane_markings": ["Lane Marking - General", "Lane Marking - Crosswalk"],
    "vegetation_terrain": ["Vegetation", "Terrain", "Sand", "Snow"],
    "vegetation": ["Vegetation"],
    "terrain": ["Terrain", "Sand", "Snow"],
    "sky": ["Sky"],
    "buildings": ["Building", "Wall", "Bridge", "Tunnel"],
    "vehicles": ["Car", "Bus", "Truck", "Motorcycle", "Bicycle", "Caravan", "Trailer",
                 "Other Vehicle", "Wheeled Slow", "On Rails", "Boat"],
    "signs_poles": ["Traffic Sign (Front)", "Traffic Sign (Back)", "Billboard", "Banner",
                    "Pole", "Utility Pole", "Street Light", "Traffic Light", "Traffic Sign Frame"],
    "signs": ["Traffic Sign (Front)", "Traffic Sign (Back)", "Billboard", "Banner"],
    "poles": ["Pole", "Utility Pole", "Street Light", "Traffic Light", "Traffic Sign Frame"],
    "ego_vehicle": ["Ego Vehicle", "Car Mount"],   # dashcam car hood / mount (uploader-specific)
}
TEXT = "text"   # from EasyOCR, not from the segmentation

# Combinations hidden together. Only run on images where every part is present.
COMBOS = {
    "vegetation+road": ("vegetation", "road"),
    "text+signs": ("text", "signs"),
}

# whole -> parts, for "do the cues compensate each other?"
# The whole is an element or combination whose mask is the union of the parts.
INTERACTIONS = {
    "road": ("road_surface", "lane_markings"),
    "vegetation_terrain": ("vegetation", "terrain"),
    "signs_poles": ("signs", "poles"),
    "vegetation+road": ("vegetation", "road"),
    "text+signs": ("text", "signs"),
}

# Minimum share of the model's visible input (after center crop and dilation) that an
# element must cover to be hidden. Combinations: smallest threshold of their parts.
MIN_MASK_AREA = {
    "default": 0.02,
    "lane_markings": 0.002,
    "signs": 0.001, "poles": 0.002, "signs_poles": 0.001,
    "text": 0.001,
}

# Effect measures (per image, masked minus original). "x" = real mask minus control.
METRICS = ("d_true", "d_orig", "d_conf", "d_entropy", "flip", "r2w", "w2r", "d_nb", "d_sim")
METRIC_NAMES = {
    "d_true": "Change in P(true country)",
    "d_orig": "Change in P(original guess)",
    "d_conf": "Change in confidence (top-1 prob.)",
    "d_entropy": "Change in entropy (bits)",
    "flip": "Top-1 changed",
    "r2w": "Right -> wrong (of right)",
    "w2r": "Wrong -> right (of wrong)",
    "d_nb": "Change in P(neighbours of true)",
    "d_sim": "Change in P(usual confusions of true)",
}
RATES = ("flip", "r2w", "w2r")               # shown in %
POINTS = ("d_true", "d_orig", "d_conf", "d_nb", "d_sim")   # shown in percentage points
SIGN = {"d_true": -1, "d_orig": -1, "d_conf": -1}   # interactions: "effect" > 0 = hurts more

# Condition groups for the plots (they do not change what is run)
# coarse: one row per visual element (Q3.1); fine: the parts, no overlaps (Q3.2 onwards)
COARSE = ("road", "vegetation_terrain", "sky", "buildings", "vehicles", "signs_poles", "ego_vehicle", "text")
FINE = ("road_surface", "lane_markings", "vegetation", "terrain", "sky", "buildings", "vehicles",
        "signs", "poles", "ego_vehicle", "text")
# Q3.2: whole element -> its parts ("other" is only a heading, not a condition)
SPLITS = {"road": ("road_surface", "lane_markings"), "vegetation_terrain": ("vegetation", "terrain"),
          "signs_poles": ("signs", "poles"), "other": ("text", "ego_vehicle")}


@dataclass
class Q3Config:
    """All Q3 settings. Change them in the notebook, e.g. Q3Config(cap=60, n_controls=1)."""

    # --- sample ---
    cap: int = 60                      # images per country (smaller countries keep all)
    focus_countries: tuple = ()        # countries that get focus_cap images instead
    focus_cap: int = 150
    max_per_sequence: int = 2          # soft limit: more only if a country has too few sequences
    seed: int = 42

    # --- elements and masking ---
    element_classes: dict = field(default_factory=lambda: dict(ELEMENT_CLASSES))
    combos: dict = field(default_factory=lambda: dict(COMBOS))
    interactions: dict = field(default_factory=lambda: dict(INTERACTIONS))
    min_mask_area: dict = field(default_factory=lambda: dict(MIN_MASK_AREA))
    n_controls: int = 3                # random same-size masks per real mask

    # --- running ---
    gpu_batch: int = 64                # StreetCLIP batch size for the masked versions
    chunk_size: int = 1000             # images per saved chunk (a crash loses at most one chunk)

    # --- analysis ---
    min_n: int = 20                    # images per (country, element) cell for country profiles
    min_country_images: int = 5        # images a country needs to count in a macro average
    n_similar: int = 5                 # "usual confusions" = top-n wrong guesses on the full test set
    n_boot: int = 1000                 # bootstrap draws (camera sequences are resampled)

    out_dir: str = "results/pair_a/q3"

    # --- derived ---
    @property
    def elements(self):
        """Segmentation elements + text."""
        return list(self.element_classes) + [TEXT]

    @property
    def conditions(self):
        """Everything that gets hidden: elements, text, then combinations."""
        return self.elements + list(self.combos)

    def parts(self, condition):
        return tuple(self.combos.get(condition, (condition,)))

    def mask_names(self, condition):
        """Element masks whose union is hidden for this condition."""
        if condition in self.combos:
            return sorted({n for p in self.combos[condition] for n in self.mask_names(p)})
        return [condition]

    def threshold(self, condition):
        if condition in self.min_mask_area:
            return self.min_mask_area[condition]
        if condition in self.combos:
            return min(self.threshold(p) for p in self.combos[condition])
        return self.min_mask_area["default"]

    def shared_cfg(self, cfg):
        """Copy of the shared Config with our element groups and thresholds (shared/ is not changed)."""
        thresholds = {c: self.threshold(c) for c in self.conditions}
        thresholds["default"] = self.min_mask_area["default"]
        return replace(cfg, element_classes=dict(self.element_classes), min_mask_area=thresholds,
                       num_random_controls=max(self.n_controls, 1), seed=self.seed)

    def check(self):
        """Catch setting mistakes early. Returns a list of warnings (errors are raised)."""
        warnings = []
        elements = set(self.elements)
        for combo, parts in self.combos.items():
            if combo in elements:
                raise ValueError(f"combination {combo!r} has the same name as an element")
            bad = [p for p in parts if p not in elements]
            if bad:
                raise ValueError(f"combination {combo!r}: parts must be elements, not {bad}")
        conditions = set(self.conditions)
        for whole, parts in self.interactions.items():
            bad = [c for c in (whole, *parts) if c not in conditions]
            if bad:
                raise ValueError(f"interaction {whole!r}: unknown conditions {bad}")
            if whole in self.element_classes and all(p in self.element_classes for p in parts):
                w = {_simplify(n) for n in self.element_classes[whole]}
                u = {_simplify(n) for p in parts for n in self.element_classes[p]}
                if w != u:
                    warnings.append(f"interaction {whole!r}: its classes are not exactly the union of {parts}")
        if self.n_controls < 1:
            warnings.append("n_controls = 0: no random controls, effects cannot be corrected for mask size")
        return warnings


def _simplify(name):
    """'Traffic Sign (Front)' -> 'trafficsignfront' (same rule as shared.models)."""
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _seed(seed, image_id, condition):
    """Stable seed per (image, condition): random controls differ between images but are reproducible."""
    return (seed * 1_000_003 + zlib.crc32(f"{image_id}|{condition}".encode())) % (2 ** 32)


# ---------------------------------------------------------------------------
# 1. Sample
# ---------------------------------------------------------------------------

class Q3Sampler:
    """
    Stratified sample from the baseline prediction table (results/shared/...).

        sampler = Q3Sampler(q3cfg, preds)
        sampler.estimate([40, 50, 60, 80])     # images for different caps
        sample = sampler.build()               # loads sample.csv if it exists, else draws + saves

    Rules per candidate country (non-candidates are left out: right/wrong is undefined):
      - n = min(images of the country, cap)  (focus countries: focus_cap)
      - right / wrong images in the same share as on the full test set
        (a country with 80 % accuracy gets 80 % right images)
      - camera sequences: first one image per sequence, then a second, ... so the sample
        spreads over as many drives as possible; more than max_per_sequence only if needed
    """

    def __init__(self, q3cfg, preds):
        self.q3cfg = q3cfg
        p = preds[preds["representable"].astype(bool) & preds["correct"].notna()]
        p = p[["id", "country_id", "correct", "sequence"]].copy()
        p["id"] = p["id"].astype(str)
        p["correct"] = p["correct"].astype(bool)
        p["sequence"] = p["sequence"].astype("string").fillna("img_" + p["id"]).astype(str)
        self.preds = p
        self.path = Path(q3cfg.out_dir) / "sample.csv"

    def _cap(self, country):
        return self.q3cfg.focus_cap if country in self.q3cfg.focus_countries else self.q3cfg.cap

    def estimate(self, caps=(30, 40, 50, 60, 80, 100, 150)):
        """How many images (and which share of the test set) each cap gives."""
        sizes = self.preds["country_id"].value_counts()
        rows = [{"cap": c, "images": int(np.minimum(sizes, c).sum()),
                 "countries_capped": int((sizes > c).sum())} for c in caps]
        t = pd.DataFrame(rows).set_index("cap")
        t["share_of_test_set"] = t["images"] / len(self.preds)
        return t

    def draw(self):
        cfg = self.q3cfg
        p = self.preds.sample(frac=1, random_state=cfg.seed)   # random order, fixed seed
        p["seq_rank"] = p.groupby(["country_id", "sequence"]).cumcount()
        acc = p.groupby("country_id")["correct"].mean()

        picked = []
        for country, g in p.groupby("country_id", sort=True):
            target = min(len(g), self._cap(country))
            if target == len(g):
                picked.append(g)
                continue
            k_right = int(round(target * acc[country]))
            for is_right, k in ((True, k_right), (False, target - k_right)):
                stratum = g[g["correct"] == is_right]
                # stable sort keeps the random order inside each seq_rank
                picked.append(stratum.sort_values("seq_rank", kind="stable").head(k))
        s = pd.concat(picked)
        s["focus"] = s["country_id"].isin(cfg.focus_countries)
        s["over_seq_limit"] = s["seq_rank"] >= cfg.max_per_sequence
        return s.sort_values(["country_id", "id"]).reset_index(drop=True)

    def build(self, overwrite=False):
        """Load the saved sample, or draw it and save it to results/pair_a/q3/sample.csv."""
        if self.path.exists() and not overwrite:
            s = pd.read_csv(self.path, dtype={"id": str, "sequence": str, "country_id": str},
                            keep_default_na=False)
            for col in ("correct", "focus", "over_seq_limit"):
                s[col] = s[col].astype(str).str.lower().eq("true")
            print(f"Sample loaded from {self.path} ({len(s):,} images)")
            return s
        s = self.draw()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        s.to_csv(self.path, index=False)
        print(f"New sample: {len(s):,} images from {s['country_id'].nunique()} countries "
              f"({len(s) / len(self.preds):.1%} of the usable test set), saved to {self.path}")
        return s

    def summary(self, sample):
        """Per country: full test set vs sample (n, accuracy, sequences). Check the stratification here."""
        full = self.preds.groupby("country_id").agg(n_full=("id", "size"), acc_full=("correct", "mean"),
                                                    seq_full=("sequence", "nunique"))
        smp = sample.groupby("country_id").agg(n_sample=("id", "size"), acc_sample=("correct", "mean"),
                                               seq_sample=("sequence", "nunique"),
                                               max_per_seq=("sequence", lambda s: s.value_counts().max()),
                                               over_seq_limit=("over_seq_limit", "sum"))
        t = full.join(smp, how="left")
        t["acc_diff"] = t["acc_sample"] - t["acc_full"]
        t["focus"] = t.index.isin(self.q3cfg.focus_countries)
        return t.sort_values("n_full", ascending=False)

    @staticmethod
    def confused_countries(preds, n=20, min_images=150):
        """Data-driven focus set: countries in the biggest confusion pairs (either side) that have
        at least min_images test images. Avoids hand-picking."""
        p = preds[preds["representable"].astype(bool) & preds["correct"].notna()]
        sizes = p["country_id"].value_counts()
        wrong = p[p["pred_1"] != p["country_id"]]
        pairs = wrong.groupby(["country_id", "pred_1"]).size().sort_values(ascending=False)
        chosen = []
        for (a, b) in pairs.index:
            for c in (a, b):
                if c not in chosen and sizes.get(c, 0) >= min_images:
                    chosen.append(c)
            if len(chosen) >= n:
                break
        return tuple(str(c) for c in chosen[:n])


# ---------------------------------------------------------------------------
# 2. Masks and presence
# ---------------------------------------------------------------------------

class ElementMasks:
    """
    Reads the cached label maps (data/masks/<id>_seg.npz) and text masks (<id>_text.npz)
    and turns them into one boolean mask per element. Needs no GPU and no model weights:
    only the class names of the segmentation model (config.json, already downloaded).
    """

    def __init__(self, q3cfg, cfg, id2label=None):
        self.cache_dir = Path(cfg.data_dir) / "masks"
        if id2label is None:
            from transformers import AutoConfig
            id2label = AutoConfig.from_pretrained(cfg.segmentation_model).id2label
        by_name = {_simplify(name): int(i) for i, name in id2label.items()}
        self.lut, self.unmatched = {}, []
        for element, names in q3cfg.element_classes.items():
            table = np.zeros(256, dtype=bool)   # label maps are uint8
            for name in names:
                if _simplify(name) in by_name:
                    table[by_name[_simplify(name)]] = True
                else:
                    self.unmatched.append(f"{element}: {name}")
            self.lut[element] = table

    def seg_path(self, image_id):
        return self.cache_dir / f"{image_id}_seg.npz"

    def text_path(self, image_id):
        return self.cache_dir / f"{image_id}_text.npz"

    def load(self, image_id):
        """{element: HxW bool mask} including 'text' (empty mask if text detection was not run)."""
        labels = np.load(self.seg_path(image_id))["labels"]
        masks = {element: table[labels] for element, table in self.lut.items()}
        text = self.text_path(image_id)
        masks[TEXT] = np.load(text)["mask"].astype(bool) if text.exists() else np.zeros(labels.shape, bool)
        return masks


def _union(masks, names):
    out = np.zeros(next(iter(masks.values())).shape, dtype=bool)
    for name in names:
        out |= masks[name]
    return out


class Q3MaskBuilder:
    """
        builder = Q3MaskBuilder(q3cfg, cfg, dataset)
        builder.benchmark(sample_ids)        # speed test on 50 images + hours for the whole sample
        builder.segment(sample_ids)          # Mask2Former, cached; skips images already done
        builder.detect_text(sample_ids)      # EasyOCR, cached
        presence = builder.presence(sample_ids)
    """

    def __init__(self, q3cfg, cfg, dataset, id2label=None):
        self.q3cfg = q3cfg
        self.cfg = q3cfg.shared_cfg(cfg)
        self.dataset = dataset
        self.masks = ElementMasks(q3cfg, self.cfg, id2label)
        self.path = Path(q3cfg.out_dir) / "presence.parquet"
        for w in q3cfg.check():
            print("WARN", w)
        if self.masks.unmatched:
            print("WARN class names not found in the segmentation model:", self.masks.unmatched)

    # --- GPU steps (cached) --------------------------------------------------

    def segment(self, ids):
        todo = [i for i in ids if not self.masks.seg_path(i).exists()]
        print(f"Segmentation: {len(ids) - len(todo):,} cached, {len(todo):,} to do")
        if not todo:
            return
        from shared.models import Segmenter, free_gpu_memory
        segmenter = Segmenter(self.cfg)
        for image_id in tqdm(todo, desc="segmentation"):
            segmenter.label_map(self.dataset.get_image(image_id), image_id)
        del segmenter
        free_gpu_memory()

    def detect_text(self, ids):
        todo = [i for i in ids if not self.masks.text_path(i).exists()]
        print(f"Text detection: {len(ids) - len(todo):,} cached, {len(todo):,} to do")
        if not todo:
            return
        from shared.models import TextDetector, free_gpu_memory
        detector = TextDetector(self.cfg)
        for image_id in tqdm(todo, desc="text detection"):
            detector.text_mask(self.dataset.get_image(image_id), image_id)
        del detector
        free_gpu_memory()

    def benchmark(self, ids, n=50):
        """Time segmentation and text detection on n not-yet-cached images (they stay cached).
        Prints the processor's resize setting: if it upscales the small OSV-5M images a lot,
        that is the main cost."""
        from shared.models import Segmenter, TextDetector, free_gpu_memory
        todo = [i for i in ids if not self.masks.seg_path(i).exists() and not self.masks.text_path(i).exists()][:n]
        if not todo:
            print("No uncached images left to time.")
            return None
        images = [self.dataset.get_image(i) for i in todo]
        result = {"images_timed": len(todo), "image_size_example": images[0].size}

        segmenter = Segmenter(self.cfg)
        result["seg_processor_size"] = str(getattr(segmenter.processor, "size", "?"))
        segmenter.label_map(images[0])                                  # warm-up, not cached
        t = time.perf_counter()
        for image_id, image in zip(todo, images):
            segmenter.label_map(image, image_id)
        result["seg_images_per_s"] = len(todo) / (time.perf_counter() - t)
        del segmenter
        free_gpu_memory()

        detector = TextDetector(self.cfg)
        detector.boxes(images[0])
        t = time.perf_counter()
        for image_id, image in zip(todo, images):
            detector.text_mask(image, image_id)
        result["ocr_images_per_s"] = len(todo) / (time.perf_counter() - t)
        del detector
        free_gpu_memory()

        n_all = len(ids)
        result["hours_seg_all"] = n_all / result["seg_images_per_s"] / 3600
        result["hours_ocr_all"] = n_all / result["ocr_images_per_s"] / 3600
        return pd.Series(result)

    # --- presence table (CPU) -------------------------------------------------

    def presence_row(self, image_id, applier):
        """One row: visible fraction (after dilation and crop) and has_<condition> for every condition."""
        q = self.q3cfg
        masks = self.masks.load(image_id)
        row, present = {"id": image_id}, {}
        for condition in q.conditions:
            parts_ok = all(present.get(p, False) for p in q.parts(condition)) if condition in q.combos else True
            if not parts_ok:
                row[f"vis_{condition}"], present[condition] = np.nan, False
                row[f"has_{condition}"] = False
                continue
            raw = _union(masks, q.mask_names(condition))
            vis = applier.visible_fraction(applier.dilate(raw)) if raw.any() else 0.0
            present[condition] = bool(vis >= q.threshold(condition))
            row[f"vis_{condition}"], row[f"has_{condition}"] = vis, present[condition]
        # overlap: how much of the text lies on signs (then "text" and "signs" are not separate clues)
        text = masks[TEXT]
        if "signs" in masks and text.any():
            row["text_on_signs"] = float((text & applier.dilate(masks["signs"])).sum() / text.sum())
        else:
            row["text_on_signs"] = np.nan
        return row

    def presence(self, ids, overwrite=False):
        """Presence table for ids; computed once and saved to results/pair_a/q3/presence.parquet."""
        ids = [str(i) for i in ids]
        if self.path.exists() and not overwrite:
            table = pd.read_parquet(self.path)
            table["id"] = table["id"].astype(str)
            if set(ids) <= set(table["id"]):
                print(f"Presence table loaded from {self.path}")
                return table.set_index("id").loc[ids].reset_index()
            print("Saved presence table does not cover all ids: recomputing")
        missing = [i for i in ids if not self.masks.seg_path(i).exists()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} images have no label map yet: run segment() first")
        no_text = sum(not self.masks.text_path(i).exists() for i in ids)
        if no_text:
            print(f"WARN {no_text} images have no text mask yet (counted as 'no text'): run detect_text()")
        from shared.masking import MaskApplier
        applier = MaskApplier(self.cfg)
        table = pd.DataFrame([self.presence_row(i, applier) for i in tqdm(ids, desc="presence")])
        self.path.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(self.path, index=False)
        return table


# ---------------------------------------------------------------------------
# 3. Running StreetCLIP on the masked versions
# ---------------------------------------------------------------------------

def make_variants(image, image_id, masks, conditions, q3cfg, applier, crop_mode=None):
    """
    All versions of one image: the original, every hidden condition and its random controls.
    Returns [(meta dict, PIL image)]. With crop_mode, the images are already fitted to the
    model input (what the model sees) and returned as uint8 arrays.
    """
    variants = [({"condition": "original", "kind": "original", "rep": 0, "vis": 0.0}, image)]
    for condition in conditions:
        raw = _union(masks, q3cfg.mask_names(condition))
        result = applier.apply(image, raw, condition, image_id)
        if result["image"] is None:
            continue
        variants.append(({"condition": condition, "kind": "real", "rep": 0,
                          "vis": result["visible_fraction"]}, result["image"]))
        if q3cfg.n_controls > 0:
            controls = applier.apply_controls(image, result["mask"], n=q3cfg.n_controls,
                                              seed=_seed(q3cfg.seed, image_id, condition))
            for rep, control in enumerate(controls):
                variants.append(({"condition": condition, "kind": "control", "rep": rep,
                                  "vis": control["visible_fraction"]}, control["image"]))
    if crop_mode is None:
        return variants
    from shared.models import fit_image
    return [(meta, np.asarray(fit_image(img, crop_mode), dtype=np.uint8)) for meta, img in variants]


class _VariantImages:
    """One item = all versions of one image, fitted to 336 x 336 (built in DataLoader workers)."""

    def __init__(self, ids, paths, present, masks, q3cfg, applier, crop_mode):
        self.ids, self.paths, self.present = ids, paths, present
        self.masks, self.q3cfg, self.applier, self.crop_mode = masks, q3cfg, applier, crop_mode

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        image_id = self.ids[i]
        image = Image.open(self.paths[i]).convert("RGB")
        variants = make_variants(image, image_id, self.masks.load(image_id), self.present[i],
                                 self.q3cfg, self.applier, self.crop_mode)
        meta = [dict(m, id=image_id) for m, _ in variants]
        return {"id": image_id, "meta": meta, "pixels": np.stack([p for _, p in variants])}


def _first(batch):
    """collate_fn for batch_size=1 (module-level so Windows workers can pickle it)."""
    return batch[0]


class Q3Runner:
    """
        runner = Q3Runner(q3cfg, cfg, dataset, presence)
        rows, logits, settings = runner.run(model, sample_ids[:20], name="test")   # quick test
        rows, logits, settings = runner.run(model, sample_ids)                     # name = fill mode

    `model` is a GeoCLIPModel with the candidates of the baseline run already set.
    A run is saved in chunks of chunk_size images; running again skips finished chunks.
    """

    def __init__(self, q3cfg, cfg, dataset, presence, masks=None, id2label=None):
        from shared.masking import MaskApplier
        self.q3cfg = q3cfg
        self.cfg = q3cfg.shared_cfg(cfg)
        if self.cfg.fill_mode not in ("gray", "blur"):
            raise ValueError("Q3Runner supports fill_mode 'gray' or 'blur' (inpaint is too slow for ~40 versions per image)")
        self.dataset = dataset
        self.applier = MaskApplier(self.cfg)
        self.masks = masks or ElementMasks(q3cfg, self.cfg, id2label)
        self.presence = presence.assign(id=presence["id"].astype(str)).set_index("id")
        self._norm = None

    def present_conditions(self, image_id):
        row = self.presence.loc[image_id]
        return [c for c in self.q3cfg.conditions if bool(row.get(f"has_{c}", False))]

    def run_dir(self, name):
        return Path(self.q3cfg.out_dir) / "runs" / name

    def _settings(self, model):
        q = self.q3cfg
        return {"model": model.name, "labels": list(model.labels), "prompt_example": model.prompts[0],
                "crop_mode": self.cfg.crop_mode, "fill_mode": self.cfg.fill_mode,
                "mask_dilation_px": self.cfg.mask_dilation_px, "n_controls": q.n_controls,
                "seed": q.seed, "element_classes": q.element_classes, "combos": q.combos,
                "thresholds": {c: q.threshold(c) for c in q.conditions}, "chunk_size": q.chunk_size}

    def run(self, model, ids, name=None, num_workers=None):
        name = name or self.cfg.fill_mode
        out = self.run_dir(name)
        out.mkdir(parents=True, exist_ok=True)
        settings = json.loads(json.dumps(self._settings(model)))   # tuples -> lists, as saved
        settings_path = out / "settings.json"
        if settings_path.exists():
            old = json.loads(settings_path.read_text())
            changed = [k for k in settings if old.get(k) != settings[k] and k != "prompt_example"]
            if changed:
                raise ValueError(f"run {name!r} was made with other settings ({changed}): use another name")
        else:
            settings_path.write_text(json.dumps(settings, indent=2))

        import torch
        ids = [str(i) for i in ids]
        size = self.q3cfg.chunk_size
        chunks = [ids[k:k + size] for k in range(0, len(ids), size)]
        workers = self.cfg.num_workers if num_workers is None else num_workers
        for k, chunk in enumerate(chunks):
            npy = out / f"chunk_{k:04d}.npy"
            if npy.exists():
                continue
            items = _VariantImages(chunk, [self.dataset.image_paths[i] for i in chunk],
                                   [self.present_conditions(i) for i in chunk],
                                   self.masks, self.q3cfg, self.applier, self.cfg.crop_mode)
            loader = torch.utils.data.DataLoader(items, batch_size=1, collate_fn=_first, num_workers=workers)
            metas, logits = [], []
            for item in tqdm(loader, desc=f"{name}: chunk {k + 1}/{len(chunks)}"):
                logits.append(self._logits(model, item["pixels"]))
                metas.extend(item["meta"])
            rows = pd.DataFrame(metas)[["id", "condition", "kind", "rep", "vis"]]
            # write to temporary names first: a crash never leaves a half-written chunk
            rows.to_parquet(out / f"chunk_{k:04d}.parquet.tmp", index=False)
            os.replace(out / f"chunk_{k:04d}.parquet.tmp", out / f"chunk_{k:04d}.parquet")
            with open(out / f"chunk_{k:04d}.npy.tmp", "wb") as f:
                np.save(f, np.concatenate(logits).astype(np.float32))
            os.replace(out / f"chunk_{k:04d}.npy.tmp", npy)
        return self.load(out)

    def _logits(self, model, pixels):
        """uint8 (n, 336, 336, 3) -> logits (n, candidates). Same normalization as CLIP's processor."""
        import torch
        if self._norm is None:
            proc = model.image_processor
            mean = torch.tensor(proc.image_mean, device=model.device).view(1, 3, 1, 1)
            std = torch.tensor(proc.image_std, device=model.device).view(1, 3, 1, 1)
            self._norm = (float(proc.rescale_factor), mean, std)
        scale, mean, std = self._norm
        out = []
        with torch.inference_mode():
            for s in range(0, len(pixels), self.q3cfg.gpu_batch):
                x = torch.from_numpy(pixels[s:s + self.q3cfg.gpu_batch]).to(model.device)
                x = (x.permute(0, 3, 1, 2).float() * scale - mean) / std
                out.append(model.logits_from_embeddings(model.embed_images(x)).float().cpu().numpy())
        return np.concatenate(out)

    @staticmethod
    def load(run_dir):
        """(rows, logits, settings) of a saved run. Only finished chunks are read."""
        run_dir = Path(run_dir)
        npys = sorted(run_dir.glob("chunk_*.npy"))
        if not npys:
            raise FileNotFoundError(f"no finished chunks in {run_dir}")
        rows = pd.concat([pd.read_parquet(p.with_suffix(".parquet")) for p in npys], ignore_index=True)
        rows["id"] = rows["id"].astype(str)
        logits = np.concatenate([np.load(p) for p in npys])
        settings = json.loads((run_dir / "settings.json").read_text())
        if len(rows) != len(logits):
            raise ValueError("rows and logits do not match - delete the last chunk and run again")
        return rows, logits, settings

    @staticmethod
    def check_original(rows, logits, labels, base_preds, base_logits=None):
        """The 'original' versions must reproduce the baseline run (same model, crop and prompts).
        Agreement of top-1 should be ~100 %; logit differences only from fp16 rounding."""
        o = (rows["kind"] == "original").to_numpy()
        ids = rows.loc[o, "id"].to_numpy()
        L = logits[o]
        pred = np.asarray(labels)[L.argmax(1)]
        base = base_preds.assign(id=base_preds["id"].astype(str)).set_index("id")
        result = {"images": len(ids), "top1_agreement": float((pred == base.loc[ids, "pred_1"].to_numpy()).mean())}
        if base_logits is not None:
            idx = pd.Index(base_preds["id"].astype(str)).get_indexer(ids)
            result["max_abs_logit_diff"] = float(np.abs(L - base_logits[idx]).max())
        return pd.Series(result)

    def variant_images(self, image_id, conditions=None):
        """Rebuild the versions of one image as the model sees them (for figures)."""
        image_id = str(image_id)
        conditions = self.present_conditions(image_id) if conditions is None else conditions
        image = self.dataset.get_image(image_id)
        variants = make_variants(image, image_id, self.masks.load(image_id), conditions,
                                 self.q3cfg, self.applier, self.cfg.crop_mode)
        return [(m, Image.fromarray(p)) for m, p in variants]


# ---------------------------------------------------------------------------
# 4. Analysis
# ---------------------------------------------------------------------------

def _split(value, sep=";"):
    if not isinstance(value, str):
        return []
    return [v.strip() for v in value.split(sep) if v.strip()]


class Q3Analyser:
    """
        ana = Q3Analyser(q3cfg, rows, logits, settings["labels"], sample, mapper.table,
                         presence=presence, baseline_preds=preds)
        ana.summary()                  # one row per hidden element: effect vs control, flips
        ana.country_profile("d_true")  # country x element
        ana.interactions()             # do cues compensate each other?
        ana.confusion_shift()          # more neighbour / usual-confusion probability?
        ana.environment(env, "urban")  # does the effect depend on the environment?

    Every effect is paired: masked minus original of the same image.
    `x` (excess) = effect of the real mask minus the mean effect of its random controls
    (same shape and area at a random place), so pure "less image" effects cancel out.
    Uncertainty: bootstrap over camera sequences (pooled) or over countries (macro).
    """

    def __init__(self, q3cfg, rows, logits, labels, sample, countries, presence=None, baseline_preds=None):
        self.q3cfg = q3cfg
        self.labels = list(labels)
        self.countries = countries
        self.presence = presence
        s = sample.copy()
        s["id"] = s["id"].astype(str)
        s["sequence"] = s["sequence"].astype("string").fillna("img_" + s["id"]).astype(str)
        self.sample = s.set_index("id")
        missing = set(rows["id"].astype(str)) - set(self.sample.index)
        if missing:
            raise ValueError(f"{len(missing)} image ids in the run are not in the sample")
        self._matrices(baseline_preds)
        self.rows = self._row_metrics(rows, logits)
        self.effects, self.flips = self._effects()

    # --- preparation ----------------------------------------------------------

    def _matrices(self, baseline_preds):
        """neighbour / same-subregion / usual-confusion matrices over the candidate labels."""
        lab = {c: i for i, c in enumerate(self.labels)}
        n = len(self.labels)
        table = self.countries.reindex(self.labels)
        nb = np.zeros((n, n), dtype=bool)
        for c, value in zip(self.labels, table["neighbors"] if "neighbors" in table else [""] * n):
            for other in _split(value):
                if other in lab:
                    nb[lab[c], lab[other]] = True
        sub = table["world_subregion"].fillna("").astype(str).to_numpy() if "world_subregion" in table else np.full(n, "")
        same_sub = (sub[:, None] == sub[None, :]) & (sub[:, None] != "")
        np.fill_diagonal(same_sub, False)

        sim = np.zeros((n, n), dtype=bool)
        self.similar = {}
        if baseline_preds is not None:
            b = baseline_preds[baseline_preds["representable"].astype(bool)]
            wrong = b[b["pred_1"] != b["country_id"]]
            counts = wrong.groupby(["country_id", "pred_1"]).size().sort_values(ascending=False)
            for (true, pred), _ in counts.items():
                if true in lab and pred in lab and len(self.similar.setdefault(true, [])) < self.q3cfg.n_similar:
                    self.similar[true].append(str(pred))
                    sim[lab[true], lab[pred]] = True
        self.lab, self.nb, self.same_sub, self.sim = lab, nb, same_sub, sim

    def _row_metrics(self, rows, logits):
        r = rows.reset_index(drop=True).copy()
        r["id"] = r["id"].astype(str)
        r["country_id"] = r["id"].map(self.sample["country_id"])
        bad = sorted(set(r["country_id"].dropna()) - set(self.lab))
        if bad or r["country_id"].isna().any():
            raise ValueError(f"true countries that are not candidates: {bad}")
        t = r["country_id"].map(self.lab).to_numpy()

        o = (r["kind"] == "original").to_numpy()
        orig_pred = pd.Series(logits[o].argmax(1), index=r.loc[o, "id"].to_numpy())
        if orig_pred.index.duplicated().any():
            raise ValueError("an image has more than one 'original' row")
        op = r["id"].map(orig_pred)
        if op.isna().any():
            raise ValueError(f"{r.loc[op.isna(), 'id'].nunique()} images have no 'original' row")
        op = op.to_numpy().astype(int)

        n = len(r)
        cols = {k: np.empty(n, dtype=np.float32) for k in ("p_max", "p_true", "p_orig", "entropy", "nb_mass", "sim_mass")}
        pred = np.empty(n, dtype=np.int64)
        for s in range(0, n, 50_000):
            e = min(s + 50_000, n)
            L = logits[s:e].astype(np.float64)
            L -= L.max(axis=1, keepdims=True)
            P = np.exp(L)
            P /= P.sum(axis=1, keepdims=True)
            idx = np.arange(e - s)
            pred[s:e] = P.argmax(1)
            cols["p_max"][s:e] = P.max(1)
            cols["p_true"][s:e] = P[idx, t[s:e]]
            cols["p_orig"][s:e] = P[idx, op[s:e]]
            cols["entropy"][s:e] = -(P * np.log2(np.clip(P, 1e-12, 1))).sum(1)
            cols["nb_mass"][s:e] = (P * self.nb[t[s:e]]).sum(1)
            cols["sim_mass"][s:e] = (P * self.sim[t[s:e]]).sum(1)
        for k, v in cols.items():
            r[k] = v
        r["pred"] = np.asarray(self.labels)[pred]
        r["correct"] = pred == t
        r["pred_is_nb"] = self.nb[t, pred]
        r["pred_same_sub"] = self.same_sub[t, pred]
        r["pred_is_sim"] = self.sim[t, pred]
        return r

    def _deltas(self, d, O):
        o = O.loc[d["id"].to_numpy()]
        c0, c1 = o["correct"].to_numpy(bool), d["correct"].to_numpy(bool)
        r2w = c0 & ~c1
        out = pd.DataFrame({
            "id": d["id"].to_numpy(), "condition": d["condition"].to_numpy(), "vis": d["vis"].to_numpy(),
            "d_true": d["p_true"].to_numpy() - o["p_true"].to_numpy(),
            "d_orig": d["p_orig"].to_numpy() - o["p_max"].to_numpy(),
            "d_conf": d["p_max"].to_numpy() - o["p_max"].to_numpy(),
            "d_entropy": d["entropy"].to_numpy() - o["entropy"].to_numpy(),
            "flip": (d["pred"].to_numpy() != o["pred"].to_numpy()).astype(float),
            "r2w": np.where(c0, (~c1).astype(float), np.nan),
            "w2r": np.where(~c0, c1.astype(float), np.nan),
            "d_nb": d["nb_mass"].to_numpy() - o["nb_mass"].to_numpy(),
            "d_sim": d["sim_mass"].to_numpy() - o["sim_mass"].to_numpy(),
        })
        # where do right -> wrong flips go?
        for col, src in (("to_nb", "pred_is_nb"), ("to_sub", "pred_same_sub"), ("to_sim", "pred_is_sim")):
            out[col] = np.where(r2w, d[src].to_numpy().astype(float), np.nan)
        out["pred"] = d["pred"].to_numpy()
        return out

    def _effects(self):
        r = self.rows
        O = r[r["kind"] == "original"].set_index("id")
        real = self._deltas(r[r["kind"] == "real"], O)
        ctrl = self._deltas(r[r["kind"] == "control"], O)
        num = ["vis", *METRICS, "to_nb", "to_sub", "to_sim"]
        ctrl_mean = ctrl.groupby(["id", "condition"])[num].mean()
        eff = real.set_index(["id", "condition"])[num].join(ctrl_mean, rsuffix="_ctrl")
        for m in METRICS:
            eff[f"{m}_x"] = eff[m] - eff[f"{m}_ctrl"]
        eff = eff.reset_index()
        eff["country_id"] = eff["id"].map(self.sample["country_id"])
        eff["sequence"] = eff["id"].map(self.sample["sequence"])
        eff["correct0"] = eff["id"].map(O["correct"])
        eff["pred0"] = eff["id"].map(O["pred"])
        for col in ("world_subregion", "continent"):
            if col in self.countries:
                eff[col] = eff["country_id"].map(self.countries[col])
        order = {c: i for i, c in enumerate(self.q3cfg.conditions)}
        eff = eff.sort_values(["condition", "id"], key=lambda s: s.map(order) if s.name == "condition" else s)

        flips = pd.concat([real.assign(kind="real"), ctrl.assign(kind="control")])
        flips = flips[flips["r2w"] == 1].copy()
        flips["country_id"] = flips["id"].map(self.sample["country_id"])
        return eff.reset_index(drop=True), flips

    # --- bootstrap helpers ----------------------------------------------------

    def _weights(self, clusters):
        codes, inv = np.unique(np.asarray(clusters).astype(str), return_inverse=True)
        C = len(codes)
        rng = np.random.default_rng(self.q3cfg.seed)
        W = rng.multinomial(C, np.full(C, 1.0 / C), size=self.q3cfg.n_boot).astype(float) if C > 1 else None
        return inv, C, W

    @staticmethod
    def _boot(values, inv, C, W):
        v = np.asarray(values, dtype=float)
        ok = ~np.isnan(v)
        sums = np.bincount(inv[ok], weights=v[ok], minlength=C)
        cnts = np.bincount(inv[ok], minlength=C).astype(float)
        if cnts.sum() == 0:
            return np.nan, np.nan, np.nan
        mean = sums.sum() / cnts.sum()
        if W is None or (cnts > 0).sum() < 2:
            return mean, np.nan, np.nan
        with np.errstate(invalid="ignore", divide="ignore"):
            boots = (W @ sums) / (W @ cnts)
        lo, hi = np.nanpercentile(boots, [2.5, 97.5])
        return mean, lo, hi

    def _stats(self, g, metrics, level, cluster="sequence"):
        """One summary row for the effects in g."""
        row = {"images": len(g), "countries": g["country_id"].nunique(), "vis": g["vis"].mean()}
        if level == "pooled":
            inv, C, W = self._weights(g[cluster])
            for m in metrics:
                row[m], row[f"{m}_ctrl"] = g[m].mean(), g[f"{m}_ctrl"].mean()
                row[f"{m}_x"], row[f"{m}_lo"], row[f"{m}_hi"] = self._boot(g[f"{m}_x"].to_numpy(), inv, C, W)
            return row
        if level != "macro":
            raise ValueError("level must be 'macro' or 'pooled'")
        sizes = g.groupby("country_id").size()
        keep = sizes.index[sizes >= self.q3cfg.min_country_images]
        cols = [c for m in metrics for c in (m, f"{m}_ctrl", f"{m}_x")]
        cm = g[g["country_id"].isin(keep)].groupby("country_id")[cols].mean()
        row["countries"] = len(cm)
        inv, C, W = self._weights(cm.index)
        for m in metrics:
            row[m], row[f"{m}_ctrl"] = cm[m].mean(), cm[f"{m}_ctrl"].mean()
            row[f"{m}_x"], row[f"{m}_lo"], row[f"{m}_hi"] = self._boot(cm[f"{m}_x"].to_numpy(), inv, C, W)
        return row

    # --- results ---------------------------------------------------------------

    def summary(self, metrics=("d_true", "d_orig", "flip", "r2w", "w2r"), level="macro", by=None,
                conditions=None):
        """
        One row per hidden condition (and per `by` group, e.g. 'continent' or 'correct0').
        For every metric m: m (real mask), m_ctrl (random control), m_x (difference) with 95 % CI.
        level='macro': every country counts once (needs min_country_images); 'pooled': every image.
        """
        e = self.effects if conditions is None else self.effects[self.effects["condition"].isin(conditions)]
        keys = ["condition"] + ([by] if by else [])
        rows = []
        for k, g in e.groupby(keys, sort=False):
            k = k if isinstance(k, tuple) else (k,)
            rows.append({**dict(zip(keys, k)), **self._stats(g, metrics, level)})
        return pd.DataFrame(rows).set_index(keys)

    def country_profile(self, metric="d_true", group="country_id", min_n=None, conditions=None):
        """Long table (group, condition): n, real, control, excess x with CI (bootstrap over sequences).
        Cells with fewer than min_n images are dropped. group: 'country_id', 'world_subregion', 'continent'."""
        min_n = self.q3cfg.min_n if min_n is None else min_n
        e = self.effects if conditions is None else self.effects[self.effects["condition"].isin(conditions)]
        rows = []
        for (grp, cond), g in e.groupby([group, "condition"], sort=False):
            if len(g) < min_n:
                continue
            inv, C, W = self._weights(g["sequence"])
            x, lo, hi = self._boot(g[f"{metric}_x"].to_numpy(), inv, C, W)
            rows.append({group: grp, "condition": cond, "images": len(g), "sequences": C,
                         "real": g[metric].mean(), "ctrl": g[f"{metric}_ctrl"].mean(), "x": x, "lo": lo, "hi": hi})
        t = pd.DataFrame(rows, columns=[group, "condition", "images", "sequences", "real", "ctrl", "x", "lo", "hi"])
        if group == "country_id" and len(t) and "prompt_name" in self.countries:
            t["name"] = t["country_id"].map(self.countries["prompt_name"])
        t["significant"] = (t["lo"] > 0) | (t["hi"] < 0)
        return t

    def strongest_cue(self, profile, conditions=None):
        """Per country (or group): the condition whose hiding hurts P(true) most (most negative x)."""
        p = profile if conditions is None else profile[profile["condition"].isin(conditions)]
        group = p.columns[0]
        best = p.sort_values("x").groupby(group, sort=False).head(1).set_index(group)
        return best.sort_values("x")

    def interactions(self, metric="d_true"):
        """
        Do two cues compensate each other? For each whole = A + B (see q3cfg.interactions),
        on images where A, B and the whole were all hidden:
            effect(A), effect(B), effect(A) + effect(B), effect(whole), extra = whole - (A + B)
        Effects are control-corrected (x) and signed so that > 0 = the prediction suffers more.
        extra > 0 (CI above 0): hiding both hurts more than the two alone -> the cues back each
        other up (when one is hidden, the other still carries the information).
        extra < 0: the parts share information or the effect saturates.
        """
        sign = SIGN.get(metric, 1)
        wide = self.effects.pivot_table(index="id", columns="condition", values=f"{metric}_x", aggfunc="first")
        seq = self.sample["sequence"]
        rows = []
        for whole, parts in self.q3cfg.interactions.items():
            cols = [whole, *parts]
            if not all(c in wide for c in cols):
                continue
            w = wide[cols].dropna() * sign
            if w.empty:
                continue
            extra = w[whole] - w[list(parts)].sum(axis=1)
            inv, C, W = self._weights(seq.loc[w.index])
            x, lo, hi = self._boot(extra.to_numpy(), inv, C, W)
            row = {"whole": whole, "parts": " + ".join(parts), "images": len(w)}
            for i, p in enumerate(parts, 1):
                row[f"part_{i}"], row[f"effect_{i}"] = p, w[p].mean()
            row["sum_parts"] = w[list(parts)].sum(axis=1).mean()
            row["effect_whole"] = w[whole].mean()
            row["extra"], row["extra_lo"], row["extra_hi"] = x, lo, hi
            row["reading"] = ("cues compensate each other" if lo > 0 else
                              "parts overlap / effect saturates" if hi < 0 else "about additive (no clear interaction)")
            rows.append(row)
        return pd.DataFrame(rows).set_index("whole") if rows else pd.DataFrame()

    def confusion_shift(self, level="pooled"):
        """
        Does hiding an element push probability to neighbours / usual confusions of the true country?
          d_nb_x, d_sim_x     control-corrected change in P(land neighbours) / P(top-n usual confusions)
          flips_real / flips_ctrl   right -> wrong flips per image set (control: mean over the controls)
          to_nb, to_sub, to_sim     share of these flips that land on a neighbour / same subregion / usual confusion
        Reference row 'original errors': where the model's own errors land on the unmasked images.
        """
        s = self.summary(("d_nb", "d_sim", "d_true"), level=level)
        f = self.flips.groupby(["condition", "kind"]).agg(n=("id", "size"), to_nb=("to_nb", "mean"),
                                                          to_sub=("to_sub", "mean"), to_sim=("to_sim", "mean"))
        f = f.unstack("kind")
        out = s[["images", "d_nb_x", "d_nb_lo", "d_nb_hi", "d_sim_x", "d_sim_lo", "d_sim_hi", "d_true_x"]].copy()
        for col in ("n", "to_nb", "to_sub", "to_sim"):
            for kind in ("real", "control"):
                key = (col, kind)
                out[f"{col}_{'real' if kind == 'real' else 'ctrl'}"] = f[key].reindex(out.index) if key in f else np.nan
        out = out.rename(columns={"n_real": "flips_real", "n_ctrl": "flips_ctrl"})
        out["flips_ctrl"] = out["flips_ctrl"] / max(self.q3cfg.n_controls, 1)
        o = self.rows[(self.rows["kind"] == "original") & ~self.rows["correct"]]
        ref = pd.DataFrame({"flips_real": [len(o)], "to_nb_real": [o["pred_is_nb"].mean()],
                            "to_sub_real": [o["pred_same_sub"].mean()], "to_sim_real": [o["pred_is_sim"].mean()]},
                           index=["original errors"])
        return pd.concat([out, ref])

    def flip_destinations(self, condition, top=15, sort="real"):
        """Right -> wrong flips of one condition: true country -> new guess, real vs control count.
        sort='excess': by real - control, i.e. the pairs that hiding the element itself creates."""
        f = self.flips[self.flips["condition"] == condition]
        real = f[f["kind"] == "real"].groupby(["country_id", "pred"]).size().rename("real")
        ctrl = (f[f["kind"] == "control"].groupby(["country_id", "pred"]).size()
                / max(self.q3cfg.n_controls, 1)).rename("ctrl")
        t = pd.concat([real, ctrl], axis=1).fillna(0)
        t["excess"] = t["real"] - t["ctrl"]
        t = t.sort_values("excess" if sort == "excess" else "real", ascending=False).head(top)
        t["neighbour"] = [self.nb[self.lab[a], self.lab[b]] for a, b in t.index]
        t["usual_confusion"] = [self.sim[self.lab[a], self.lab[b]] for a, b in t.index]
        return t

    def environment(self, env, var, metric="d_true", min_cell=10, min_countries=5):
        """
        Does the effect depend on the environment (e.g. var='urban' or 'climate_group' from
        q4.add_env_columns)? Like Q4, environment and country are mixed up, so besides the pooled
        effect there is a within-country difference: cell effect minus the country's own effect,
        averaged over countries (cells need min_cell images, rows need min_countries countries).
        """
        e = self.effects.merge(env[["id", var]].assign(id=env["id"].astype(str)), on="id", how="left")
        e = e[e[var].notna()]
        x = f"{metric}_x"
        rows = []
        country_mean = e.groupby(["condition", "country_id"])[x].mean()
        for (cond, cat), g in e.groupby(["condition", var], sort=False):
            inv, C, W = self._weights(g["sequence"])
            mean, lo, hi = self._boot(g[x].to_numpy(), inv, C, W)
            cells = g.groupby("country_id")[x].agg(["mean", "size"])
            cells = cells[cells["size"] >= min_cell]
            d = cells["mean"] - country_mean.loc[cond].reindex(cells.index)
            ok = len(d) >= min_countries
            rows.append({"condition": cond, var: cat, "images": len(g), "countries": g["country_id"].nunique(),
                         "x": mean, "lo": lo, "hi": hi, "within_countries": len(d),
                         "within_diff": d.mean() if ok else np.nan,
                         "within_ci": 1.96 * d.sem() if ok and len(d) > 1 else np.nan})
        return pd.DataFrame(rows).set_index(["condition", var])

    def presence_overview(self):
        """Per condition: how many sample images contain it, mean visible share, countries with >= min_n."""
        if self.presence is None:
            raise ValueError("Q3Analyser was created without presence")
        p = self.presence.assign(id=self.presence["id"].astype(str)).set_index("id")
        p = p.loc[p.index.intersection(self.sample.index)]
        country = self.sample.loc[p.index, "country_id"]
        rows = []
        for c in self.q3cfg.conditions:
            has = p[f"has_{c}"].astype(bool)
            per_country = has.groupby(country).sum()
            rows.append({"condition": c, "images": int(has.sum()), "share": has.mean(),
                         "mean_vis": p.loc[has, f"vis_{c}"].mean(),
                         f"countries_n>={self.q3cfg.min_n}": int((per_country >= self.q3cfg.min_n).sum())})
        t = pd.DataFrame(rows).set_index("condition")
        if "text_on_signs" in p:
            t.loc["text", "text_on_signs"] = p["text_on_signs"].mean()
        return t

    def presence_by_country(self):
        """Country x condition: share of the country's sample images that contain the condition."""
        p = self.presence.assign(id=self.presence["id"].astype(str)).set_index("id")
        p = p.loc[p.index.intersection(self.sample.index)]
        cols = [f"has_{c}" for c in self.q3cfg.conditions]
        t = p[cols].astype(float).groupby(self.sample.loc[p.index, "country_id"]).mean()
        t.columns = self.q3cfg.conditions
        t.insert(0, "images", self.sample.loc[p.index].groupby("country_id").size())
        return t

    def pick(self, condition, which="r2w", n=4, seed=None):
        """Random image ids (not hand-picked) for examples: which = 'r2w', 'w2r', 'flip' or 'any'."""
        e = self.effects[self.effects["condition"] == condition]
        if which != "any":
            e = e[e[which] == 1]
        if e.empty:
            return []
        return e.sample(min(n, len(e)), random_state=self.q3cfg.seed if seed is None else seed)["id"].tolist()


# ---------------------------------------------------------------------------
# 5. Tables and plots
# ---------------------------------------------------------------------------

COLORS = {"road_surface": (255, 0, 255), "lane_markings": (255, 255, 255), "vegetation": (0, 200, 0),
          "terrain": (170, 120, 50), "sky": (0, 150, 255), "buildings": (255, 150, 0),
          "vehicles": (255, 0, 0), "signs": (255, 255, 0), "poles": (120, 120, 120),
          "ego_vehicle": (0, 255, 255), "text": (255, 0, 120)}


# one colour per cue for plots that colour by cue (COLORS above is for image overlays: lane markings are white)
CUE_COLORS = {"road": "#7570b3", "road_surface": "#7570b3", "lane_markings": "#e7298a",
              "vegetation_terrain": "#1b9e77", "vegetation": "#1b9e77", "terrain": "#a6761d", "sky": "#1f78b4",
              "buildings": "#d95f02", "vehicles": "#e31a1c", "signs_poles": "#e6ab02", "signs": "#e6ab02",
              "poles": "#666666", "ego_vehicle": "#17becf", "text": "#222222"}


def subset(summary, conditions):
    """Rows of a summary for these conditions, in this order (missing ones are skipped)."""
    keys = summary.index.get_level_values(0)
    return summary[keys.isin(conditions)].sort_index(
        level=0, key=lambda s: s.map({c: i for i, c in enumerate(conditions)}), sort_remaining=False)


def effect_order(summary, metric="d_true"):
    """Conditions sorted by the control-corrected effect (strongest first), e.g. to give two plots one order."""
    s = summary[f"{metric}_x"].groupby(level=0).mean()
    return s.sort_values(ascending=SIGN.get(metric, 1) == -1).index.tolist()


def _clean(ax):
    ax.spines[["top", "right"]].set_visible(False)


def _fmt(metric, v, signed=True):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if metric in RATES and not signed:
        return f"{v * 100:.1f} %"
    if metric in RATES or metric in POINTS:
        return f"{v * 100:+.1f} pts" if signed else f"{v * 100:.1f}"
    return f"{v:+.2f}" if signed else f"{v:.2f}"


class Q3Reporter:
    """Readable tables, plots and example images. Titles: write the finding, then check it.
    Q3Reporter(None, runner) is enough for show_overlays (checking the masks before any run)."""

    def __init__(self, analyser, runner=None):
        self.ana = analyser
        self.runner = runner

    # --- tables ------------------------------------------------------------------

    def table(self, summary, metrics=None):
        """Summary as readable text: per metric 'real (control) -> difference [95 % CI]'."""
        metrics = metrics or [m for m in METRICS if f"{m}_x" in summary]
        out = pd.DataFrame(index=summary.index)
        out["Images"] = summary["images"].map(lambda v: f"{v:,.0f}")
        out["Countries"] = summary["countries"].map(lambda v: f"{v:,.0f}")
        out["Hidden (share of view)"] = summary["vis"].map(lambda v: f"{v * 100:.0f} %")
        for m in metrics:
            out[METRIC_NAMES.get(m, m)] = [
                f"{_fmt(m, r[m], m not in RATES)} (ctrl {_fmt(m, r[f'{m}_ctrl'], m not in RATES)}) "
                f"-> {_fmt(m, r[f'{m}_x'])} [{_fmt(m, r[f'{m}_lo'])}, {_fmt(m, r[f'{m}_hi'])}]"
                for _, r in summary.iterrows()]
        return out

    # --- plots -------------------------------------------------------------------
    # Sign convention everywhere (except plot_interactions): change in P(true) vs control,
    # < 0 = hiding the element hurts more than hiding random pixels of the same size.

    def _name(self, c):
        """'CA' -> 'Canada' (falls back to the code)."""
        t = self.ana.countries
        if "prompt_name" in t and c in t.index and isinstance(t.loc[c, "prompt_name"], str):
            return t.loc[c, "prompt_name"]
        return str(c)

    def plot_presence(self, overview, conditions=FINE, title=None):
        """Q3.0: how often each element is there, how big it is, and in how many countries it can be analysed."""
        t = overview.reindex([c for c in conditions if c in overview.index])
        ncol = next(c for c in t.columns if c.startswith("countries_n"))
        total = self.ana.sample["country_id"].nunique()
        y = np.arange(len(t))
        fig, axes = plt.subplots(1, 3, figsize=(12.5, 0.4 * len(t) + 1.9), sharey=True)
        panels = [("share", 100, "images that contain it (%)", "{:.0f} %", "#6baed6"),
                  ("mean_vis", 100, "share of the view it covers when present (%)", "{:.1f} %", "#fd8d3c"),
                  (ncol, 1, f"countries with >= {self.ana.q3cfg.min_n} such images (of {total})", "{:.0f}", "#74c476")]
        for ax, (col, scale, label, fmt, color) in zip(axes, panels):
            v = t[col].to_numpy(float) * scale
            ax.barh(y, v, color=color)
            for yy, vv in zip(y, v):
                ax.text(vv, yy, " " + fmt.format(vv), va="center", fontsize=8)
            ax.set_xlim(0, np.nanmax(v) * 1.2)
            ax.set_xlabel(label, fontsize=9)
            _clean(ax)
        axes[0].set_yticks(y, [f"{c}  (n={n:,.0f})" for c, n in zip(t.index, t["images"])])
        axes[0].invert_yaxis()
        fig.suptitle(title or "How often each element is visible in the sample", fontsize=11)
        plt.tight_layout()
        return fig

    def plot_effects(self, summary, metric="d_true", title=None, ax=None, order=None):
        """Grey = random control, blue = real mask, bar = difference with 95 % CI (one row per condition).
        order: list of conditions (top to bottom); default: strongest effect on top."""
        s = summary.reset_index()
        if order is not None:
            s = s.set_index("condition").loc[[c for c in order if c in set(s["condition"])]].reset_index()
        else:
            s = s.sort_values(f"{metric}_x", ascending=SIGN.get(metric, 1) == -1)   # strongest effect on top
        labels = s["condition"].astype(str) + "  (n=" + s["images"].map("{:,.0f}".format) + ")"
        scale = 100
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 0.42 * len(s) + 1.4))
        y = np.arange(len(s))
        x = s[f"{metric}_x"] * scale
        err = np.vstack([x - s[f"{metric}_lo"] * scale, s[f"{metric}_hi"] * scale - x])
        ax.barh(y, x, xerr=np.nan_to_num(err), color="#9ecae1", capsize=3, label="real - control (95 % CI)")
        ax.scatter(s[metric] * scale, y, color="#08519c", zorder=3, s=18, label="real mask")
        ax.scatter(s[f"{metric}_ctrl"] * scale, y, color="#888", marker="|", zorder=3, s=120, label="random control")
        hi_end = (s[f"{metric}_hi"] * scale).fillna(x).to_numpy()
        for yy, xx, he in zip(y, x, hi_end):
            ax.annotate(f"{xx:+.1f}", (0 if xx < 0 else he, yy), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=7, color="#08306b")
        lo_, hi_ = ax.get_xlim()
        ax.set_xlim(lo_, hi_ + 0.08 * (hi_ - lo_))
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        unit = "%" if metric in RATES else "points"
        ax.set_xlabel(f"{METRIC_NAMES.get(metric, metric)} ({unit})")
        ax.set_title(title or METRIC_NAMES.get(metric, metric), fontsize=10)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09 - 1.0 / (len(s) + 3)), ncol=3, fontsize=8,
                  frameon=False)
        _clean(ax)
        plt.tight_layout()
        return ax.figure

    def plot_flips(self, summary, title=None, order=None):
        """Right -> wrong and wrong -> right rates: real mask vs random control (values printed)."""
        s = summary.reset_index()
        if order is not None:
            s = s.set_index("condition").loc[[c for c in order if c in set(s["condition"])]].reset_index()
        else:
            s = s.sort_values("r2w_x", ascending=False)
        y = np.arange(len(s))
        fig, axes = plt.subplots(1, 2, figsize=(11, 0.42 * len(s) + 1.6), sharey=True, sharex=True)   # same scale
        for ax, m, col in ((axes[0], "r2w", "#de2d26"), (axes[1], "w2r", "#31a354")):
            real, ctrl = s[m] * 100, s[f"{m}_ctrl"] * 100
            ax.barh(y - 0.2, real, height=0.4, color=col, label="real mask")
            ax.barh(y + 0.2, ctrl, height=0.4, color="#bbb", label="random control")
            for yy, r, c in zip(y, real, ctrl):
                ax.text(r, yy - 0.2, f" {r:.0f} %", va="center", fontsize=7)
                ax.text(c, yy + 0.2, f" {c:.0f} %", va="center", fontsize=7, color="#666")
            ax.set_xlabel(f"{METRIC_NAMES[m]} (%)")
            ax.legend(fontsize=8, loc="lower right")
            _clean(ax)
        axes[0].set_yticks(y, s["condition"])
        axes[0].invert_yaxis()
        fig.suptitle(title or "Flips: real mask vs random control of the same size", fontsize=11)
        plt.tight_layout()
        return fig

    def plot_split(self, summary, metric="d_true", groups=None, title=None):
        """Q3.2. Left: each whole element (bold) and its parts, control-corrected effect with 95 % CI.
        Right: effect vs share of the view that is hidden (log scale) for every part / single element:
        small elements can matter more than large ones."""
        groups = groups or SPLITS
        s = summary.reset_index().set_index("condition")
        rows, ys, bold, y = [], [], [], 0.0
        for parent, parts in groups.items():
            rows.append(parent)
            ys.append(y)
            bold.append(True)
            y += 1
            for p in parts:
                if p in s.index:
                    rows.append(p)
                    ys.append(y)
                    bold.append(False)
                    y += 1
            y += 0.5
        ys = np.array(ys)
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 0.4 * len(rows) + 2.0), gridspec_kw={"width_ratios": [1.15, 1]})
        labels = []
        for c, yy, is_parent in zip(rows, ys, bold):
            if c not in s.index:          # a heading only, e.g. "other"
                labels.append(c)
                continue
            r = s.loc[c]
            x, lo, hi = r[f"{metric}_x"] * 100, r[f"{metric}_lo"] * 100, r[f"{metric}_hi"] * 100
            ax.barh(yy, x, color="#bdbdbd" if is_parent else CUE_COLORS.get(c, "#6baed6"), height=0.7,
                    xerr=[[x - lo], [hi - x]] if np.isfinite(lo) else None, capsize=3)
            ax.text(hi if x >= 0 else lo, yy, f" {x:+.1f}" if x >= 0 else f"{x:+.1f} ", va="center", fontsize=8,
                    ha="left" if x >= 0 else "right")
            labels.append(f"{c}  ({r['vis'] * 100:.0f} % of view)" if is_parent else
                          f"      {c}  ({r['vis'] * 100:.1f} % of view)")
        ax.set_yticks(ys, labels)
        for tick, is_parent in zip(ax.get_yticklabels(), bold):
            tick.set_fontweight("bold" if is_parent else "normal")
        ax.invert_yaxis()
        ax.axvline(0, color="k", lw=0.8)
        lo_lim = min(ax.get_xlim()[0], -1)
        ax.set_xlim(lo_lim * 1.15, max(ax.get_xlim()[1], 1) * 1.3)
        ax.set_xlabel(f"{METRIC_NAMES.get(metric, metric)} vs random control (points)")
        ax.set_title("whole element (grey) and its parts", fontsize=10)
        _clean(ax)

        parents = {p for p in groups if p in s.index}
        sc = s[~s.index.isin(parents)]
        for c, r in sc.iterrows():
            x, yv = r["vis"] * 100, r[f"{metric}_x"] * 100
            err = [[yv - r[f"{metric}_lo"] * 100], [r[f"{metric}_hi"] * 100 - yv]]
            ax2.errorbar(x, yv, yerr=err, fmt="o", color=CUE_COLORS.get(c, "#333"), ms=7, capsize=3)
            ax2.annotate(c, (x, yv), xytext=(6, 4), textcoords="offset points", fontsize=8)
        from matplotlib.ticker import FixedLocator, NullFormatter
        ax2.set_xscale("log")
        vmin, vmax = sc["vis"].min() * 100, sc["vis"].max() * 100
        ax2.set_xlim(vmin / 1.6, vmax * 1.8)
        ticks = [t for t in (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100) if vmin / 1.6 <= t <= vmax * 1.8]
        ax2.xaxis.set_major_locator(FixedLocator(ticks))
        ax2.set_xticks(ticks, [f"{t:g}" for t in ticks])
        ax2.xaxis.set_minor_formatter(NullFormatter())
        ax2.axhline(0, color="k", lw=0.8)
        ax2.set_xlabel("share of the view that is hidden (%, log scale)")
        ax2.set_ylabel(f"{METRIC_NAMES.get(metric, metric)} vs control (points)")
        ax2.set_title("effect vs size of the hidden element", fontsize=10)
        _clean(ax2)
        fig.suptitle(title or "Split elements: which part carries the information?", fontsize=11)
        plt.tight_layout()
        return fig

    def plot_confidence(self, summary, title=None):
        """Q3.3: control-corrected change in confidence (top-1 prob.) and entropy, for images that were
        originally right vs originally wrong. Input: summary(metrics=('d_conf', 'd_entropy'), by='correct0')."""
        s = summary.reset_index()
        s["correct0"] = s["correct0"].astype(bool)
        right = s[s["correct0"]].set_index("condition")
        conds = right["d_conf_x"].sort_values().index.tolist()
        conds += [c for c in dict.fromkeys(s["condition"]) if c not in conds]
        y = np.arange(len(conds))
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 0.45 * len(conds) + 2.0), sharey=True)
        groups = ((True, "originally right", "#08519c", -0.17), (False, "originally wrong", "#fd8d3c", 0.17))
        panels = ((axes[0], "d_conf", 100, "{:+.1f}",
                   "change in confidence vs control (points; < 0 = less sure)"),
                  (axes[1], "d_entropy", 1, "{:+.2f}",
                   "change in entropy vs control (bits; > 0 = probability more spread out)"))
        for ax, m, scale, fmt, label in panels:
            for flag, name, color, off in groups:
                g = s[s["correct0"] == flag].set_index("condition").reindex(conds)
                x = g[f"{m}_x"].to_numpy(float) * scale
                lo, hi = g[f"{m}_lo"].to_numpy(float) * scale, g[f"{m}_hi"].to_numpy(float) * scale
                ax.errorbar(x, y + off, xerr=np.nan_to_num(np.vstack([x - lo, hi - x])), fmt="o", ms=5,
                            color=color, capsize=2, label=name)
                for xx, hh, yy in zip(x, hi, y + off):
                    if np.isfinite(xx):
                        ax.text(hh if np.isfinite(hh) else xx, yy, " " + fmt.format(xx), va="center", fontsize=7,
                                color=color)
            ax.axvline(0, color="k", lw=0.8)
            ax.set_xlabel(label, fontsize=9)
            _clean(ax)
        axes[0].set_yticks(y, conds)
        axes[0].invert_yaxis()
        axes[0].legend(fontsize=8, loc="lower left")
        fig.suptitle(title or "Confidence and uncertainty after hiding an element", fontsize=11)
        plt.tight_layout()
        return fig

    def plot_destinations(self, shift, conditions=None, min_flips=100, title=None):
        """Q3.4: where do the right -> wrong flips go? Share landing on a land neighbour / the same
        subregion / one of the usual confusions: real mask (dot) vs random control (ring).
        Dashed line: where the model's own errors on the unmasked images land.
        Faded: fewer than min_flips flips. Input: confusion_shift() (pooled flips)."""
        ref = shift.loc["original errors"] if "original errors" in shift.index else None
        t = shift.drop(index="original errors", errors="ignore")
        if conditions is not None:
            t = t.reindex([c for c in conditions if c in t.index])
        t = t.assign(_d=t["to_sim_real"] - t["to_sim_ctrl"]).sort_values("_d")
        y = np.arange(len(t))
        panels = (("to_nb", "lands on a land neighbour"), ("to_sub", "lands in the same subregion"),
                  ("to_sim", "lands on a usual confusion (top 5)"))
        fig, axes = plt.subplots(1, 3, figsize=(14, 0.42 * len(t) + 2.0), sharey=True)
        for ax, (col, label) in zip(axes, panels):
            real, ctrl = t[f"{col}_real"].to_numpy(float) * 100, t[f"{col}_ctrl"].to_numpy(float) * 100
            for k in range(len(t)):
                alpha = 0.35 if t["flips_real"].iloc[k] < min_flips else 1.0
                color = "#de2d26" if real[k] < ctrl[k] else "#2171b5"
                ax.plot([ctrl[k], real[k]], [y[k], y[k]], color="#bbb", lw=2, alpha=alpha, zorder=1)
                ax.scatter(ctrl[k], y[k], s=40, facecolor="white", edgecolor="#777", zorder=2, alpha=alpha)
                ax.scatter(real[k], y[k], s=45, color=color, zorder=3, alpha=alpha)
                ax.annotate(f"{real[k]:.0f} % (ctrl {ctrl[k]:.0f} %)", (max(real[k], ctrl[k]), y[k]), xytext=(6, 0),
                            textcoords="offset points", va="center", fontsize=7, alpha=alpha)
            if ref is not None and np.isfinite(ref.get(f"{col}_real", np.nan)):
                ax.axvline(ref[f"{col}_real"] * 100, color="#555", ls="--", lw=1)
            top = np.nanmax(np.r_[real, ctrl, ref[f"{col}_real"] * 100 if ref is not None else 0, 1.0])
            ax.set_xlim(0, top * 1.45)
            ax.set_xlabel(f"share of new errors that {label} (%)", fontsize=9)
            _clean(ax)
        axes[0].set_yticks(y, [f"{c}  ({n:,.0f} flips)" for c, n in zip(t.index, t["flips_real"])])
        axes[0].invert_yaxis()
        handles = [plt.Line2D([], [], marker="o", ls="", color="#de2d26", label="real mask (below control)"),
                   plt.Line2D([], [], marker="o", ls="", color="#2171b5", label="real mask (above control)"),
                   plt.Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor="#777",
                              label="random control"),
                   plt.Line2D([], [], ls="--", color="#555", label="model's own errors (no mask)")]
        fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, frameon=False)
        fig.suptitle(title or "Where do the new errors go?", fontsize=11)
        plt.tight_layout(rect=(0, 0.05, 1, 1))
        return fig

    def plot_probability_shift(self, summary, title=None):
        """Q3.4: control-corrected change of the probability on the true country, on its land neighbours and
        on its usual confusions. Input: summary(('d_true', 'd_nb', 'd_sim'), ...)."""
        s = summary.reset_index().sort_values("d_true_x")
        y = np.arange(len(s))
        parts = (("d_true", "true country", "#08519c", -0.27), ("d_nb", "its land neighbours", "#6baed6", 0),
                 ("d_sim", "its 5 usual confusions", "#fd8d3c", 0.27))
        fig, ax = plt.subplots(figsize=(9, 0.6 * len(s) + 1.8))
        for m, name, color, off in parts:
            x = s[f"{m}_x"].to_numpy(float) * 100
            err = np.vstack([x - s[f"{m}_lo"] * 100, s[f"{m}_hi"] * 100 - x])
            ax.barh(y + off, x, height=0.26, color=color, xerr=np.nan_to_num(err), capsize=2, label=name)
            hi_end = (s[f"{m}_hi"] * 100).fillna(pd.Series(x, index=s.index)).to_numpy()
            for xx, he, yy in zip(x, hi_end, y + off):
                ax.annotate(f"{xx:+.1f}", (0 if xx < 0 else he, yy), xytext=(4, 0), textcoords="offset points",
                            va="center", fontsize=6.5)
        lo_, hi_ = ax.get_xlim()
        ax.set_xlim(lo_, hi_ + 0.1 * (hi_ - lo_))
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(y, s["condition"])
        ax.invert_yaxis()
        ax.set_xlabel("change in probability vs random control (points)")
        ax.legend(fontsize=8, loc="lower left")
        _clean(ax)
        ax.set_title(title or "Where does the probability go?", fontsize=10)
        plt.tight_layout()
        return fig

    def plot_flip_pairs(self, conditions, top=12, title=None):
        """Q3.4: the country pairs that hiding an element creates (right -> wrong flips, sorted by real - control).
        Dot = real mask (colour: neighbour / usual confusion / neither), ring = random control."""
        conditions = [conditions] if isinstance(conditions, str) else list(conditions)
        kinds = {"land neighbour": "#1b9e77", "usual confusion": "#7570b3", "neither": "#d95f02"}
        fig, axes = plt.subplots(1, len(conditions), figsize=(6.8 * len(conditions), 0.38 * top + 2.0))
        for ax, cond in zip(np.atleast_1d(axes), conditions):
            t = self.ana.flip_destinations(cond, top=top, sort="excess")
            y = np.arange(len(t))
            kind = np.where(t["neighbour"], "land neighbour", np.where(t["usual_confusion"], "usual confusion", "neither"))
            ax.hlines(y, t["ctrl"], t["real"], color="#ccc", lw=2, zorder=1)
            ax.scatter(t["ctrl"], y, s=40, facecolor="white", edgecolor="#777", zorder=2)
            ax.scatter(t["real"], y, s=50, c=[kinds[k] for k in kind], zorder=3)
            for yy, r, c in zip(y, t["real"], t["ctrl"]):
                ax.annotate(f"{r:.0f} vs {c:.1f}", (max(r, c), yy), xytext=(6, 0), textcoords="offset points",
                            va="center", fontsize=7)
            ax.set_yticks(y, [f"{self._name(a)} → {self._name(b)}" for a, b in t.index], fontsize=8)
            ax.invert_yaxis()
            ax.set_xlim(0, max(t["real"].max() if len(t) else 0, 1) * 1.3)
            ax.set_xlabel("right -> wrong flips (control: mean over the random masks)", fontsize=9)
            ax.set_title(f"hiding {cond}", fontsize=10)
            _clean(ax)
        handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=f"real mask: {k}") for k, c in kinds.items()]
        handles.append(plt.Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor="#777",
                                  label="random control"))
        fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, frameon=False)
        fig.suptitle(title or "Which errors does hiding an element create?", fontsize=11)
        plt.tight_layout(rect=(0, 0.06, 1, 1))
        return fig

    def plot_strongest(self, table, top=20, title=None):
        """Q3.6: per country the cue whose hiding hurts most (from strongest_cue), with 95 % CI.
        Hollow marker: CI crosses 0 (no clear evidence). Labels show images / camera sequences."""
        t = table.head(top)
        group = t.index.name or "group"
        y = np.arange(len(t))
        fig, ax = plt.subplots(figsize=(10, 0.38 * len(t) + 2.0))
        for yy, (g, r) in zip(y, t.iterrows()):
            color = CUE_COLORS.get(r["condition"], "#333")
            x, lo, hi = r["x"] * 100, r["lo"] * 100, r["hi"] * 100
            ax.plot([lo, hi], [yy, yy], color=color, lw=2)
            ax.scatter(x, yy, s=55, zorder=3, facecolor=color if r["significant"] else "white", edgecolor=color, lw=1.8)
            ax.text(hi, yy, f"  {x:+.1f}", va="center", fontsize=8)
        names = [f"{self._name(g)} ({g})" if group == "country_id" else str(g) for g in t.index]
        ax.set_yticks(y, [f"{n}   {r['images']:.0f} img / {r['sequences']:.0f} seq" for n, (_, r) in zip(names, t.iterrows())],
                      fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel("change in P(true country) vs random control (points)")
        cues = list(dict.fromkeys(t["condition"]))
        handles = [plt.Line2D([], [], marker="o", ls="", color=CUE_COLORS.get(c, "#333"), label=c) for c in cues]
        handles.append(plt.Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor="#333",
                                  label="CI crosses 0"))
        ax.legend(handles=handles, fontsize=8, loc="lower right")
        ax.set_title(title or "Strongest cue per country", fontsize=10)
        _clean(ax)
        plt.tight_layout()
        return fig

    def plot_heatmap(self, profile, conditions=None, title=None, max_rows=80, sort_by=None, rows=None, mark_ns=True):
        """Country (or group) x condition heatmap of the control-corrected effect (points).
        rows: only these countries / groups. sort_by: a condition, 'strongest' (rows grouped by their
        strongest cue) or None (continent, then name). Hatched: fewer than min_n images.
        Small dot: 95 % CI crosses 0. Colours are clipped (see the colour bar)."""
        from matplotlib.patches import Rectangle
        group = profile.columns[0]
        p = profile if conditions is None else profile[profile["condition"].isin(conditions)]
        if rows is not None:
            p = p[p[group].isin(rows)]
        wide = p.pivot(index=group, columns="condition", values="x") * 100
        sig = p.pivot(index=group, columns="condition", values="significant")
        cols = [c for c in (conditions or self.ana.q3cfg.conditions) if c in wide.columns]
        wide = wide[cols]
        if sort_by == "strongest":
            filled = wide.fillna(np.inf)
            key = pd.DataFrame({"cue": filled.idxmin(axis=1).map({c: i for i, c in enumerate(cols)}),
                                "x": wide.min(axis=1)})
            wide = wide.loc[key.sort_values(["cue", "x"]).index]
        elif sort_by in wide:
            wide = wide.sort_values(sort_by)
        elif group == "country_id" and "continent" in self.ana.countries:
            key = pd.DataFrame({"c": self.ana.countries["continent"].reindex(wide.index),
                                "n": self.ana.countries.get("prompt_name", pd.Series(dtype=str)).reindex(wide.index)})
            wide = wide.loc[key.sort_values(["c", "n"]).index]
        if len(wide) > max_rows:
            print(f"plot_heatmap: showing {max_rows} of {len(wide)} rows (pass rows=... to choose them)")
        wide = wide.head(max_rows)
        sig = sig.reindex(index=wide.index, columns=cols)
        names = list(wide.index)
        if group == "country_id":
            names = [f"{self._name(c)} ({c})" for c in wide.index]
        vals = wide.to_numpy(float)
        lim = np.nanpercentile(np.abs(vals), 95) if np.isfinite(vals).any() else 1
        fig, ax = plt.subplots(figsize=(0.6 * len(cols) + 3.5, 0.27 * len(wide) + 2.4))
        cmap = plt.get_cmap("RdBu").copy()
        cmap.set_bad((0, 0, 0, 0))   # transparent: the hatched background shows through
        ax.add_patch(Rectangle((-0.5, -0.5), len(cols), len(wide), facecolor="#f4f4f4", edgecolor="#c8c8c8",
                               hatch="///", lw=0, zorder=-1))
        im = ax.imshow(np.ma.masked_invalid(vals), cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
        if mark_ns:
            ns = (~sig.fillna(True).astype(bool)).to_numpy() & np.isfinite(vals)
            r, c = np.nonzero(ns)
            ax.scatter(c, r, s=6, color="#333", marker="o", linewidths=0)
        ax.set_xticks(range(len(cols)), cols, rotation=60, ha="right")
        ax.set_yticks(range(len(wide)), names, fontsize=8 if len(wide) <= 45 else 7)
        ax.set_xlim(-0.5, len(cols) - 0.5)
        ax.set_ylim(len(wide) - 0.5, -0.5)
        fig.colorbar(im, ax=ax, extend="both", shrink=0.8 if len(wide) > 15 else 1,
                     label=f"change in P(true) vs random control (points)\nred = hiding it hurts; clipped at ±{lim:.0f}")
        note = f"hatched: fewer than {self.ana.q3cfg.min_n} images" + ("   ·   dot: 95 % CI crosses 0" if mark_ns else "")
        ax.set_xlabel(note, fontsize=8)
        ax.set_title(title or "Which cue matters where?", fontsize=10)
        plt.tight_layout()
        return fig

    def plot_interactions(self, table, title=None):
        """Per pair: the two single effects stacked vs the effect of hiding both (with CI of the difference)."""
        t = table.reset_index()
        fig, ax = plt.subplots(figsize=(9, 0.75 * len(t) + 1.6))
        y = np.arange(len(t))
        left = np.zeros(len(t))
        for i, col in ((1, "#9ecae1"), (2, "#fdae6b")):
            vals = t[f"effect_{i}"].to_numpy() * 100
            ax.barh(y - 0.2, vals, left=left, height=0.38, color=col,
                    label="single parts (stacked)" if i == 1 else None)
            for yy, l, v, name in zip(y, left, vals, t[f"part_{i}"]):
                ax.text(l + v / 2, yy - 0.2, name, ha="center", va="center", fontsize=7)
            left = left + vals
        whole = t["effect_whole"].to_numpy() * 100
        ax.barh(y + 0.2, whole, height=0.38, color="#3182bd", label="both hidden")
        for yy, s_, r in zip(y, left, t.itertuples()):
            ax.text(s_, yy - 0.2, f"  sum {s_:.1f}", va="center", fontsize=7)
        for yy, w, r in zip(y, whole, t.itertuples()):
            ax.text(w, yy + 0.2, f"  both {w:.1f}  |  extra {r.extra * 100:+.1f} [{r.extra_lo * 100:+.1f}, "
                                 f"{r.extra_hi * 100:+.1f}]", va="center", fontsize=7)
        ax.set_yticks(y, t["whole"] + "  (n=" + t["images"].map("{:,.0f}".format) + ")")
        ax.invert_yaxis()
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlim(0, max(left.max(), whole.max()) * 1.6)
        ax.set_xlabel("drop in P(true country) vs control (points; here > 0 = hurts)")
        ax.set_title(title or "Do cues compensate each other?\n(extra > 0: hiding both hurts more than the two alone)", fontsize=10)
        ax.legend(fontsize=8, loc="lower right")
        _clean(ax)
        plt.tight_layout()
        return fig

    def plot_environment(self, table, var, conditions=None, title=None, within=True, min_images=300):
        """Left: control-corrected change in P(true) per environment category (pooled, 95 % CI;
        < 0 = hiding it hurts). Right: within-country difference = category effect minus the country's own
        effect, averaged over countries (> 0 = hurts less in this category than in the same country overall);
        'n/a' = too few countries. Faded: fewer than min_images images in the category."""
        t = table.reset_index()
        if conditions is not None:
            t = t[t["condition"].isin(conditions)]
            conds = [c for c in conditions if c in set(t["condition"])]
        else:
            conds = list(dict.fromkeys(t["condition"]))
        cats = sorted(dict.fromkeys(t[var].astype(str)))
        t[var] = t[var].astype(str)
        palette = plt.get_cmap("tab10")
        ncol = 2 if within else 1
        fig, axes = plt.subplots(1, ncol, figsize=(7 * ncol, 0.42 * len(conds) * max(1.4, len(cats) / 1.6) + 2.0),
                                 sharey=True)
        axes = np.atleast_1d(axes)
        h = 0.8 / max(len(cats), 1)
        show_values = len(cats) <= 2
        for i, cat in enumerate(cats):
            s = t[t[var] == cat].set_index("condition").reindex(conds)
            yy = np.arange(len(conds)) + (i - (len(cats) - 1) / 2) * h
            color = palette(i)
            for k, (cond, r) in enumerate(s.iterrows()):
                if not np.isfinite(r["x"]):
                    continue
                alpha = 0.35 if r["images"] < min_images else 1.0
                x, lo, hi = r["x"] * 100, r["lo"] * 100, r["hi"] * 100
                axes[0].errorbar(x, yy[k], xerr=[[x - lo], [hi - x]], fmt="o", ms=5, color=color, alpha=alpha,
                                 capsize=2)
                if show_values:
                    axes[0].text(hi, yy[k], f" {x:+.1f}", va="center", fontsize=7, alpha=alpha)
                if within:
                    n = int(r["within_countries"]) if np.isfinite(r["within_countries"]) else 0
                    if np.isfinite(r["within_diff"]):
                        w, ci = r["within_diff"] * 100, r["within_ci"] * 100
                        axes[1].errorbar(w, yy[k], xerr=ci if np.isfinite(ci) else None, fmt="o", ms=5,
                                         color=color, alpha=alpha, capsize=2)
                        axes[1].text(w + (ci if np.isfinite(ci) else 0), yy[k], f" {w:+.1f} ({n} countries)",
                                     va="center", fontsize=7, alpha=alpha)
                    else:
                        axes[1].text(0, yy[k], f" n/a ({n} countries)", va="center", fontsize=7, color=color)
        handles = [plt.Line2D([], [], marker="o", ls="", color=palette(i), label=c) for i, c in enumerate(cats)]
        handles.append(plt.Line2D([], [], marker="o", ls="", color="#999", alpha=0.35,
                                  label=f"faded: < {min_images} images"))
        axes[0].set_yticks(np.arange(len(conds)), conds)
        axes[0].invert_yaxis()
        axes[0].set_xlabel("change in P(true country) vs random control (points; < 0 = hiding it hurts)", fontsize=9)
        axes[0].set_title("pooled: all images of the category", fontsize=10)
        if within:
            axes[1].set_xlabel("within-country difference (points; > 0 = hurts less here\n"
                               "than in the same country overall)", fontsize=9)
            axes[1].set_title("country mix removed", fontsize=10)
            lo1, hi1 = axes[1].get_xlim()
            axes[1].set_xlim(lo1, hi1 + (hi1 - lo1) * 0.35)
        for ax in axes:
            ax.axvline(0, color="k", lw=0.8)
            _clean(ax)
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8, frameon=False)
        fig.suptitle(title or f"Effect by {var}", fontsize=11)
        plt.tight_layout(rect=(0, 0.05, 1, 1))
        return fig

    # --- images ------------------------------------------------------------------

    def show_overlays(self, ids, elements=None, cols=4, size=3.2):
        """What the model sees, with the elements coloured (check the segmentation by eye)."""
        from shared.models import fit_image, fit_mask
        r = self.runner
        elements = elements or [e for e in COLORS if e in r.q3cfg.elements]
        fig, axes = plt.subplots((len(ids) + cols - 1) // cols, cols,
                                 figsize=(cols * size, ((len(ids) + cols - 1) // cols) * (size + 0.5) + 0.8))
        for ax in np.atleast_1d(axes).ravel():
            ax.axis("off")
        for ax, image_id in zip(np.atleast_1d(axes).ravel(), ids):
            image = r.dataset.get_image(image_id)
            masks = r.masks.load(image_id)
            shown = np.array(fit_image(image, r.cfg.crop_mode)).astype(float)
            found = []
            for e in elements:
                m = fit_mask(masks[e], r.cfg.crop_mode)
                if m.mean() > 0.001:
                    shown[m] = 0.45 * shown[m] + 0.55 * np.array(COLORS[e])
                    found.append(e)
            ax.imshow(shown.astype(np.uint8))
            country = r.dataset.meta.loc[str(image_id), "country_id"] if hasattr(r.dataset, "meta") and r.dataset.meta is not None else ""
            lines = [", ".join(found[k:k + 4]) for k in range(0, len(found), 4)]
            ax.set_title(f"{country} {image_id}\n" + "\n".join(lines), fontsize=7)
        handles = [plt.Rectangle((0, 0), 1, 1, facecolor=np.array(COLORS[e]) / 255, edgecolor="#555")
                   for e in elements]
        fig.legend(handles, elements, loc="lower center", ncol=min(len(elements), 6), fontsize=8)
        plt.tight_layout(rect=(0, 0.06, 1, 1))
        return fig

    def show_examples(self, condition, which="r2w", n=4, seed=None, size=3.0):
        """Random examples (not hand-picked): original | element hidden | random control, with guesses."""
        ids = self.ana.pick(condition, which, n, seed)
        if not ids:
            print(f"no images with {which} for {condition}")
            return None
        rows = self.ana.rows.set_index(["id", "condition", "kind", "rep"])
        fig, axes = plt.subplots(len(ids), 3, figsize=(3 * size, len(ids) * (size + 0.4) + 0.5))
        axes = np.atleast_2d(axes)
        for k, image_id in enumerate(ids):
            versions = self.runner.variant_images(image_id, [condition])
            want = [("original", "original", 0), (condition, "real", 0), (condition, "control", 0)]
            by_key = {(m["condition"], m["kind"], m["rep"]): img for m, img in versions}
            true = self.ana.sample.loc[image_id, "country_id"]
            for ax, key in zip(axes[k], want):
                ax.axis("off")
                if key not in by_key:
                    continue
                ax.imshow(by_key[key])
                r = rows.loc[(image_id, *key)]
                label = "original" if key[1] == "original" else ("hidden: " + condition if key[1] == "real" else "random control")
                ax.set_title(f"{label}\ntrue {true} | guess {r['pred']} ({r['p_max']:.0%}) | "
                             f"P(true) {r['p_true']:.0%}", fontsize=8)
        what = {"r2w": "right guesses that turn wrong", "w2r": "wrong guesses that turn right",
                "flip": "guesses that change", "any": "images"}.get(which, which)
        fig.suptitle(f"Random examples (not hand-picked): {what} when {condition} is hidden")
        plt.tight_layout()
        return fig
