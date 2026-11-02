# Can AI play GeoGuessr? Understanding StreetCLIP's predictions

We study how [StreetCLIP](https://huggingface.co/geolocal/StreetCLIP) predicts the country of a street-level photo,
using the [OpenStreetView-5M](https://huggingface.co/datasets/osv5m/osv5m) (OSV-5M) test set.

| Phase | Question | Approach | Status |
| --- | --- | --- | --- |
| **Phase I** | How does the model behave? | Black box: only inputs and outputs (maps, confusions, confidence, hiding image parts) | **Current** |
| Phase II | What did the model learn? | Open the box: gradient heatmaps, image embeddings | Later |

**This README describes Phase I.** Everything below (questions, setup, rules) applies to Phase I unless stated otherwise.

## Phase I questions

**Main question:** What makes StreetCLIP succeed or fail at guessing the country, and which visual clues does it rely on?

| ID | Question |
| --- | --- |
| Q1 | Error structure and geography: which countries are confused, what do confused pairs share, are there attractor countries, do errors cluster geographically? |
| Q2 | Overconfidence and training coverage: when is the model confidently wrong, and can we infer which countries were in its Street View training? |
| Q3 | Visual clues: which elements (text, road, vegetation, sky, ...) change the prediction when hidden, and does this differ between countries? |
| Q4 | Environment (secondary): do accuracy and confidence vary with climate, land cover, distance to the sea, or urban vs. rural setting? |

## Pairs

The pairs swap roles after Phase I, so code folders are named after the pair, not the role.

| Pair | Members | Phase I | Phase II |
| --- | --- | --- | --- |
| `pair_a` | *Manuel Lisa - k12205828; Umut  - kxxx* | *AI-aided* | *the other role* |
| `pair_b` | *Maximilian Hammer - kxxx; Florian Weigelt - kxxx* | *human-only* | *the other role* |

## Repository

```text
README.md
requirements.txt
shared/                 used by both pairs; changed only by agreement
  data.py               Config (all settings), CountryMapper, OSV5MDataset, MetadataDecoder
  models.py             GeoCLIPModel, PredictionRunner, Segmenter, TextDetector, Inpainter
  masking.py            MaskApplier
pair_a/                 pair A's own .py files (analysis, visualization)
pair_b/                 pair B's own .py files
resources/
  country_aliases.csv   country ids, prompt names, territories, candidate + training-list flags
notebooks/
  setup_and_test.ipynb  loads everything and checks every part (PASS / WARN / FAIL)
  phase1_AI.ipynb       Phase I analysis of the AI-aided pair
  phase1_Human.ipynb    Phase I analysis of the human-only pair
logs/
  phase1_ai_log.md      AI interaction record for Phase I (required deliverable)
data/                   created by the setup notebook, not in git
results/                created by the notebooks, not in git
  shared/               shared baseline runs - read-only once written
  pair_a/               pair A's own results
  pair_b/               pair B's own results
```

`shared/` only covers data, models and masking. **Analysis and visualization are built separately by each pair**,
so the two approaches can differ.

## Setup

1. **Python:** we use Python 3.11 (3.10-3.12 should also work).
2. **PyTorch first, matching your computer.** Run `nvidia-smi` and look at "CUDA Version" in the top right:
   - 12.8 or higher (e.g. RTX 50-series): nothing to do, `requirements.txt` installs the CUDA 12.8 build.
   - lower: update the NVIDIA driver, or install the matching build from [pytorch.org](https://pytorch.org/get-started/locally/) first.
   - no NVIDIA GPU: install the CPU build from [pytorch.org](https://pytorch.org/get-started/locally/) first.
     Analysing the shared results works on CPU; segmentation and masking are very slow without a GPU.
3. `pip install -r requirements.txt` (an already installed PyTorch is kept - see if the pytorch req. is compatible).
   Check: `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`, where `True` means the GPU is used.
4. Open `notebooks/setup_and_test.ipynb`, adjust the settings in the first code cell if needed, and run all cells.
   Every check should print PASS (WARN is a hint, FAIL needs fixing).
5. **Precompute the predictions once:** set `RUN_STREETCLIP_MAIN = True` in the last section and run it (under an hour
   on an RTX 5070). Using the settings in `Config`, it computes StreetCLIP's prediction for every test image and
   saves them in `results/shared/` (e.g. `streetclip_v1_center_crop/`). Each person runs this on their own machine;
   results are not uploaded to GitHub.

Downloads: OSV-5M `test.csv` 116 MB + 5 zips 9.43 GB (extraction briefly needs ~19 GB free), StreetCLIP ~1.7 GB,
Mask2Former ~1 GB, EasyOCR ~0.1 GB; optional CLIP ~1.7 GB and LaMa ~0.2 GB. Everything is downloaded only once.

## Settings

All settings and their defaults are in `Config` (`shared/data.py`) and are changed in the notebook's first cell.
Every run saves the settings it used (`run_info.json` and columns in `predictions.parquet`).

| Setting | Default | Meaning |
| --- | --- | --- |
| `images_per_country` | 50 | Pilot sample size per country (Q3, quick checks) |
| `use_clip_baseline` | False | Original CLIP as baseline for Q2; downloaded on first use |
| `crop_mode` | `center_crop` | `center_crop` (CLIP standard) / `pad` (robustness check) / `squash` |
| `candidate_list` | `v1` | `v1` / `all_test_locations` (sensitivity run) |
| `prompt_template` | `A Street View photo in {country}.` | Template from the StreetCLIP paper |
| `fill_mode` | `gray` | `gray` / `blur` / `inpaint` (LaMa) |
| `mask_dilation_px` | 5 | Mask grown by a few pixels |
| `min_mask_area` | 2 %, text and signs 0.1 % | Share of the visible model input an element must cover |
| `num_random_controls` | 3 | Random same-size masks per real mask |

## Fixed decisions (Phase I)

- **Data:** OSV-5M test split only (~210k Mapillary images). The train split (~250 GB) is not used.
- **Main run:** StreetCLIP on the full test set, all candidate scores (logits) and image embeddings saved.
  The 50-per-country pilot is used for Q3 and the `pad` robustness check.
- **Country ids:** one canonical id per country or territory, its ISO alpha-2 code. Predictions, accuracy and maps
  all use it; the prompt name is only the text given to the model.
- **United States:** one label (no state-level split).
- **Not representable:** images whose country is not a candidate get `representable = False` and `correct = NA`;
  they are excluded from accuracy, not counted as errors.
- **Segmentation:** Mask2Former trained on Mapillary Vistas (same image source). **Text:** EasyOCR detection only,
  no recognition-confidence filter (it would drop non-Latin text).
- **Masks are judged by what the model sees:** the visible fraction is measured after `crop_mode`, so a sign that
  the center crop cuts off does not count as hidden.

## Country labels and candidate lists

Source: StreetCLIP paper ([arXiv:2302.00275](https://arxiv.org/abs/2302.00275), Appendix B.1 and B.3.1), taken from the
[ar5iv HTML version](https://ar5iv.labs.arxiv.org/html/2302.00275). `country_aliases.csv` was built from the ISO 3166-1
list plus the paper's two lists.

- **Candidate list v1 (197):** the paper's 193 probing countries + Bermuda, Guam, Monaco and Puerto Rico, which are in
  the training list but not the probing list.
- **all_test_locations:** v1 + every location that occurs in the test set. Used as a sensitivity run, because adding
  only training-list territories favours exactly what Q2 tests.
- **Training list:** stored as published (`in_paper_training_list`). Treat it as uncertain evidence: the paper says
  101 countries but lists 92, and the model card says India and China were excluded while the list includes them.
- **Territories** keep their own id (e.g. Puerto Rico = `PR`, both as ground truth and as prediction).
  `sovereign_parent` (e.g. `PR -> US`) allows a lenient comparison if a pair wants one.

Normalizations made:

| Paper spelling | Id | Note |
| --- | --- | --- |
| Macedonia (training list) | MK | Same country as "North Macedonia" in the probing list |
| Swaziland | SZ | Former name of Eswatini; kept as prompt text, as in the paper |
| Czech Republic, Turkey, East Timor, Brunei, Russia, ... | CZ, TR, TL, BN, RU | Paper spelling kept as prompt text |
| Côte d’Ivoire | CI | Curly apostrophe kept as in the paper |
| Kosovo | XK | Not in ISO 3166-1; added in case OSV-5M uses it (not a v1 candidate) |

The country-label format of OSV-5M's `test.csv` is checked in the setup notebook; unknown labels are listed there.

## Metadata codes

`MetadataDecoder` uses the sources OSV-5M names: Köppen-Geiger climate classes (Beck et al. 2018, codes 0-30) and
FAO GLC-SHARE land cover (11 classes). OSV-5M does not document its exact numbering, so the setup notebook compares
the codes in the data with these tables. `drive_side` is decoded from the data itself (the value used by
left-driving countries means "left"). `road_index` and `soil` are left numeric.

Note for Q2: OSV-5M already removed blurry, dark, badly exposed and rotated images, so image-quality categories
(night, blur) are exploratory.

## Shared results and pair results

- `results/shared/` holds the baseline runs. `PredictionRunner` refuses to overwrite an existing run folder.
- Each pair writes its own experiments (masking runs, re-scored variants, figures) to `results/pair_a/` or
  `results/pair_b/`.
- `results/` is not in git (logits and embeddings are hundreds of MB). If the pairs work on different machines,
  copy `results/shared/` or re-create it with the same settings.

## Agreed reporting rules (implemented by each pair)

- Per-country results always come with the number of images (n); flag low-sample countries.
- Report uncertainty (e.g. confidence intervals) for per-country rates.
- Summarise with macro-averaged accuracy (every country counts equally); pooled accuracy only as a supplement.
- Confusion pairs: show both the number of errors and their share of the origin country's errors.
- For masking (Q3), report separately: change in probability of the originally predicted country, change in
  probability of the true country, and top-1 flips (including wrong -> right).

## Phase I rule: black box only

The cached image embeddings exist only so that candidate lists and prompts can be changed without re-running all
images. Neither pair analyses embeddings (similarity, clustering, projections) in Phase I; that belongs to Phase II.

## AI use

The AI-aided pair built the shared setup (`shared/`, country table, setup notebook) with AI assistance. It is
infrastructure only and contains no analysis or visualization. This is recorded in `logs/phase1_ai_log.md`; its
compatibility with the human-only condition is to be confirmed with the course staff.
