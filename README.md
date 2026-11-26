# Can AI play GeoGuessr? Understanding StreetCLIP's predictions

We study how [StreetCLIP](https://huggingface.co/geolocal/StreetCLIP) predicts the country of a street-level photo,
using the [OpenStreetView-5M](https://huggingface.co/datasets/osv5m/osv5m) (OSV-5M) test set.

| Phase | Question | Approach | Status |
| --- | --- | --- | --- |
| Phase I | How does the model behave? | Black box: only inputs and outputs (maps, confusions, confidence, hiding image parts) | Finished (midpoint 11 Nov 2026) |
| **Phase II** | What did the model learn? | Open the box: image embeddings (kNN, PCA), Grad-CAM heatmaps | **Current** (deadline 9 Dec 2026) |

**Most of this README describes Phase I.** Questions, setup and rules still apply in Phase II unless stated otherwise.
Everything new for Phase II is in the section [Phase II](#phase-ii) at the end.

## Link to Phase 1 Presentation can be found [here](https://drive.google.com/file/d/1ZOneTfTiHEHJhox5lmwiccnPlwCOUIjy/view?usp=sharing).

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
| `pair_a` | Manuel Lisa (k12205828), Umut Tanriverdi (k12105807) | AI-aided | human-only |
| `pair_b` | Maximilian Hammer (k12222785), Florian Weigelt (k12210011) | human-only | AI-aided |

## Repository

```text
README.md
requirements.txt
shared/                 used by both pairs; changed only by agreement
  data.py               Config (all settings), CountryMapper, OSV5MDataset, MetadataDecoder
  models.py             GeoCLIPModel, PredictionRunner, Segmenter, TextDetector, Inpainter
  masking.py            MaskApplier
  add_country_attributes.py   adds region, neighbour, language and driving-side columns to country_aliases.csv
pair_a/                 pair A's own .py files (analysis, visualization)
  q2_confidence_phase2.py, q4_environment_phase2.py   Phase II files end with _phase2
pair_b/                 pair B's own .py files
resources/
  country_aliases.csv   country ids, prompt names, territories, candidate + training-list flags, country attributes
  country_sources/      saved source files for the country attributes (see "Country attributes")
notebooks/
  setup_and_test.ipynb  loads everything and checks every part (PASS / WARN / FAIL)
  phase1_AI.ipynb       Phase I analysis of the AI-aided pair
  phase1_Human_florian.ipynb, phase1_Human_max.ipynb   Phase I analysis of the human-only pair
  phase2_Human.ipynb    Phase II analysis of the human-only pair (pair A, one shared notebook)
ai-interactions/
  phase1_ai_log.md      AI interaction record for Phase I (required deliverable)
ai_reflection_midpoint.md   reflection on GenAI use in Phase I (required deliverable)
data/                   created by the setup notebook, not in git
results/                created by the notebooks, not in git
  shared/               shared baseline runs - read-only once written
  pair_a/               pair A's own results (Phase II in pair_a/phase2/)
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
3. `pip install -r requirements.txt`. An installed PyTorch is kept, so check that its version fits (`torch>=2.7`).
   Check: `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`, where `True` means the GPU is used.
4. Open `notebooks/setup_and_test.ipynb`, adjust the settings in the first code cell if needed, and run all cells.
   Every check should print PASS (WARN is a hint, FAIL needs fixing).
5. **Precompute the predictions once:** set `RUN_STREETCLIP_MAIN = True` in the last section and run it (under an hour
   on an RTX 5070). Using the settings in `Config`, it computes StreetCLIP's prediction for every test image and
   saves them in `results/shared/` (e.g. `streetclip_v1_center_crop/`). Each person runs this on their own machine;
   results are not uploaded to GitHub.
6. **PNG copies of Plotly figures** (so they show on GitHub): kaleido needs Chrome. Run `plotly_get_chrome -y` once.
   On Linux / WSL, Chrome also needs `sudo apt install libnss3`. Without it, the figures still show in Jupyter.

Downloads: OSV-5M `test.csv` 116 MB + 5 zips 9.43 GB (extraction briefly needs ~19 GB free), StreetCLIP ~1.7 GB,
Mask2Former ~1 GB, EasyOCR ~0.1 GB; optional CLIP ~1.7 GB and LaMa ~0.2 GB. Everything is downloaded only once.

## Settings

All settings and their defaults are in `Config` (`shared/data.py`) and are changed in the notebook's first cell.
Every run saves the settings it used (`run_info.json` and columns in `predictions.parquet`).

| Setting | Default | Meaning |
| --- | --- | --- |
| `images_per_country` | 50 | Pilot sample size per country (pair B's Q2 and Q3, quick checks) |
| `use_clip_baseline` | False | Original CLIP as baseline for Q2; downloaded on first use |
| `crop_mode` | `center_crop` | `center_crop` (CLIP standard) / `pad` (robustness check) / `squash` |
| `candidate_list` | `v1` | `v1` / `all_test_locations` (sensitivity run) |
| `prompt_template` | `A Street View photo in {country}, {subregion}.` | Paper template plus the subregion (since 6 Nov, fixes "Georgia" = US state); old runs kept as `*_oldprompt` |
| `fill_mode` | `gray` | `gray` / `blur` / `inpaint` (LaMa) |
| `mask_dilation_px` | 5 | Mask grown by a few pixels |
| `min_mask_area` | 2 %, text and signs 0.1 % | Share of the visible model input an element must cover |
| `num_random_controls` | 3 | Random same-size masks per real mask |

## Fixed decisions (Phase I)

- **Data:** OSV-5M test split only (~210k Mapillary images). The train split (~250 GB) is not used.
- **Main run:** StreetCLIP on the full test set, all candidate scores (logits) and image embeddings saved.
  The 50-per-country pilot (`resources/pilot_ids.csv`) is used by pair B: for the CLIP comparison in Q2 and,
  with 20 images per country, for Q3. Pair A's Q3 uses its own sample: at most 60 images per country and 150 for
  the 20 most confused countries (12,154 images, `results/pair_a/q3/sample.csv`).
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

## Country attributes

`country_aliases.csv` also holds country-level attributes for grouping and colouring results (e.g. by region,
neighbours or driving side). **The CSV in git already contains these columns, so nobody needs to run the script.**
`shared/add_country_attributes.py` records how they were made and can rebuild them.

| Column | Source | Meaning |
| --- | --- | --- |
| `world_region`, `world_subregion` | mledoze/countries | Region and subregion, e.g. `Europe` / `Central Europe` |
| `center_lat`, `center_lng` | mledoze/countries | Approximate centre point of the country (for maps) |
| `landlocked` | mledoze/countries | 1 = no sea coast |
| `continent` | GeoNames | `AF`, `AN`, `AS`, `EU`, `NA`, `OC`, `SA` (`NA` = North America) |
| `neighbors` | GeoNames | Land neighbours as ids separated by `;` (empty for islands) |
| `languages` | GeoNames | Language codes separated by `;`, regional variants merged (`en-IN` -> `en`) |
| `main_language` | GeoNames | First entry of `languages`. GeoNames' order, **not necessarily the most spoken** (India -> `en`) |
| `driving_side` | Wikidata (P1622) | `left` / `right` per country |

- **Snapshot:** retrieved on 2 Nov 2026, mledoze pinned to commit `6031f5d`. The raw files are in
  `resources/country_sources/`; `sources.json` lists URLs, retrieval time, licences and manual overrides.
- **Manual fix:** Argentina (`AR`) is set to `right`. Wikidata lists both sides because Argentina switched in 1945.
  Overrides live in `DRIVING_SIDE_OVERRIDES` in the script, each with a source.
- **Empty values:** `driving_side` is empty for Antarctica, Bouvet Island and Heard and McDonald Islands
  (uninhabited, not candidates).
- **Cross-check:** every country in `DRIVE_SIDE_FIXES` (`shared/data.py`) is `left` in Wikidata.
- **Not covered:** writing system (Latin, Cyrillic, ...); none of the sources provides it.
- **Reading the CSV:** use `pd.read_csv(path, keep_default_na=False)` as `CountryMapper` does, otherwise pandas
  turns `NA` (Namibia, North America) into a missing value.
- **Rebuilding:** `python -m shared.add_country_attributes --offline` rebuilds from the saved files (same result).
  Without `--offline` it downloads again; GeoNames and Wikidata change over time, so values may differ. The original
  columns are never changed, and running it again is safe.
- **Licences:** mledoze/countries ODbL 1.0, GeoNames CC BY 4.0, Wikidata CC0.

## Metadata codes

`MetadataDecoder` uses the sources OSV-5M names: Köppen-Geiger climate classes (Beck et al. 2018, codes 0-30) and
FAO GLC-SHARE land cover (11 classes). OSV-5M does not document its exact numbering, so the setup notebook compares
the codes in the data with these tables. `drive_side` is decoded from the data itself (the value used by
left-driving countries means "left"). This is OSV-5M's per-image value; the country-level `driving_side` from
Wikidata is in `country_aliases.csv` (see "Country attributes"). `road_index` and `soil` are left numeric.

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
infrastructure only and contains no analysis or visualization. `shared/add_country_attributes.py` was written by
pair B with AI assistance before its human-only analysis started; it only collects country attributes from public
sources. This is recorded in `ai-interactions/phase1_ai_log.md` (row #38).

## Phase II

**Question:** What did the model learn? We open the box and look at the image embeddings and at gradient heatmaps.

- Each pair uses at least two techniques and compares or combines them.
- Hypotheses are tested across groups (correct vs. wrong, built-up vs. rural, ...), not on single images.
- The questions continue from the Phase I findings.
- Roles are swapped: pair A is human-only, pair B is AI-aided (see "Pairs").
- Notebooks: `phase2_Human.ipynb` (pair A, one shared notebook), `phase2_AI_max.ipynb` (pair B, Q2 and Q3) and
  `phase2_AI_florian.ipynb` (pair B, Q1 and Q4).
  New `.py` files end with `_phase2` (e.g. `pair_a/q2_confidence_phase2.py`).
- The Phase I rule "black box only" no longer applies. The reporting rules still do.

### Pair A: questions and techniques

| ID | Phase II question | Technique |
| --- | --- | --- |
| Q1 | (to be added) | |
| Q2.1 | Where do confident errors lie in the embedding space? | kNN, PCA (US and Canada) |
| Q2.2 | Training coverage: are list countries better separated after fine-tuning (CLIP vs. StreetCLIP)? | kNN on both models |
| Q2.3 | Where does the model look in confident US -> Canada errors? | Grad-CAM + Mask2Former segments |
| Q3 | (to be added) | |
| Q4.1 | Are built-up images more typical for their country? | kNN, world maps |
| Q4.2 | Do climate and land cover move the embeddings? | kNN, within-country difference |
| Q4.3 | Where does the model look in built-up vs. rural images? | Grad-CAM + Mask2Former + EasyOCR text |

### Settings (pair A, Q2 and Q4)

The values are constants or function arguments in `pair_a/q2_confidence_phase2.py` and `pair_a/q4_environment_phase2.py`.

| Setting | Value | Meaning |
| --- | --- | --- |
| `K` | 10 | Number of nearest neighbours (cosine similarity of the cached image embeddings) |
| `min_km` | 50 | Neighbours closer than this, or from the same camera sequence, are skipped |
| References | max. 300 per country | Images the neighbours come from (`random_state=1`) |
| Usable references | min. 10 | Queries with fewer own-country references are dropped (`keep_reachable`) |
| Queries | 50 per country and group (Q2.1, Q4.2), 100 per country (Q2.2, Q4.1) | Images that get a kNN vote (`random_state=42`) |
| Confident | top-1 probability >= 0.8 | Same as Phase I (`CONF_HIGH`) |
| `MIN_CELL` | 20 | A country needs this many images in a category for the within-country comparison |
| Grad-CAM layer | `layer_norm1` of the last vision layer | Standard choice for ViT models |
| Grad-CAM target | predicted country (Q2.3), true country (Q4.3) | Class the heatmap explains |
| Grad-CAM precision | float32, batch 4 (2 for mean maps) | Peak GPU memory about 12 GB with batch 4, 7.5 GB with batch 2 |

### Fixed decisions (Phase II, pair A)

- **Same place is not the same country:** without the 50 km rule, kNN finds other photos of the same street.
  Macro kNN accuracy then drops from 64 % to 36 %.
- **Balanced references:** with all images as references, big countries win the vote by size.
- **Small countries:** with the 50 km rule, 23 countries (e.g. Bermuda, Tonga, Kuwait) have no usable references
  and are dropped. kNN is still hard on small countries, so a low kNN value there is weak evidence.
- **Grad-CAM per element:** the heatmap is split by the Mask2Former segments after the center crop, so it matches
  what the model sees. "Extra heat" = heat share minus area share.
- **Only groups:** single confused pairs change with the reference set, so they are not reported as results.
- **Caches:** Grad-CAM results are saved in `results/pair_a/phase2/q2/` and `q4/` (`.parquet`, `.npz`).
  Delete them to recompute (about 2 to 3 minutes each on an RTX 4070 Ti SUPER).

### Running the Phase II notebook

1. `grad-cam` (pytorch-grad-cam) is in `requirements.txt` and `pyproject.toml`. Install again or run `uv sync`.
2. Q2.2 needs the original CLIP run (`results/shared/clip_v1_center_crop/`, see `use_clip_baseline`).
3. Close other notebook kernels that use the GPU. On WSL, a full GPU spills into normal memory and Grad-CAM
   becomes very slow.

### Pair B: questions and techniques

Pair B is AI-aided in Phase II; the AI use is recorded in `ai-interactions/phase2_ai_log.md`.
All pair B runs use the paper's prompt (`A Street View photo in {country}.`), the same as pair B's Phase I runs.

| ID | Phase II question | Technique |
| --- | --- | --- |
| Q1 | (to be added) | |
| Q2.1 | Where in the US do images "look like Georgia", and does the image encoder know US states? | Cached logits on a map (every US image), state prompts |
| Q2.2 | Image side vs. text side: do Georgia photos look like the Caucasus while the word "Georgia" looks like the US? | Image embeddings (country centroids) + text embeddings |
| Q2.3 | Which words free "Georgia" from the US state? | Prompt variants rescored on cached image embeddings |
| Q3.1 | Does the model look where hiding hurts? Grad-CAM vs. the Phase I masking on the same 3,660 images and segments | Grad-CAM + masking (Phase I) |
| Q3.2 | Same image, two explanations: where do the two techniques agree and disagree? | Grad-CAM + masking examples |
| Q3.3 | Heat budget: where does the attention go in confident errors vs. confident correct images? | Grad-CAM + segments |
| Q3.4 | Which scenes help or hurt inside a country? | Concept probing with the text encoder (AI-suggested concepts), within-country comparison |
| Q4 | (to be added) | |

Grad-CAM (pair B): last vision layer (`layer_norm1`), float32, target = true country, heat per element from the
cached Mask2Former segments and EasyOCR text masks. Results in `results/pair_b/phase2/`.
