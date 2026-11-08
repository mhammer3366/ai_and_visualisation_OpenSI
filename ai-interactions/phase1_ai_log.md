# AI interaction record - Phase I (AI-aided pair)

Each row is one AI interaction that changed our analysis or shows a strength or limit of AI assistance.

**Decision:** accepted / modified / rejected

**Prompt:** rows #1 to #30 (before 6 Nov) use the paper's prompt ("A Street View photo in {country}."). Their
numbers can differ from the notebook, which uses the new prompt with the subregion (#33).

## Log

| # | Date | Topic / prompt (short) | AI suggestion | Decision | Why | Effect on the work |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2026-11-01 | "Give me an overview of Phase I and split it in two" | Two tracks: errors (A) and clues/masking (B), day plan to Sat 7.11 | Accepted | Clear split, fits deadline | Basis of our task division |
| 2 | 2026-11-01 | "Can we just let AI do the work and watch?" | Direct and verify instead; log as you go | Accepted | Log is graded; we swap roles in Phase II | We review every AI output before using it |
| 3 | 2026-11-01 | Main research questions | Main question + 4 sub-questions | Modified | We merged overconfidence and training coverage; added geographic patterns | Final Q1-Q4 |
| 4 | 2026-11-01 | How to infer training countries | Compare StreetCLIP with original CLIP per country | Accepted after questioning | Accuracy alone mixes "easy" with "trained on" | Optional CLIP baseline in setup |
| 5 | 2026-11-01 | Top up small countries with train images | Offered a train-split switch | Rejected | Train split is ~250 GB | Test split only |
| 6 | 2026-11-01 | Setup structure | 18 files, config.yaml, pytest, analysis/viz stubs | Modified | Too complex; stubs would steer both pairs | 3 source files, settings in notebook, no stubs |
| 7 | 2026-11-01 | Candidate countries | Default: countries in our sample | Rejected (by external review) | Candidate list would change with the sample | Fixed list v1 + sensitivity run |
| 8 | 2026-11-01 | OCR settings | Filter text boxes by recognition confidence 0.3 | Rejected (by external review) | Would silently drop non-Latin text | Detection only |
| 9 | 2026-11-01 | Paper country list | AI reported typos/duplicates in the paper's list | Rejected | They came from a broken PDF-to-text conversion, not the paper | List taken from the HTML version |
| 10 | 2026-11-01 | Dataset size | AI estimated ~10 GB | Corrected | Repository lists 9.43 GB + 116 MB | Sizes in README |
| 11 | 2026-11-03 | "Can we run the setup on my Mac?" | AI checked `uv.lock`: no macOS PyTorch builds, and `get_device` never used the Mac GPU. Proposed a Mac path (PyTorch from PyPI on macOS, `mps` device) | Accepted | Umut only has a Mac (no NVIDIA GPU) | Setup also runs on Mac. Tested: same top-1 as CPU, ~20 images/s with fp16 (full run ~3 h). Our own test script first crashed on macOS (missing `__main__` guard), AI found and fixed it. Kept on branch `mac-setup`, not merged: we later moved all work to the homeserver (row 12) |
| 12 | 2026-11-04 | "Set up my homeserver (RTX 4070 Ti Super, WSL2) and run the baselines" | AI found that only PyPI downloads were slow on the server (0.2 MB/s, other sites 35+ MB/s). It took 9 CUDA packages from the PyTorch mirror and checked each file against the SHA-256 in `uv.lock` | Accepted | `uv sync` was stuck for 30 min | Server ready in about 1 h; StreetCLIP and CLIP runs take about 23 min each |
| 13 | 2026-11-04 | "Start Q2 (overconfidence, training coverage)" | Three parts: calibration of StreetCLIP vs original CLIP, per-country gap (mean confidence - accuracy), and accuracy gain CLIP -> StreetCLIP compared with the paper's training list | Accepted | Uses only the prediction tables (black box); CLIP is the same model before the Street View fine-tuning | Code in `pair_a/q2_confidence.py`, Q2 section in `phase1_AI.ipynb` |
| 14 | 2026-11-04 | Q2 first results | AI noticed that "Georgia" is never right: 0 of 290 real Georgia images, but 2,436 predictions. Most come from the US state Georgia and nearby southern states | Accepted after check | Region column: 401 of 451 US images from the state Georgia are predicted as Georgia. The prompt name means both a country and a US state | Biggest single source of confident errors (US -> Georgia: 1126 images with probability >= 0.8) |
| 15 | 2026-11-04 | Precision per predicted country | Namibia had precision 0 with 909 predictions. AI first suspected the known "NA" read-as-missing bug | Rejected after check | The raw `test.csv` has no Namibia images at all | Added `n_true` column: countries without test images are never right by design |
| 16 | 2026-11-04 | "Start Q4 (environment)" | AI showed that environment and country are mixed up (about half of "sparse vegetation" is Australia). Proposed to show every effect twice: raw, and within-country (category accuracy minus its own country's accuracy) | Accepted | A plain table per climate would mostly show which countries live in that climate | Climate effect almost gone inside countries; built-up areas +6.7 points |
| 17 | 2026-11-04 | Urban vs. rural proxy | AI checked `road_index` as an urban measure | Rejected | It has only one value per country (no spread inside a country) | Urban proxy is land cover "artificial surfaces" (built-up) |
| 18 | 2026-11-04 | Urban map | First map showed Libya (+68 points) and Afghanistan (+62) as the strongest urban effects. AI checked the camera sequences | Modified | Libya's 54 urban images come from only 7 sequences | Map only shows countries with >= 10 urban sequences; effect also checked with one vote per sequence (+8.3 points) |
| 19 | 2026-11-04 | First Q4 plot | AI's own plot showed within-country values for mangroves (-24 points) and snow, based on one country and without error bar | Corrected | Found only by looking at the rendered plot, not the table | Within-country values need >= 5 countries |
| 20 | 2026-11-04 | "Be honest, can we make the explainability better?" | AI said the tables were raw data dumps and the plots did not state their message. Proposed: readable tables, an "expected from country mix vs. actual" plot, a "who is in each category" bar chart, titles that say the finding, random example images | Accepted | We have to present and defend the results | New `pair_a/viz.py`, Georgia map, gain box plot, Q2 and Q4 findings cells |
| 21 | 2026-11-04 | Titles that state the finding | AI's first title said "One pair dominates: US -> Georgia" | Corrected | The rendered bar chart showed US -> Canada almost as big (995 vs. 1126) | Title now: two US pairs are 40 % of all confident errors |
| 22 | 2026-11-04 | Distance to the sea | AI's earlier Q4 finding was "no effect". The new expected-vs-actual plot showed the coast 4.4 points worse (US coast 38 % vs. US overall 63 %) | Corrected | The two measures weight countries differently: per image the coast is worse, per country the effect is small | Findings updated. The first plot also zoomed the axis to 47-55 %, now at least 20 points wide |
| 23 | 2026-11-04 | Training coverage | AI checked the original CLIP per group: list countries 41.5 % vs. 28.1 % already before fine-tuning | Accepted | Shows that most of the accuracy gap (13 of 18 points) is not from the Street View training | Q2 finding: the gain does not clearly confirm the paper's list |
| 24 | 2026-11-04 | "Is the Georgia error our problem or in the pretrained model?" | AI checked the prompt ("A Street View photo in Georgia.", from the paper) and compared with the original CLIP: it already sends 1,962 US images to Georgia, StreetCLIP 2,346 (634 vs. 1,126 with >= 80 % confidence) | Accepted | Same prompt for both models, so the error comes from pretraining and fine-tuning makes it stronger | New CLIP vs. StreetCLIP table in the Georgia part, Q2 finding updated |
| 25 | 2026-11-04 | "Analyse my plan for Q1" (difficulty, confusions, shared attributes, attractors, geography) | Plan is sound. Added: Wilson CI and a minimum n per country, keep direction (A -> B is not B -> A), prediction ratio for attractors, a within-continent baseline, two meanings of "cluster" (errors land near the truth vs. hard countries border each other, Moran's I), km distance for islands | Modified | Additions handle small samples and confounding (continent also decides driving side and language). We kept km distance only if the data has coordinates | Final Q1 analysis plan |
| 26 | 2026-11-04 | How to show confusions | Top-N pair table plus each country's top 3 confusions, one heatmap of the ~20-30 most confused countries instead of one matrix per continent. Top-k only as extra evidence, not repeated for every table | Accepted | A Europe matrix is still too big; top-k answers "what is plausible", not "what is predicted" | Confusion tables in Q1 |
| 27 | 2026-11-04 | "Attractor = big training class?" | AI asked for training image counts per country | Not possible | StreetCLIP's training counts are not public | Listed as a Q1 limitation |
| 28 | 2026-11-04 | Code structure for Q1 (our design: builder -> analyser -> reporter, keep the image-level table for GPS) | One data bundle from the builder, one config (min n, k values, seed, permutations), fixed seed, no printing in the analyser | Accepted | Same structure for all parts, results reproducible | `q1_errorstructure.py` with `Q1Config`, `Q1DataBuilder`, `Q1Analyser`, `Q1Reporter` |
| 29 | 2026-11-05 | Q1 code | Baseline: replace the predicted country of each error by a random country (all countries, or same continent), report lift with bootstrap CI and permutation p-value. Tested on made-up data with broken rows; this found two bugs in its own code (one short top-k list switched off top-3/5 for all rows; p-value shown when there was no data) and AI fixed them | Accepted | Our column names were not known yet | Later run on the full test set (208,314 images, 1,000 permutations), see row 31 |
| 30 | 2026-11-05 | "What is still open in Q2 and Q4?" | AI found that the CLIP vs. StreetCLIP Georgia table was never run: the findings quoted its numbers, but the notebook had no output for it. It also saw that the low-precision table had no finding | Accepted | Every number in the findings must have a visible output we can show | Cell run, all numbers matched. New finding on labels that are never right, and an open questions line for Q2 |
| 31 | 2026-11-06 | "Run the Q1 analysis on our real predictions and tell us what stands out" | Errors go to neighbours 13× more often than random; language 1.9×, driving side 1.3× | Accepted | Random baseline shows what is more than chance | Q1 finding: mostly geography |
| 32 | 2026-11-06 | "Build an interactive view so we can explore Q1 by country" | Dash dashboard with 5 views and world maps | Accepted | 175 countries do not fit in static plots | `Q1Dashboard`. Does not show on GitHub, so the findings cell is the main result |
| 33 | 2026-11-06 | "Georgia is read as the US state. How can we fix the prompt?" | Add the subregion: "A Street View photo in Georgia, Western Asia." | Accepted, agreed with pair B | US to Georgia errors drop from 2,346 to 62 | Shared prompt changed for both pairs. Cost: 1.7 points accuracy. Real Georgia still not recognised |
| 34 | 2026-11-06 | "How can we measure if hiding an element really matters, and not just that less image is left?" | Compare each mask with random masks of the same size | Accepted | Any large mask lowers accuracy, just because less image is left | All Q3 effects are control-corrected |
| 35 | 2026-11-06 | "Masking the full test set takes too long. How should we choose the images?" | Max 60 images per country, 150 for the 20 most confused; cache masks instead of saving images | Accepted | Full test set too slow; saved images ~20 GB | Q3 runs in hours, not days |
| 36 | 2026-11-06 | "Road and vegetation are very big masks. Can we see which part of them matters?" | Road into surface and lane markings, etc.; test two cues hidden together | Accepted | A whole element can hide which part matters | Lane markings matter more than road surface |
| 37 | 2026-11-08 | "Is the AI-aided part finished?" (with task description) | Missing: Q1 findings, reflection, log rows for Q3, `ai-interactions/` folder, README IDs | Accepted | Missing parts lose points | AI drafted these; we checked and rewrote them |
| 38 | 2026-11-02 | Pair B (before its human-only work): country attributes for the shared country table. Prompt not saved; added here later | Script `shared/add_country_attributes.py` that collects region, neighbours, languages and driving side from mledoze/countries, GeoNames and Wikidata, and keeps the raw files | Accepted | Shared infrastructure only, no analysis or visualization | Attribute columns in `country_aliases.csv`, used by both pairs. Argentina's driving side set by hand (Wikidata lists both sides) |

## Failure patterns observed

- **Defaults that quietly bias results** (#7): a convenient default (candidates = sampled countries) made the
  experiment depend on the sample. Only caught by an external review.
- **Settings tuned for one case** (#8): an English OCR confidence filter looked reasonable but fails exactly on the
  non-Latin scripts that matter for GeoGuessr.
- **Trusting a faulty source extraction** (#9): the AI reported "errors" in the paper that were artifacts of its
  own text extraction, and stated them confidently.
- **Over-engineering** (#6): the AI's first setup had far more files, checks and abstractions than needed.
- **Plots that hide weak evidence** (#19): the first Q4 plot showed a value based on one country, with no error bar. Only seen when we looked at the rendered plot.
- **Titles that claim more than the plot shows** (#21, #22): short "finding" titles were wrong twice. A zoomed axis also made a 3 point effect look big.
- **Edge cases in new code** (#29): one bad row was enough to switch off top-3/5 for the whole dataset. Only found by a test with deliberately broken rows.
- **Building more than you can show** (#32): the dashboard only works with a running kernel, and the written Q1 finding was missing.
- **Log falls behind** (#37): when the work got faster, we stopped logging and had to rebuild rows later.

## Best practices so far

- Ask the AI to cite where a fact comes from, and check numbers against the original source.
- Have a second reviewer (person or separate AI) look at the plan before building.
- Ask "what would make this result depend on our own choices?" for every default.
- Ask for the simplest version first; add structure only when it is needed.
- When the AI uses another download source, check the files against the lockfile hashes.
- Before blaming a known bug, check the raw data first.
- Look at every rendered plot, not only the numbers.
- Count independent units (countries, camera sequences), not only images.
- Write the finding in the plot title, then check it against the rendered plot and the numbers.
- Use random example images, not hand-picked ones.
- Compare every "shared attribute" against random pairs, and also within the same continent.
- Test new code on small made-up data with broken rows (unknown country, short list) before the real run.
- Before handing in, check that every number in the findings has a visible output in the notebook.
- Every masking test needs a random mask of the same size as control.
- Every interactive view also needs a static plot and a written finding.
- Let the AI check the repo against the task description before the deadline.
