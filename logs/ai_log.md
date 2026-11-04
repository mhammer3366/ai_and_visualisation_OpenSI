# AI interaction record - Phase I (AI-aided pair)

Record interactions that influenced the analysis or show strengths and limits of AI assistance.
Add a row right after it happens; it takes 30 seconds and becomes the reflection report later.

**Decision:** accepted / modified / rejected

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
| 14 | 2026-11-04 | Q2 first results | AI noticed that "Georgia" is never right: 0 of 290 real Georgia images, but 2436 predictions. Most come from the US state Georgia and nearby southern states | Accepted after check | Region column: 401 of 451 US images from the state Georgia are predicted as Georgia. The prompt name means both a country and a US state | Biggest single source of confident errors (US -> Georgia: 1126 images with probability >= 0.8) |
| 15 | 2026-11-04 | Precision per predicted country | Namibia had precision 0 with 909 predictions. AI first suspected the known "NA" read-as-missing bug | Rejected after check | The raw `test.csv` has no Namibia images at all | Added `n_true` column: countries without test images are never right by design |

## Failure patterns observed

- **Defaults that quietly bias results** (#7): a convenient default (candidates = sampled countries) made the
  experiment depend on the sample. Only caught by an external review.
- **Settings tuned for one case** (#8): an English OCR confidence filter looked reasonable but fails exactly on the
  non-Latin scripts that matter for GeoGuessr.
- **Trusting a faulty source extraction** (#9): the AI reported "errors" in the paper that were artifacts of its
  own text extraction, and stated them confidently.
- **Over-engineering** (#6): the AI's first setup had far more files, checks and abstractions than needed.

## Best practices so far

- Ask the AI to cite where a fact comes from, and check numbers against the original source.
- Have a second reviewer (person or separate AI) look at the plan before building.
- Ask "what would make this result depend on our own choices?" for every default.
- Ask for the simplest version first; add structure only when it is needed.
- When the AI uses another download source, check the files against the lockfile hashes.
- Before blaming a known bug, check the raw data first.
