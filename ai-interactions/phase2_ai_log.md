# AI interaction record - Phase II (AI-aided pair: pair B)

Each row is one AI interaction that changed our analysis or shows a strength or limit of AI assistance.

**AI used:** Claude (chat with access to our repo)

**Decision:** accepted / modified / rejected

**Prompt:** all Phase II runs of pair B use the paper's prompt ("A Street View photo in {country}."), the same as
pair B's Phase I runs.

## Log

| # | Date | Who | Topic / prompt (short) | AI suggestion | Decision | Why | Effect on the work |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2026-11-26 | Max | "Check the repo for updates. We are now the AI pair, what do we need to do, how do we split it?" | Summary of pair A's Phase II work (kNN, PCA, Grad-CAM on US -> Canada). Keep the Phase I split (Max Q2 + Q3, Florian Q1 + Q4) so each of us continues his own findings; choose techniques pair A does not use | Accepted | Same split as in Phase I; Phase II should continue from Phase I | Split and separate notebooks (`phase2_AI_max.ipynb`, `phase2_AI_florian.ipynb`) |
| 2 | 2026-11-26 | Max | "Here is the Moodle page, check that we are not missing anything" | Checklist: at least 2 techniques compared or combined, group hypotheses, richer interaction for the AI pair, log with failure patterns and best practices, presentation structure, optional `suppl.md` | Accepted | Missing parts cost points | To-do list; one interactive plot per question planned |
| 3 | 2026-11-26 | Max | Which prompt for Phase II? | Keep the paper prompt from our Phase I runs; Grad-CAM depends on the prompt, so it has to be fixed | Accepted | Results stay comparable with Phase I | Note in README (shared default is still the subregion prompt) |
| 4 | 2026-11-26 | Max | Q3 Phase II design | Grad-CAM on the same 3,660 images and segments as the Phase I masking; target = true country (same as the masking effect P(true)); "extra heat" = heat share minus area share; hypotheses H1 to H3 (same clues, same images, same regions) | Accepted | Two techniques on the same data can be compared directly | Q3 section in `phase2_AI_max.ipynb` |
| 5 | 2026-11-26 | Max | Reuse pair A's Grad-CAM code? | Own small helper file, but the same layer and the same "extra heat" measure | Modified | Pipelines of the pairs should stay separate, but results should be comparable | `pair_b/gradcam_phase2.py`; AI tested it on a tiny random CLIP model before the real run |
| 6 | 2026-11-26 | Max | "The Grad-CAM loop takes almost 10 hours on my RTX 4070 laptop (8 GB)" | AI found its own setup computed gradients for all ~430M weights and kept the graph of all 24 layers; batch 4 in float32 does not fit into 8 GB, so the GPU spills into normal memory. Fix: freeze all weights except the target layer (heatmap unchanged, checked on a small test model: max difference 0.0), segmentation model to CPU (masks are cached), batch 8 | Accepted | 10 h is not usable; the fix does not change the result | Grad-CAM run (see notebook output for the new time) |
| 7 | 2026-11-26 | Max | After the first Grad-CAM run: "Is this really the right direction to answer the questions?" | AI agreed: Grad-CAM vs. masking only confirms Phase I (same ranking: vegetation +4.1 pts extra heat, buildings +2.2, sky -2.3; per image only weak agreement, r = -0.28 for vegetation) but does not explain what the model learned. New plan: Q2 = text encoder (where the "Georgia" prompt sits, image side vs. text side, crowded prompts), Q3 = image encoder (Grad-CAM vs. masking as a short check, correct vs. wrong, landscape concepts). Dropped: Grad-CAM US -> Georgia and kNN coverage (pair A already did the same kind), re-running the masking, rotated images (too few) | Modified | The push-back came from us, not from the AI; the AI's first plan was too narrow | Notebook rebuilt with Q2.1 to Q2.3 and Q3.1 to Q3.3 |
| 8 | 2026-11-26 | Max | Landscape concepts for the Q3.3 probe | AI suggested 18 concepts ("palm trees", "eucalyptus trees", "red soil", "rice paddies", ...) and the prompt "A Street View photo with {concept}." | Accepted, to be checked | AI-made attributes are not ground truth | Gallery cell shows the top images per concept; concepts that fail the check are reported |
| 9 | 2026-11-26 | Max | Q3.2 crashed: "nan is not in list" | AI found that pandas reads the country code "NA" (Namibia) as missing; 69 confident errors predicted as Namibia became NaN. Fix: read the CSVs with `keep_default_na=False`. Checked that the Phase I Q3 numbers are not affected (only wrong predictions were NaN, and they count as wrong either way) | Accepted | Known pitfall (pair A log #15), but the AI did not apply it to its own new code | Q3.2 runs on all 10,085 images |
| 10 | 2026-11-26 | Max | "Are the plots good enough for a vis course? Test freely, then we keep the highlights" | AI reviewed every plot (finding visible? own role? drill-down?), then tested ~10 new designs on cached data: image-level US map, Georgia image-vs-text mirror chart, prompt dial (10 Georgia prompt variants), embedding surgery, all-pairs text-vs-image scatter, heat-budget bars, explanation cards (Grad-CAM vs. masking on one image), contrastive Grad-CAM ("why Georgia?" vs. "why United States?"), concept probe on all 210k images | Accepted as exploration | AI-aided pair should use AI for breadth, then select | Highlights chosen with Max, the rest dropped |
| 11 | 2026-11-26 | Max | Concept probe on all images | First AI plot: "unique landscape = easy, shared landscape = hard" (r = -0.47 over 42 concepts). AI then checked it within countries: r = 0.00. The effect came from the country mix (eucalyptus = Australia, which is easy anyway) | Rejected, replaced | Same confounding trap as in Phase I (pair A row #16) | Replaced by a within-country plot: cultural / man-made cues help, generic nature hurts; "cypress trees" in the US is the Georgia trap (71 % of these US images -> Georgia) |
| 12 | 2026-11-26 | Max | Embedding surgery on the "Georgia" text embedding | Removing the "US-state direction" step by step lowers US -> Georgia from 2,343 to 452, but real Georgia stays at 3 % | Kept only as a side note | The prompt dial (real words instead of vector arithmetic) shows the same more clearly | Prompt dial used instead |
| 13 | 2026-11-26 | Max | "passt" (selection of the highlights) | Final notebook rebuilt around one story: Q2 = image-level US map, image vs. text space, prompt dial; Q3 = Grad-CAM vs. masking, explanation cards, heat budget, within-country concepts with a check gallery. Findings written in the notebook from the exploration numbers | Accepted, findings to be checked by us after the run | Every number in the findings must have a visible output | `phase2_AI_max.ipynb`; extra plots for `suppl.md` |
| 14 | 2026-11-26 | Max | Check of the final run against the written findings | All numbers matched the outputs. But the concept gallery showed that the AI's reading of "cypress trees" was wrong: the top images are real Italian cypresses, while the 787 US images with this label are Southern forest roads (concept score 0.1 to 0.5), not cypresses. "Tropical rainforest" also picks up aerial drone shots | Modified | The AI had interpreted the concept name literally; only the visual check showed what the label means | Findings rewritten ("cypress" = US-South forest detector), extra gallery row with random US examples |

## Failure patterns observed

- **Code that works but does not fit our hardware** (#6): the first Grad-CAM setup was copied from a 16 GB GPU setup and was far too slow on 8 GB. The AI did not ask about our GPU first.
- **Plausible but shallow plan** (#7): the first Phase II plan produced clean results that only confirmed Phase I. Only our own critical question moved it to "what did the model learn".
- **Known bug repeated** (#9): the "NA" = Namibia problem was already in the Phase I log, but the AI wrote new code without the fix.
- **Attractive but confounded plot** (#11): the AI's first concept plot showed a strong trend that disappeared inside countries. The AI found it only because we asked for a within-country check.
- **Concept names taken literally** (#14): the AI wrote "cypress trees" into the findings as if the images showed cypresses. The gallery showed they don't.

## Best practices so far

- Let the AI explore many designs on cached data, then pick a few with a clear role (#10).
- Check every between-group effect again inside countries (#11).
- Never trust an AI-made label without looking at random examples of it, not only the best ones (#14).
