# Reflection on GenAI use, Phase II

**AI-aided pair, team OpenSI**

| First name | Last name | Student ID |
| --- | --- | --- |
| Maximilian | Hammer | k12222785 |
| Florian | Weigelt | k12210011 |

Full record: `ai-interactions/phase2_ai_log.md` (#n = row number). This part covers Q2 and Q3 (Max); Q1 and Q4 (Florian) are added below.

## How we used AI

We used Claude with access to our repo for planning, code, plots, checking results and writing the findings. We ran every notebook ourselves on our own GPU and decided which results go into the notebook.

## What worked

- **New direction:** after we questioned the first plan (#7), AI suggested opening the **text side** of the model, which pair A never looked at. That gave our main result: Georgia photos look like the Caucasus and the Balkans, but the word "Georgia" sits among the US-state prompts. The model also knows US states from the images (46 %, chance 2 %).
- **Many designs, fast:** in one exploration round AI tried about 10 designs on cached data in under an hour (#10). We kept 7, among them the US map with every image as a dot, the image-vs-text mirror chart and the "same image, two explanations" cards.
- **Speed in debugging:** the first Grad-CAM run would have taken about 10 hours on our laptop GPU. AI found the cause and fixed it, the same run then took 15 minutes with identical heatmaps (#6).

## What failed

- **Shallow first plan:** Grad-CAM vs. masking gave clean results that only confirmed Phase I (#7).
- **Wrong assumptions:** code taken over from a 16 GB GPU setup without asking about ours (#6), and a known bug repeated ("NA" = Namibia read as missing, #9).
- **Confounded plot:** the first concept plot showed "unique landscape = easy" (r = -0.47); inside countries the effect was gone (r = 0.00) (#11).
- **Labels taken literally:** AI wrote "cypress trees" into the findings, but the US images with this label are Southern forest roads, not cypresses (#14).
- **Failed design:** mean Grad-CAM maps were all blue and said nothing.

Like in Phase I, these mistakes looked correct. We only found them by asking "is this the right direction?", by repeating effects inside countries and by looking at random example images.

## Speed and confidence

Our Q2 and Q3 for Phase II took one afternoon and evening (26 Nov), including a full exploration round. Checking took a big part of that time: 7 of our 14 log rows are AI suggestions or AI code that we rejected, changed or corrected. We trust a result when it holds against a control (random masks, same countries, within-country comparison) and when the example images show what the label says, not because the AI wrote it in the findings.

## Best practices

1. Ask "is this really the right direction?" before polishing a result.
2. Let AI explore many designs, then keep only the few with a clear message.
3. Repeat every between-group effect inside countries.
4. Look at random examples of every AI-made label, not only the best ones.
5. Tell the AI your hardware and known data pitfalls first.
6. Check every number in the findings against a visible output.

## Comparison with the human-only pair

- **Different directions, same story:** pair A worked on the image side (kNN, PCA, Grad-CAM for US -> Canada), we worked on the text side and on Grad-CAM vs. masking. For Georgia this fits together: their kNN shows the photos are recognisable (29 %), our text analysis shows the word points to the US.
- **Same main results:** both pairs found that confident errors lean on vegetation (they: US -> Canada, we: all confident errors, and almost half the heat for name-clash errors) and both used the same Grad-CAM layer and "extra heat" measure, so the numbers can be compared.
- **They found what we missed:** confident errors lie inside the other country's cluster in the embedding space (44.7 % kNN vote), and built-up images are easier but not more typical for their country.
- **We found what they missed:** the text side of the Georgia error, US-state knowledge in the image encoder, where Grad-CAM and masking disagree, and which scenes help or hurt inside a country.
- **Visualizations:** they used more statistical plots (kNN rates, PCA, box plots, world maps). We used more visual and interactive ones (maps of single images, explanation cards, prompt dial, Plotly maps), but we also threw more designs away.
- **Size:** similar amount of code (about 580 lines for our Q2/Q3 vs. about 500 for their Q2/Q4) and plots (9 vs. 13). With AI we did not write more, we tried more and kept less.
- **Confidence:** both pairs used controls (they: 50 km rule and balanced references for kNN; we: random masks, within-country checks, check galleries).
- **Phase I caveat:** in Phase I, AI was also used for pair B's Q2 and Q3 (brainstorming, discussing results, reviewing code and plots), which the human-only role did not allow. This is added to `ai-interactions/phase1_ai_log.md`. So for Q2/Q3, Phase I pair B is not a clean human-only baseline; the comparison holds best for Phase II.

## Q1 and Q4 (Florian)

(to be added)
