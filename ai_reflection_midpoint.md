# Reflection on GenAI use, Phase I

**AI-aided pair, team OpenSI**

| First name | Last name | Student ID |
| --- | --- | --- |
| Manuel | Lisa | k12205828 |
| Umut | Tanriverdi | k12105807 |

Full record: `ai-interactions/phase1_ai_log.md` (#n = row number).

## How we used AI

We used Claude for planning, code, plots and reading results. We decided what to ask and checked every output (#2).

## What worked

- **More breadth:** four questions, a masking experiment and an interactive dashboard in one week.
- **Better methods:** AI suggested random baselines, confidence intervals and within-country checks (#16, #25, #34).
- **Key finding:** AI noticed that "Georgia" was almost never right (#14). The prompt meant the US state, so we
  changed the prompt for both pairs (#33).

## What failed

- **Bad defaults:** some settings quietly biased the results (#7, #8).
- **Confident but wrong:** AI reported "errors" in the paper that came from its own bad PDF reading (#9).
- **Too complex:** the first setup had far too many files (#6).
- **Weak plots:** a value from one country with no error bar, and plot titles that claimed too much (#19, #21).
- **Log fell behind** when the work got faster (#37).

These mistakes looked correct. We only found them by checking the raw data, the plot or the source.

## Speed and confidence

AI made coding much faster, but checking took a lot of the saved time. We used the saved time for more breadth
(four questions, a dashboard), not to finish earlier. We trust a result when it holds against a control (random pairs,
random masks, within-country), not because the AI agrees.

## Best practices

1. Check every number against the raw data and every plot by eye.
2. Ask: "Does this result depend on our own choices?"
3. Compare every effect with a random baseline.
4. Ask for the simple version first.
5. Log right away, not later.

## Comparison with the human-only pair

- **Same main results:** both pairs found that errors stay in the same region, that some countries attract many
  wrong guesses (they: Angola, Canada, Syria; we: North Korea, Moldova), that vegetation is the strongest visual
  cue, and that "Georgia" is read as the US state.
- **Different prompts:** the human pair used the paper's prompt ("A Street View photo in Georgia."), we used the
  new prompt with the subregion. So exact numbers can only be compared for the paper's prompt.
- **They found what we missed:** rotated images are the second cause of confident errors. Their Georgia prompt
  ("the country in the Caucasus") made real Georgia recognisable (0 % to 23 %); our subregion prompt did not (1 of 290).
- **We found what they missed:** most climate and land cover effects disappear within countries; their raw numbers
  mostly show the country mix. We also measured how much stronger than chance the patterns are (neighbours 13×),
  found that hard countries border each other, and that lane markings matter more than the road surface.
- **Visualizations:** they used fewer, static plots. We used interactive Plotly charts and a dashboard.
- **Breadth vs. depth:** with AI we covered more topics and used more controls. The human pair went deeper into
  single causes.
- **Time and size:** our first Q2 and Q4 versions took one afternoon (4 Nov). The human pair finished Q1 and Q4 on
  4 Nov, Q2 on 6 Nov and Q3 on 7 Nov (dates from git). We wrote about five times more code (about 4,500 vs. 840
  lines) and made more plots (34 vs. 19).
- **Checking cost:** 14 of our 37 log rows are AI suggestions that we rejected, changed or corrected. The human pair
  had no such step, because they wrote and understood every line themselves.
- **More data, same main result:** their Q3 used 3,660 images (20 per country), ours 12,154. Both found the same
  order of cues (vegetation, then buildings, sky less than a random area). More data and code gave us more detail
  (parts of cues, cues hidden together), not a different main result.
- **Confidence:** both pairs checked results against a control. They compared confident errors with correct images
  from the same countries and named their limits (0.8 threshold, 50 pilot images per country). We used random
  baselines, random masks and within-country checks.
