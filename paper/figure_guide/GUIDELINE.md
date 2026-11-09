# Figure guideline for NeurIPS, ICML, ICLR, and CVPR papers

Version 1.1, 2026-11-09. The code that applies these rules is [paperplot.py](paperplot.py), and every rule is shown in [demos/](demos/) (made by [make_demos.py](make_demos.py), with invented data only).

```python
import matplotlib.pyplot as plt
import paperplot as pp

pp.use("cvpr")                                          # or icml, neurips, iclr, iccv
fig, ax = plt.subplots(figsize=pp.size("single", h=2.0))
pp.line(ax, x, y, "ours", lo=y_lo, hi=y_hi)
pp.save(fig, "figures/my_plot")                         # .pdf + .png, then automatic checks
```

## 1. Start with the claim

1. Write the one sentence the figure must prove before you plot anything. If it needs two sentences, make two figures or two panels.
2. Choose the form from the claim (table below). If exact values are the point, use a table.
3. Make "ours" the first thing the eye lands on. A reviewer skimming the page should get the claim from the figure and its caption alone.

| The claim is about | Use | Demo |
|---|---|---|
| Progress over training or steps | Line + band, small multiples per dataset | 01 |
| Selective prediction or abstention | Risk-coverage curve with AURC in the legend | 02 |
| A few methods on a few datasets | Grouped bars from zero, or dots with CIs | 03, 04 |
| Aggregate performance over many tasks | Interval plot (IQM, mean, optimality gap) | 04 |
| Before vs. after for many categories | Dumbbell, sorted by gain | 05 |
| Calibration | Reliability curve + confidence histogram | 06 |
| Two categorical factors | Annotated heatmap | 07 |
| Run-to-run distribution and significance | Violin + points + paired test | 08 |
| Trade-off between two metrics | Scatter with Pareto front | 09 |
| What each component contributes | Ablation bars, one hue getting stronger | 10 |
| Robust comparison of score distributions | Performance profile | 11 |
| Hyperparameter sensitivity | Log-x line with baseline band and default marker | 12 |
| Part-to-whole outcomes | 100% stacked bars | 13 |
| How the method works | Method figure with a worked example (Section 9) | 15 |
| What the model actually says | Qualitative response cards (Section 10) | 16 |

Avoid pie charts, 3D plots, radar charts for numeric comparison, and dual y-axes. Split two scales into two panels.

## 2. Author at the final printed size

Set `figsize` to the printed size and include the PDF at full width, so 8 pt in code is 8 pt on paper. Large canvases shrunk in LaTeX make fonts and lines unpredictable.

| Venue | Column (in) | Full width (in) | Source |
|---|---|---|---|
| CVPR / ICCV | 3.28125 | 6.875 | `cvpr.sty` in this repo |
| ICML | 3.25 | 6.75 | `icml2025.sty` |
| NeurIPS | one column | 5.5 | `neurips_2023.sty` |
| ICLR | one column | 5.5 | `iclr2026_conference.sty` |

Check these against each year's kit. Typical heights: a single-column plot 2.0-2.4 in, a row of small multiples 1.3-1.7 in.

```latex
\includegraphics[width=\linewidth]{figures/my_plot.pdf}   % one column
\includegraphics[width=\textwidth]{figures/my_plot.pdf}   % inside figure*
```

Never use `bbox_inches="tight"`: it changes the output width, so LaTeX rescales the figure. `paperplot` uses constrained layout at the exact size.

## 3. Text

- Axis labels 8 pt, ticks and legend 7-7.5 pt, nothing below 6 pt.
- Computer Modern serif by default, matching the math in the paper. Use `font="sans"` for slides and posters. Long model responses in qualitative cards are set in sans.
- Lowercase axis labels with units: `accuracy (%)`. Capitalize only proper names and method names.
- Regular weight in plots, with bold only for panel tags. Method figures use bold component labels, and qualitative cards use bold titles.
- No figure titles; the caption does that job. Panel titles name the dataset or condition.
- Use math mode for real minus signs and symbols: `$-1.2$`, not `-1.2`.

## 4. Color and encoding

The fixed slots below passed a palette validator (all-pairs, white background): worst deuteranopia ΔE 7.9, worst normal-vision ΔE 18.1, all contrasts at least 3:1. A ΔE below 8 is acceptable only with a second encoding, so every slot has one.

| Role | Hex | Marker | Line | Hatch | Use |
|---|---|---|---|---|---|
| `ours` | `#2E7BD6` | circle | solid, thicker | none | The proposed method, always |
| `a` | `#C9302C` | square | solid | `////` | Strongest baseline |
| `b` | `#BF8700` | triangle | dashed | `\\\\` | Baseline |
| `c` | `#A23BA6` | diamond | dash-dot | dots | Baseline |
| `d` | `#159E86` | down triangle | dotted | `xxxx` | Baseline |
| `base` | `#8C8C8C` | circle | solid | none | Untreated model, background points |
| `oracle` | `#333333` | none | long dash | none | Upper bound or reference |

- A method keeps its color in every figure in the paper. Assign slots once and never by rank.
- At most five methods in one panel. Beyond that, show the best few and move the rest to the supplement.
- Variants of one idea share a hue. Ablations use tints of `ours`.
- Sequential data: `viridis`. Signed data: `BrBG`, centred at exactly zero. Never `jet` or rainbow.
- Do not use red and green to mean good and bad in heatmaps, because colorblind readers confuse them and red is a method color.
- Light bar shades do not separate methods by color alone. At the default shades the validator gives a worst normal-vision ΔE of 8.0, a worst colorblind ΔE of 3.8, and contrast of 1.6-2.4:1 against white. Every baseline bar therefore keeps its hatch, drawn in the full role color.
- Text is ink or muted gray, not the series color. The one exception is a wrong span in a qualitative card, which is red and underlined.

## 5. Marks and chrome

- Thin full frame (0.6 pt), no tick marks, light solid grid (0.5 pt) behind the data.
- 3-5 ticks per axis. Remove the grid along a categorical axis.
- Lines 1.2 pt, `ours` 1.7 pt, nothing under 0.5 pt (ICML template).
- Bands: same hue at 18% opacity, no edge. Whiskers: 0.6 pt dark with small caps.
- Bars use light shades, mixed with white so gridlines do not show through: 46% of the role color for baselines and 66% for `ours`, so ours still reads first (`pp.bar`). Use small white gaps, no outlines, and hatches in the full role color.
- Markers only with few x positions. Dense curves rely on color and dash.

## 6. Honest axes

- **Bars start at zero.** If the interesting range is narrow, switch to dots or dumbbells, which can use a tight axis.
- Lines, dots, and heatmaps may use a tight range. Say so in the axis or caption if it could mislead.
- Log axes are labeled as such, with readable powers.
- Panels being compared share y-limits. Use `sharey=True` wherever units match.
- Put "lower is better" on the axis or title if the direction is not obvious.

## 7. Uncertainty and statistics

- No point estimate without an interval when runs vary. Every band or whisker is defined in the caption: statistic, level, number of seeds.
- Prefer 95% bootstrap CIs (`pp.boot_ci`) over bare standard deviations. With few runs on many tasks, report IQM and optimality gap as interval estimates (`pp.iqm`), following Agarwal et al.
- Show individual runs when n is below about 30 (demo 08).
- Use paired tests for paired runs (`pp.paired_p`, a sign-flip permutation test). Print the actual p-value, not stars. Bracket only the comparisons the claim needs, and correct for multiple comparisons (for example Holm) when you make several.
- State exactly what is averaged over and what the unit of variation is.

## 8. Legends, labels, and panels

- Put the legend at the best location: the free spot inside the plot that covers no line, band, bar, or label. `pp.legend` finds it and verifies it. Call it last, after the limits are set.
- If no spot is free, add headroom by raising the y-limit while keeping the ticks in the real range (demos 02, 03), and stop reference lines short of the free area (demo 12).
- Multiple panels: one legend, `Ours` first, in whichever panel has room.
- A legend goes below the figure only when the plot truly has no free space, as with 100% stacked bars or dense small multiples (demos 01, 13). `pp.legend` falls back to this by itself.
- Few, well-separated curves can use direct labels instead of a legend.
- Never put a label or legend on top of data. `pp.save` warns if you do.
- Label values only where the reader needs them, usually `ours` or the best baseline. Otherwise use a table.
- Show y-labels only on the leftmost panel with shared axes. Use one shared x-label (`pp.shared_xlabel`).
- Use `pp.tag` for (a), (b), (c) only when the text refers to individual panels.

## 9. Method figures

Demo 15 draws this style with `pp.canvas`, `pp.box`, `pp.trapezoid`, `pp.arrow`, `pp.tokens`, `pp.score_card`, and `pp.image`. The rules also apply in draw.io or TikZ.

- Read left to right: the input on the left, one main path, the final answer on the right.
- Use two tones. Existing parts are pale gray (`base`), and everything the paper adds is blue (`ours`), matching the plots.
- Mark what is new twice: a dashed outline around the added module, and blue arrows for its path. Existing paths use ink arrows.
- Arrows are thick (about 1.3 pt), straight or right-angled, and do not cross.
- Run one real example through the pipeline: the actual image, the actual question, tokens as small boxes, real numbers inside the score cards, and the final answer as text with the deciding word in bold.
- Put the key equation in the flow, at the place where it happens, so the cards become its terms.
- Rounded boxes for modules, trapezoids for encoders. Labels are short and bold at 7-8 pt, never below 6 pt.
- Export as vector PDF at the printed width. Photos inside it follow the raster rules in Section 11.

## 10. Qualitative examples

Demo 16 builds this layout with `pp.text_card` and `pp.badge`.

- Layout: image top left with the reference below it, the question in a band across the top right, and model responses stacked under the question.
- One pale card per model, in that method's color: `ours` blue, `base` gray, reference amber. No dark borders.
- Each card has a bold title and a small round badge, with the response in 7 pt sans.
- Mark only the wrong words, in red and underlined. Red alone is hard to see with deuteranopia. Leave correct text alone.
- Show every model on the same input, with outputs verbatim. If you shorten one, write `[...]` and say so in the caption.
- Say how examples were chosen, and include at least one failure of your own method in the paper or supplement.

## 11. Export

- Plots are vector PDF, with `rasterized=True` only for very dense scatters (over about 5k points).
- Raster content: PNG for sharp images, JPEG for photographs, at least 300 dpi at printed size.
- Fonts must be embedded and never Type 3. `pp.use` sets `pdf.fonttype = 42`, and `pp.save` checks with `pdffonts`.
- **ICML's template asks for embedded Type-1 only**, and matplotlib's default embeds TrueType. For ICML, render with `pp.use("icml", tex=True)` on a machine with LaTeX, then run `pdffonts paper.pdf`. NeurIPS objects to Type 3 and to non-embedded TrueType, so embedded TrueType fits its wording.
- Keep the plotting script and data next to the figure, and have the script read the data from a file. Every number in a figure must be regenerable.

## 12. Captions

- First sentence states the takeaway. Then define what is plotted, the band or whisker, n, and any abbreviations.
- Caption goes below the figure. NeurIPS asks for lower-case captions except the first word and proper nouns.
- The figure and caption should make sense without reading the main text.

## 13. Before submission

Automatic in `pp.save`: font size, line width, text clipped by the figure or frame, text collisions, text on lines, legends over data, Type 3 or unembedded fonts.

By hand:

- [ ] One sentence states what this figure proves, and the caption opens with it
- [ ] `ours` has the same color in every figure
- [ ] Bars start at zero, compared panels share limits
- [ ] Every band and whisker is defined, with n
- [ ] `pp.print_check(png)` output is readable in grayscale and deuteranopia
- [ ] Readable at 100% zoom on a laptop and on a printed page
- [ ] Figure regenerates from its script and data
- [ ] `pdffonts paper.pdf` shows no Type 3 and everything embedded
- [ ] Anonymous submission: no names, paths, or logos inside the figure

## 14. Where the sources disagreed

| Topic | Swarm-Visualization | arXiv:2609.12278 | figures4papers | Decision |
|---|---|---|---|---|
| Background | `darkgrid` | White, full box, light grid | White, top/right removed | White, thin full box, light grid |
| Text | Bold, 20 pt, LaTeX | Regular CM serif, lowercase | Helvetica, 16-24 pt | Regular CM serif at true 8 pt |
| Canvas | Not set (20 pt fonts imply shrinking) | Small panels | Up to 45×12 in | Exact print size |
| Y-axis on bars | Not specified | Zero (Fig. 8) | Tightened to the data | Zero, or switch to dots |
| Value labels | Not specified | None | On every bar | Selective |
| Bar fill | Seaborn default | Plain fill | Black edges, hatch | Light shades, white gaps, role-colored hatch |
| Ours color | Not specified | Red | Blue | Blue |
| Legend | `loc="best"` | Shared, below | Dedicated panel | Best free spot inside the plot, verified; below only if none |
| Statistics | Wilcoxon annotations | Shaded bands | Optional bands | Intervals always, exact paired p-values |
| File output | 600 dpi `savefig` | Embedded rasters | PNG + PDF | Vector PDF + preview PNG |

Taken from each: from Swarm, best-location legends, paired significance testing and run-level distributions; from arXiv:2609.12278, the overall look, shared legends, baseline bands on sweeps; from figures4papers, semantic color roles, hatches for print, tint-based ablations, consistent helper functions. The method-figure and qualitative-card styles and the light bar shades follow MARINE (arXiv:2402.08680). Added from wider guidance: exact venue geometry and font rules, validated color, interval estimates, and automated checks.

## Sources

- [Swarm-Visualization](https://github.com/UTAustin-SwarmLab/Swarm-Visualization) (`general_utils.py`, `statistics_utils.py`)
- [Palafox and Fridovich-Keil, arXiv:2609.12278](https://arxiv.org/pdf/2609.12278) (Figs. 4-8)
- [figures4papers](https://github.com/ChenLiu-1996/figures4papers) (`design-theory.md`, `api.md`, `common-patterns.md`)
- [MARINE, Mitigating Object Hallucination in Large Vision-Language Models via Image-Grounded Guidance, arXiv:2402.08680](https://arxiv.org/abs/2402.08680) (method figure, qualitative example, bar shades)
- [Rougier, Droettboom, and Bourne, Ten Simple Rules for Better Figures](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4161295/)
- [Agarwal et al., Deep RL at the Edge of the Statistical Precipice](https://arxiv.org/pdf/2108.13264)
- [Paul Tol's color schemes, via khroma](https://rdrr.io/cran/khroma/man/tol.html), and the Okabe-Ito palette
- Machado, Oliveira, and Fernandes (2009), for the color-vision simulation matrix
- Author kits: [CVPR 2026 guidelines](https://cvpr.thecvf.com/Conferences/2026/AuthorGuidelines), `cvpr.sty`, the ICML 2025 example paper and style, the NeurIPS 2023 template, and the ICLR 2026 template

Venues change their kits. The CVPR 2026 guidelines page has no figure-specific rules beyond using the template, so recheck the current kit before each deadline.
