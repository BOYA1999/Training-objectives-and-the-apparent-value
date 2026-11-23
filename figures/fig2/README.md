# Figure 2 Shortlist boundary cases

Figure 2 shows how training objectives change a fixed-budget CYP2C9 shortlist. Four compounds are displayed across three feature sets, three objectives and five fitted seeds. Ranks are calculated among the same 750 Octant candidates, with ranks 1–150 selected.

For FP at seed 20261120, moving from absolute-point to squared-point training replaces 33 of 150 entries. Nineteen supported hits enter and four leave, giving a net gain of 15. Moving from squared-point to interval training replaces 13 entries: four supported hits enter and one leaves, giving a net gain of three. The row headings include all entering and leaving outcomes.

The examples were selected after examining the measured labels. In each transition, the entering supported hit has the largest new rank inside the selected set; the excluded below-threshold compound has the smallest new rank outside it. Augmented-model scores and other seeds were not used for selection. These are favorable boundary illustrations, not representative samples or one-for-one molecular replacements.

Measured potency intervals are the released 95% credible intervals from curve fitting. A supported hit has a lower pIC50 bound ≥6; a below-threshold case has an upper bound <6. The five sets of rank points describe model-fitting sensitivity, not assay intervals or calibrated predictive uncertainty.

## Files

- `plot_allocation_cases.py` and `academic.mplstyle`: plotting code and style.
- `source_data/locked_predictions.csv`: unchanged 750-candidate, five-seed prediction table; identical to the existing `reproducibility/objectives/locked_predictions.csv`.
- `source_data/late_labels.csv`: unchanged measured point estimates, intervals and outcomes; identical to `reproducibility/late_labels.csv`.
- `source_data/selection.json`: post hoc selection rule, case values and complete shortlist exchanges.
- `plotted_data.csv`: 180 records, comprising four compounds × nine configurations × five seeds.
- `CYP2C9_Assay_Allocation_Cases.png`, `.svg`, `.pdf` and `.tiff`: preview, vector files and RGB 600 dpi TIFF.
- `figure_validation.json`: saved data checks and subsequent format checks. `Manuscript_Fig2_caption.txt` contains the manuscript caption.

## Redraw

Tested with Python 3.12 and the package versions in `requirements.txt`. CairoSVG also requires the native Cairo library.

Copy this directory to a separate working folder, then run in that copy:

```sh
python -m pip install -r requirements.txt
python plot_allocation_cases.py
```

The script checks selection and ranking against the full saved inputs and writes the four figure formats, plotted CSV and a validation report. It does not fit models. It overwrites those outputs in its own directory, so running in a copy preserves the release checksums. The regenerated report contains the script's data checks; the saved report also contains later format checks. SVG/PDF serialization may differ between runs or software environments.

## Sources

The measured data come from the [OpenADMET CYP challenge](https://doi.org/10.57967/hf/11464). Their source versions and modeling workflow are documented in the repository's analysis directories. The two input CSV files retain their original bytes and checksums.

The case-based presentation was inspired by Fig. 6 of Horwood et al., [Improving molecular property prediction with score-based diffusion models](https://doi.org/10.1038/s41467-026-79240-y). All structures, values and graphics here were generated from this study's data. No original figure artwork, diffusion trajectory or predictive density was reused. Code follows the repository license; upstream data retain their own terms.
