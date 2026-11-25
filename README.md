# Reproduce the CYP2C9 analyses

## Replay the saved results

Use Python 3.10 or later. Clone the repository and run the verifier from its root.

```sh
git clone https://github.com/BOYA1999/Training-objectives-and-the-apparent-value.git
cd Training-objectives-and-the-apparent-value
python verify_release.py
```

The verifier checks the file hashes in `FILE_MANIFEST.csv` and runs the [Octant and NCATS replay](reproducibility/README.md) and the [corrected Tox21 replay](tox21/README.md). Both use the saved scores and assay labels. Look for three `PASS` lines. Once the files are local, the verifier runs offline with the Python standard library.

To use the version cited in the manuscript, run `git checkout 2577a441669101068d7c6d72ce6f58871964edb4` before running the verifier.

## Redraw the archived Figure 2

Copy `figures/fig2/` to a separate working directory. In that copy, use Python 3.12 and install the native Cairo library for CairoSVG, then run:

```sh
python -m pip install -r requirements.txt
python plot_allocation_cases.py
```

The script checks the case selection and ranks from the saved inputs, then writes the figure files, plotted CSV and validation report. See the [Figure 2 instructions](figures/fig2/README.md) for the files and dependencies.

## Work from upstream inputs

For model fitting, embedding extraction, raw assay adjudication or scaffold resampling, follow the [analysis source guide](analysis_source/README.md) and the [Tox21 input and dependency notes](tox21/README.md). These steps require upstream files and dependencies in addition to this checkout.
