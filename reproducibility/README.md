# Octant and NCATS frozen-result replay

This folder is a public distribution derivative of the local `Reproducibility_Round2.zip` research archive. It starts from saved predictions, adjudicated labels and saved shortlist membership. It does not train models, rebuild molecular embeddings, re-adjudicate raw assay records or recompute scaffold-bootstrap intervals.

## Run

From this folder, run `python -B replay_decisions.py` with Python 3.10 or later. Only the Python standard library is required. The expected result is:

```
PASS: 45 locked shortlists, 90 seeded contrasts, 30 endpoint contrasts, archive inventories and NCATS bounds.
```

The replay verifies the saved score checksum, reconstructs the five-seed top-150 CYP2C9 shortlists, checks 90 paired contrasts and 30 cross-endpoint comparisons, checks the archived prediction inventory, and checks that the full-catalogue NCATS bounds span zero. Included scaffold-bootstrap tables are archived descriptive results, not recomputed by this script.

## Contents

- `objectives/`: the 3-by-3 matrices at feature fractions 0.7 and 1.0, five-seed saved predictions, settings, paired differences and archived scaffold resampling results.
- `archives/`: pinned source URLs and checksums, the 21 selected archive contrasts, and PXR/NCATS result tables.
- `cross_endpoint_matched_interval.csv` and `selected_memberships.csv`: observed and exclusive unresolved counts for both embeddings and all budgets.
- `late_labels.csv`: derived CYP2C9 outcomes; adjacent neighbor and assay CSVs document the fixed-list PXR analysis.
- `FILE_MANIFEST.csv`: sizes and SHA-256 values for every distributed file except this manifest itself.
- `PUBLIC_DERIVATIVE.json`: original archive and prediction-lock hashes and the public-copy transformations.

## Provenance and scope

The original ZIP and prediction lock remain preserved locally and are not overwritten. This public copy is not the byte-identical original frozen lock. In `objectives/prediction_lock.json`, 92 generated-cache read entries were omitted and the other 21 historical read paths were converted to source-project-relative paths. These are documentary provenance entries; they do not imply that every referenced training file or checkpoint is distributed here. Score hashes, scientific settings, labels, predictions and source fingerprints were retained. The README and distribution manifest were updated, and derivative provenance was added.

The distinction between local result replay and training reproduction must be preserved. Third-party data, model checkpoints and upstream code retain their original terms; inclusion of derived tables or provenance does not grant additional rights to upstream materials.
