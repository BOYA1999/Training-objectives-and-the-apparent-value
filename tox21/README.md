# Tox21 CYP2C9 transfer: frozen-result replay

From the repository root, run:

```text
python tox21/replay_tox21.py
```

Python 3.10 or later suffices. The replay uses only the standard library and does not write result files. It checks the public-release manifest, saved predictions and corrected outcome hashes, then reconstructs all five seeds, three label rules and four budgets. Expected output includes:

```text
PASS: 5 seeds, 22,140 scored rows, 540 shortlist cells, 1,200 paired contrasts, 60 sign-pattern checks.
```

The primary condition is seed 20261120, 885 slots and the counterscreen-adjusted primary label rule: 17 supported hits, 2,668 below-threshold outcomes and 1,743 unresolved outcomes. Blank labels mean unresolved; zero means below-threshold evidence under the applicable rule.

## Public derivative and historical evidence

This directory is a path-sanitized derivative of `Tox21_Frozen_Replay.zip`, not the original archive or an unchanged historical lock. The original archive remains preserved locally. Its SHA-256 is `f5bf7bca5c862fcebdc607d8cbd9e38024b304f88896d8d8ed80f27a12074cda`.

The original `fit/prediction_lock.json` contained 116 absolute project-path entries. This copy removes 92 generated-cache read identifiers and converts the remaining 24 to relative identifiers beginning `research_project/`. These identifiers describe the original research tree, not files promised to be present in this repository. The source fingerprints, prediction and protocol hashes, model settings, execution timestamps and results are unchanged. No original path-bearing lock, generated cache file, model weight or raw assay export is included.

`fit/prediction_lock.json` records its original SHA-256 and derivation. `repaired/repair_lock.json` is also a labeled derivative: its prediction-lock dependency hash is updated, while the original repair-lock hash and original prediction-lock hash are retained. `PUBLIC_DERIVATIVE.json` preserves the original archive member fingerprints and the original/public fingerprints of added source scripts. `SHA256.json` verifies this public derivative. Historical timestamps refer to the original runs, not a new fit or analysis lock.

## Assays and scientific scope

The source assays are [PubChem AID 1671198](https://pubchem.ncbi.nlm.nih.gov/bioassay/1671198) for CYP2C9 probe-turnover inhibition and [AID 1224835](https://pubchem.ncbi.nlm.nih.gov/bioassay/1224835) for the firefly luciferase counterscreen, described by [Sakamuru et al. (2025)](https://doi.org/10.3390/ijms26114976). `protocol.json` records source-export and candidate-registry fingerprints; `fit/fit_plan.json` records training provenance.

The initial adjudication summary is in `evaluation/summary.json`; corrected outcomes and results are in `repaired/`. The corrected decoder treats an all-zero 15-position mask as excluding no doses and uses the visible dose window. Both versions reused the same predictions. They are two versions of one retrospective analysis, not independent replications. The local protocol file was unchanged, but its mask handling was revised in the implementation. Neither version met the frozen finite-panel criterion. This local lock is not an external preregistration.

The supported replay starts from corrected derived outcomes and saved scores. It checks rankings, hit counts, paired unresolved-case bounds and sign-pattern decisions. It does not retrain models, rebuild embeddings, re-adjudicate raw PubChem records or recompute scaffold-bootstrap intervals. The scaffold-bootstrap file is an archived descriptive result. The initial full candidate-label table is not included, so this replay does not reconstruct both adjudication versions.

Scores are ranked by decreasing prediction, then SHA-256 of UTF-8 representative isomeric SMILES and candidate identifier. For equal-sized lists A and B, the observed hit difference is A minus B; the lower bound subtracts unresolved candidates unique to B, and the upper bound adds those unique to A. These are finite-panel bounds conditional on classified outcomes, not sampling confidence intervals.

## Scientific source code

The following four files are added byte-for-byte from the original research tree:

- `scripts/adjudicate.py`: initial dose-mask adjudication.
- `scripts/evaluate_external.py`: initial full evaluation (the actual historical filename).
- `repaired/adjudicate_v2.py`: corrected dose-mask adjudication.
- `repaired/evaluate_v2.py`: corrected full evaluation, including scaffold bootstrap.

Two further files retain the original scientific logic with documented repository-layout changes:

- `scripts/prepare_registry.py`: construct the value-blind candidate registry and remove connectivity overlaps; shared inputs now resolve under `../analysis_source/`.
- `scripts/fit_external.py`: fit the frozen-embedding 3 by 3 design across five seeds; shared imports/inputs now resolve under `../analysis_source/`, and read-audit identifiers and input-hash keys are relative to the repository root. The guard against reading test outcomes during fitting is retained.

These six scripts write outputs. They are supplied for inspection and reconstruction in a separate working copy, not as the standard-library replay command validated above. Restoring their inputs and dependencies is required before running them. No end-to-end raw-data reconstruction or new model fit was performed for this release.

## Inputs needed beyond the supported replay

Adjudication requires the exact files fingerprinted by `protocol.json`: `source/aid1671198.csv`, `source/aid1224835.csv`, their corresponding `*_description.json` files, and `external_structures.csv`. Those raw exports and the candidate registry are not included here. The adjudication scripts also expect a duplicate frozen protocol at `.aidd/candidates/tox21-cyp2c9/protocol.json` and verify its equality with `protocol.json`. The initial evaluator additionally requires the initial `external_outcomes.csv`.

Registry construction additionally requires the original Octant CYP2C9/PXR structure tables and earlier NCATS candidate structure table at their paths under `../analysis_source/`. Fitting requires the original 1,285-row training input, candidate registry, checkpoints and shared encoder modules. See the repository's `analysis_source` documentation and external-input manifest for restoration paths and fingerprints. Do not substitute differently versioned exports or weights while claiming an exact reproduction.

Adjudication uses the standard library. Full evaluation additionally needs NumPy, pandas and RDKit; the original environment recorded Python 3.12, NumPy 1.26.4, pandas 2.3.3 and RDKit 2026.03.3. Fitting also requires LightGBM and the encoder dependencies, including PyTorch. The frozen metadata records model settings and library versions. Source-dataset and model licenses remain those of their owners; the repository license does not replace upstream terms.
