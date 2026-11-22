# Training objectives and the apparent value of molecular embeddings

Computational materials for the CYP2C9 assay-prioritization study, aligned with the fifth-round scientific revision dated 2026-11-22.

## Quick replay

Use Python 3.10 or later. No third-party packages, model weights or network access are needed for the frozen-result replay:

```sh
python verify_release.py
```

The command verifies the repository file manifest and runs both saved-result checks:

```sh
python reproducibility/replay_decisions.py
python tox21/replay_tox21.py
```

- `reproducibility/` checks the Octant CYP2C9 factorial and five-seed comparisons, endpoint/budget contrasts, archived prediction inventories and NCATS result tables.
- `tox21/` checks five seeds, three label rules and four budgets from frozen predictions and corrected adjudicated labels: 540 shortlist cells, 1,200 paired contrasts and 60 sign-pattern decisions.
- `analysis_source/` contains the corresponding original scientific preparation, modeling and evaluation code, with its input and dependency documentation. Tox21-specific source scripts are under `tox21/`.

The replay verifies calculations conditional on saved scores and labels. It does not train models, regenerate embeddings, re-adjudicate raw PubChem curves or recompute archived scaffold-bootstrap intervals. Raw-data modeling requires the separately identified data, checkpoints, upstream code and scientific environment. These steps have not been rerun end to end as part of this repository synchronization.

## Main results and scope

In the enriched 750-compound Octant CYP2C9 catalogue, the conventional feature models returned 50, 65 and 68 supported hits in 150 slots under absolute-point, squared-point and interval objectives. Interval-trained Mol2Mol and ChemGPT augmentations returned 57 and 60. Their apparent gains of +7/+10 against absolute-point FP became −11/−8 against interval FP. These observations condition on the fitted workflow and measured catalogue.

For Tox21, the initial decoder yielded 0/482/3,946 supported-hit/below-threshold/unresolved candidates. The corrected primary rule yielded 17/2,668/1,743. The correction concerns 15-position all-zero dose-exclusion masks with mismatched visible-dose lengths; non-activating records with mismatched nonzero masks supply no determinate evidence. The protocol file was unchanged, while the adjudication implementation was revised. Both versions failed the locally frozen criterion; they are successive versions of one retrospective analysis, not independent replications or external preregistration.

At the primary seed 20261120 and 885 slots, all 12 corrected contrasts across the primary, CYP-only and three-record rules were negative and all paired finite-panel bounds spanned zero. This statement does not extend to every other seed or budget. For the primary label rule, the two contrasts against absolute-point FP contradicted the expected positive observed direction; the two same-interval negative directions were expected, but not identified by bounds excluding zero. Finite-panel bounds vary unresolved outcomes and are not 95% confidence intervals or evidence of equivalence.

The full NCATS and Tox21 catalogues leave the direction of the cross-catalogue effect uncertain. These results concern retrospective in vitro signal recovery at a fixed assay budget, not prospective operational yield or clinical drug-interaction risk.

## Public copies and provenance

This repository contains path-sanitized public derivatives of the historical result packages. Local host paths and generated cache read records have been removed from lock metadata. Dependent public manifests and repair-lock references have been updated and verified. Frozen prediction scores, adjudicated labels and scientific result tables remain unchanged.

The `PUBLIC_DERIVATIVE.json` files record the original archive and lock checksums and the public-copy changes. A modified public lock is not the original historical prediction lock. Original archives remain preserved separately. The Tox21 public replay starts from the corrected labels; the initial version is retained through summary and QC records, not a complete initial candidate-label table.

`FILE_MANIFEST.csv` identifies this repository release. Each replay subtree also has its own integrity manifest. This repository does not include the manuscript, author declarations, model weights or raw assay exports.

## Data and licensing

Source measurements: [OpenADMET CYP](https://doi.org/10.57967/hf/11464), [OpenADMET PXR](https://doi.org/10.57967/hf/9731), [NCATS AID 1645842](https://pubchem.ncbi.nlm.nih.gov/bioassay/1645842), [Tox21 CYP2C9 AID 1671198](https://pubchem.ncbi.nlm.nih.gov/bioassay/1671198) and its [firefly luciferase counterscreen AID 1224835](https://pubchem.ncbi.nlm.nih.gov/bioassay/1224835).

The pre-existing repository [MIT license](LICENSE) is retained for the code. Source measurements, participant prediction archives, model checkpoints and upstream software retain their respective terms; this repository does not relicense those third-party materials. Pinned source URLs, revisions and checksums are recorded in the subdirectories.
