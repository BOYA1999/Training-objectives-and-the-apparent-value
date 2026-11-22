# Scientific fitting and evaluation source

This directory contains 32 author-written Python source files used by the manuscript's computational analyses, preserving their original relative layout. It complements the runnable frozen-result replays in `../reproducibility/` and `../tox21/` by exposing feature construction, objectives, fitting, selection and evaluation.

This is a source release, not a self-contained training distribution. Raw data, prepared intermediate tables, archived participant submissions, pretrained weights, third-party implementations and feature caches are not bundled here. No new fitting or embedding extraction was run. Source syntax, provenance, dependency relationships and four path-only adaptations were checked. Numerical replay validation is reported separately by the replay packages. Historical sources and frozen archives remain unchanged.

## Source map

| Analysis | Included source | Required inputs and scope |
|---|---|---|
| Standardization and labels | `cycle06/prepare.py`; `cycle02_data_b/prepare_pxr_labels.py` | CYP downloaded tables and PXR prepared DRC tables. The preparation main block processes PXR and four CYP endpoints. |
| Frozen representations | `cycle06_model/frozen_encoder.py`; `cycle07_model/frozen_chemgpt.py`; `models_outputs/mol2mol_cpu.py` | External checkpoints and implementations listed below. Encoders remain frozen. |
| Absolute-error and interval objectives | `cycle06/fit.py`; `cycle06/fit_continuous.py`; `cycle07/run_chemgpt.py` | Standardized structures and early labels. Shared modules retain original historical binary baselines; the ChemGPT driver retains its five-assay loop. |
| Squared-error control and comparisons | `cycle09/fit_point_control.py`; `cycle09/evaluate_comparisons.py`; `cycle09/bootstrap_primary.py`; `cycle06/evaluate.py`; `cycle06/evaluate_continuous.py` | Locked scores, labels and manifests. Legacy evaluation also requires `cycle02/chembl28_structure_overlap.csv`. |
| Full-feature sensitivity | `paper_revision_20261120/feature_sampling/fit_full_features.py`, `evaluate_locked.py` | Same 1,285 training and 750 test structures; original scores and labels. |
| Five-seed sensitivity | `paper_revision_round2/analyses/objectives/fit_seeds.py`, `evaluate_objectives.py` | Same structures and early labels; original score outputs. Both sensitivity fitters recompute fingerprints and embeddings. |
| NCATS external | `cycle08_data/build_ncats_fixed_labels.py`; `cycle08_ncats/score.py`, `evaluate.py` | Prepared NCATS replicate, structure-mapping and candidate tables from AID 1645842. Their upstream construction is not reconstructed here. |
| NCATS sensitivity | `cycle08_independent/ncats_metric_sensitivity.py`; `cycle09/external/same_cohort_sensitivity.py` | External scores/manifests, labels, qualified curves and assay description. The latter provides the 1,765-compound, 353-slot dose-supported comparison; the former retains the earlier analysis with different MAE and selection denominators. |
| PXR neighbor and label rules | `cycle02/score_catalogue.py`; `cycle03/neighbor_methods.py`, `score_pxr_neighbors.py`, `evaluate_pxr_neighbors.py`, `audit_assay_rules.py`, `verify_endpoint_sensitivity.py` | Prepared PXR tables, labels and frozen neighbor scores. Shared modules retain unused original entry points; these do not represent added manuscript experiments. |
| Participant archives | `cycle05/evaluate_external.py`; `cycle08_benchmark/evaluate_multiteam.py`, `compare_examples.py`; `paper_revision_round2/analyses/archives/audit_archives.py` | Original public submission CSV collections, challenge labels and comparison outputs. Official ST-RAE code is an external dependency. |
| Tox21 | Fitting, preparation, adjudication and evaluation source in `../tox21/scripts/` and `../tox21/repaired/` | Imports shared fitting/encoder modules here. `cycle09/fit_point_control.py` also forms part of its source fingerprint. See the Tox21 package for adaptations. |

## Dependencies and inputs

`source_manifest.json` records original/published SHA-256, byte counts, imports and adaptations for every included source. It also records hashes for named primary inputs and key prepared intermediates, with acquisition pointers. This is not an exhaustive runtime-file inventory: scripts consume each other's outputs and archival analyses require participant prediction collections. Do not describe it as a complete training lock.

Restore corresponding prerequisites to the listed relative paths beneath this directory. Scripts retain their original filenames and assertions. Several perform computation at import time; do not import them merely to inspect their API. Use a separate working copy for a new fit, because derived outputs may be overwritten.

Python dependencies are NumPy, pandas, SciPy, scikit-learn, RDKit, LightGBM, PyTorch, Transformers and tokenizers, plus the upstream modules below. A newly installed environment and cross-version equivalence have not been tested for this release. Historical ChemGPT environment records specify PyTorch 2.11.0+cu128 (CPU execution), Transformers 5.10.2, tokenizers 0.22.2 and RDKit 2026.03.3; these records are not a fresh installation test.

| External prerequisite | Retrieval and expected location |
|---|---|
| Octant CYP data | Hugging Face `openadmet/cyp-challenge-train-test`, revision `5ab9dd9daee4dd98840ad585455a2b5aa386f5d0`; pinned URLs and hashes in the manifest. Place files under `cycle05_data/`. The historical dataset DOI is retained separately from the snapshot identifier. |
| PXR data | <https://huggingface.co/datasets/openadmet/pxr-challenge-train-test>. No unverified revision is asserted. Prepared DRC tables, labels and the label contract under `cycle02_data_b/` remain required; their hashes are recorded. |
| NCATS data | <https://pubchem.ncbi.nlm.nih.gov/bioassay/1645842>. Prepared inputs belong under `cycle08_data/`. The assay download alone does not reproduce every prepared structure mapping. |
| REINVENT implementation | <https://github.com/MolecularAI/REINVENT4>, recorded tree `0f075b349ed22015e0406350c6f426701bb0815d`; per-file Git blob hashes are in the manifest. Supply the matching package at `models_outputs/reinvent_minimal/reinvent/`. |
| Mol2Mol checkpoint | Manifest pins the download to REINVENT revision `b44124489e8ddeabaccf5856b5d823f1574db063`; expected location `models_outputs/mol2mol_similarity.prior`. SHA-256: `64039ba739e9a0464f13840495d35b60b14c6575a0f4571ef4fcdae4e077e14f`. The adapter also checks historical MD5. |
| ChemGPT checkpoint | <https://huggingface.co/ncfrey/ChemGPT-4.7M/tree/7438a282460b3038e17a27e25b85b1376e9a23e2>, under `cycle07_model/checkpoint/`. Weight SHA-256: `a8a39c5ea1d50c450632e003d21c90f847d815a801934b03a12b02010b74d21b`. Obtain weights subject to upstream terms. |
| SELFIES | PyPI distribution 1.0.4 (internal version 1.0.3), exact artifact URLs and hashes in the manifest. Adapter expects `cycle07_model/vendor/selfies/`. |
| Official challenge scorer | <https://raw.githubusercontent.com/OpenADMET/CYP-Challenge-Tutorial/main/evaluation/custom_scoring_functions.py>, under `cycle06_novelty/official_custom_scoring_functions.py`. This URL is mutable; verify the recorded historical hash. |

Third-party implementations and weights are not redistributed or relicensed by this source package.

```text
fit_continuous / fit_point_control / fit_seeds / fit_full_features / Tox21 fit
  -> cycle06/fit.py -> cycle06_model/frozen_encoder.py
  -> cycle07_model/frozen_chemgpt.py
frozen_encoder -> models_outputs/mol2mol_cpu.py -> external REINVENT + weight
frozen_chemgpt -> external Transformers + SELFIES + checkpoint
NCATS score -> prepare.standardize + fit_continuous + both encoders
PXR neighbor_methods -> cycle02/score_catalogue.py -> Mol2Mol
public archive evaluation -> prepare.standardize + official ST-RAE
objective / external evaluators -> locked predictions + adjudicated labels
```

After all corresponding prerequisites are restored, `python cycle06/fit_continuous.py cyp2c9` is the original Mol2Mol continuous-fitting entry point. `python paper_revision_round2/analyses/objectives/fit_seeds.py` fits five seeds; `python paper_revision_round2/analyses/objectives/evaluate_objectives.py seeds` evaluates them. These are scientific computations, not release validation commands. Use the repository replay instructions for the verified lightweight checks.

## Adaptations and validation boundary

Twenty-eight sources are byte-for-byte historical copies. Four have only path/provenance adaptations: the two sensitivity fitters match project-local reads using the resolved source root instead of a hard-coded project-folder name and log relative paths; the two NCATS sensitivity evaluators write relative provenance paths. Forbidden-label guards, features, objectives, parameters, seeds, selection rules and numerical formulas remain unchanged. Reversing the specified substitutions recovers the original text exactly.

A new run will fingerprint these public adapted sources, not the historical ones. Do not overwrite historical locks or describe a new lock as the original frozen record. Release checks establish traceability and syntax, not independent reproduction of model fitting, embedding extraction or the entire preparation chain.
