# Data eligibility and provenance audit for the manuscript

Audited 20 November 2026. The previous predictions, labels and results were read without modification. Counts were recomputed by `audit_flow.py`; `verification.json` records input hashes. `supplementary_data_flow.csv`, `method_eligibility.csv`, and `embedding_eligibility.csv` are the machine-readable tables. Evidence identifiers below are resolved in `source_evidence.csv`.

## Table S: candidate and measurement flow

| Analysis | Released input | Eligibility or grouping | Final analysis sample | Supported hit / non-hit / indeterminate |
|---|---:|---|---:|---:|
| Octant CYP2C9 training | 4,905 multi-assay records | 3,620 have all CYP2C9 measurement fields missing; no partially populated CYP2C9 records | 1,285 | Continuous labels; threshold-straddling intervals retained |
| Octant CYP2C9 test | 750 | No exclusions; all valid unique identities and complete CYP2C9 fields | 750; Top150 | 96 / 612 / 42 |
| Octant PXR continuous-model training | 4,139 | Remove 1,945 intervals not contained in the measured dose window and 3 without a matched window | 2,191 | Includes 60 threshold-straddling intervals |
| Octant PXR earlier binary-neighbor training | 2,191 eligible above | Additionally remove 60 threshold-indeterminate labels | 2,131 | 49 / 2,082 / 0 |
| Octant PXR local test | 513 | All candidate structures retained; unresolvable measurements remain unknown | 513; Top102 | 19 / 283 / 211 |
| Octant PXR historical final-submission evaluation | 513 | Remove 253 Phase 1 candidates whose labels had already been released | 260 Phase 2; Top52 | 14 / 145 / 101 |
| NCATS CYP2C9 external panel | 5,242 SIDs | Remove 3 invalid structures and 335 SIDs matching any Octant training connectivity; collapse 140 duplicate SID rows among 4,904 eligible SIDs | 4,764 identities; Top952 | 190 / 3,625 / 949 |
| NCATS same-cohort finite-potency sensitivity | 1,914 qualified curves | Remove 6 ambiguous excluded-point masks and 1 out-of-window fitted AC50; retain 1,907 curves and aggregate each compound by its median | 1,765 identities; Top353 | 190 / 1,534 / 41 |

Top-list budgets are floor(0.20 × cohort size). SIDs and repeated curves are not independent compounds. NCATS full-panel labels use all linked curves, including conflicts; the finite-potency sensitivity does not replace these labels by the median threshold call. Its MAE and shortlist are both computed on the same 1,765 compounds.

## Main CYP2C9 eligibility

CYP2C9 was chosen by the recorded completeness rule: maximize complete test measurements, then complete training measurements, then alphabetical order. Complete test counts were 748, 750, 748 and 748 for CYP1A2, CYP2C9, CYP2D6 and CYP3A4; complete training counts were 1,412, 1,285, 1,493 and 2,335. Missing source measurements cannot be relabeled inactive. The 4,905 count is a sparse multi-assay structure registry, not 4,905 measured CYP2C9 activities. [C1, C2]

Continuous-model preprocessing requires a finite point and ordered lower/upper bounds containing that point. It standardizes with RDKit FragmentParent and canonical isomeric SMILES. All rows belonging to duplicated training identities would be removed; invalid, early-overlapping or duplicated candidate identities would be excluded. None of these structure rules further reduces the CYP2C9 1,285/750 cohort. The four released fields are jointly complete or jointly missing, so the initial audit requiring point, bounds and std gives the same count as the final model setup requiring point and bounds. [M1]

All representations use the same 1,285 training and 750 evaluation rows. Mol2Mol cannot encode one training structure; ChemGPT cannot encode 140 training and 33 test structures. Those rows remain, with missing embedding coordinates handled by LightGBM and conventional features still available. They are not excluded candidates. Report this missing-feature handling rather than calling every structure fully represented. [M1; embedding_eligibility.csv]

The test rule is lower bound ≥6 for an interval-supported hit, upper bound <6 for a non-hit, and unknown otherwise. No raw-dose-window, Emax, or counter-screen qualification was applied to CYP test labels. Per-compound raw dose windows were not available in the evaluated package. The L2 point control was added after earlier results were observed; it is a retrospective mechanism check, not a preregistered experiment. [M1]

## Readouts and uncertainty terminology

Octant CYP2C9 measures recombinant human enzyme direct inhibition through EOMCC fluorescence, with Bayesian dose-response fitting. The dataset card calls its released bounds 95% CI. Prefer **released 95% curve-fit intervals from Bayesian fitting**; a credible-interval interpretation should not be confused with a frequentist interval across independent experimental repeats. The released std is not established as an independent-replicate standard deviation. The exact posterior/interval construction is not supplied in the cited card. CYP2D6 uses mass spectrometry, so the four-CYP panel must not collectively be called fluorescent. [C1, C3]

PXR measures activation of a cell-based chimeric human PXR ligand-binding-domain luciferase reporter. It is not a direct measure of endogenous CYP induction or clinical drug-interaction risk. The official source also describes Bayesian curve fitting. Use **released PXR curve-fit intervals**, because their nominal coverage and derivation were not fully defined in the retrieved documentation. The numerical audit reveals that all 4,139 training intervals equal pEC50 ± 1.96 × the released `std.error` to floating-point precision; none of the 513 test intervals matches that formula within 1e−10. Test bounds can be asymmetric. This is an observed difference in released-field construction, not evidence of empirical 95% calibration. Test bounds must not be reconstructed from the standard-error column. [P1, P3; verification.json]

PXR unknown outcomes comprise 177 intervals outside the observed dose window and 34 crossing the threshold. The three training records labeled `unknown_missing` have available fitted pEC50/bounds but lack matched raw-DRC windows. Calling them missing pEC50 would be incorrect. No additional test Emax or null-line specificity filter is used; the evaluated test labels do not establish confirmed selective PXR activation. [P2, M1]

NCATS measures apparent biochemical CYP2C9 probe-turnover inhibition by P450-Glo luminescence. The PubChem title contains “cell-based”, whereas the assay description and original paper describe the biochemical system. Follow the experimental methods rather than that title. Its numeric fit is a curve half-maximal concentration; `pIC50 = −Fit_LogAC50` is the analysis convention, not proof that all curves reach 50% absolute inhibition. The analysis accepts inhibition classes −1.1/−1.2/−2.1/−2.2 with efficacy >65% and finite pIC50. Qualified curves are thresholded at 6; class 4 is a non-hit only when the measured maximum covers 1 µM, and its arbitrary numeric fit is ignored. Other curves or conflicts remain unknown. There are no released fit CIs or independently dated replicate batches. Within-compound ranges and unknown-outcome identification ranges are not statistical confidence intervals. [N1, N2, M4]

## Dates and source independence

CYP training first appears in the repository on 3 August 2026 and was finalized by the 6 August snapshot. The saved early training and blinded-structure files are byte-identical to that snapshot. Test-unblinding is dated 4 November in the card and 5 November 01:55:55 UTC in the commit history; retain both descriptions without presenting either as a wet-experiment date. This analysis was performed after unblinding. [C1, C2]

PXR staged data appears in the saved history on 31 March 2026; training corrections were committed on 9 April. Phase 1 unblinding was committed on 27 May and Phase 2 on 3 July. Raw DRCs used for the present dose-window adjudication were released on 28 September, after the challenge. Consequently the current dose-window labels are retrospective evidence qualification, not evidence available to participants at original submission. Repository creation on 16 March does not prove public file availability, and source narratives differ on a March versus April challenge start. Use explicit file/phase cutoffs. [P1, P4]

NCATS publication is 28 June 2021 online and September 2021 issue; these are publication dates, not individual experiment dates. It provides independent laboratory and assay-system transfer, not a later temporal holdout. Octant produced both CYP and PXR, and the four CYP endpoints reuse the 750-compound panel: five endpoints are not five independent studies. Removal of Octant supervised-training overlap from NCATS does not prove absence from frozen-model pretraining corpora. [N2, C3, P3]

## Additional wording requiring correction

The NCATS script refits the fixed procedure on the same 1,285 Octant training records. It does not load an identical serialized fitted tree model. Write **same training data and fixed modeling procedure**, unless a separate serialized-model replay is actually performed. The external run has the earlier point-L1 and interval comparisons; it does not include the later L2-point matched control and cannot replicate the entire main factorial comparison. [M4]

The external ordinary MAE is identifiable only for a selected finite-potency subset. Class-4 inactivity does not imply a finite latent relative AC50 beyond one universal maximum dose. Do not assign pseudo-potencies, claim full-4,764-panel MAE, or compare full-panel shortlist gains directly to subset MAE as though they concern one population. The same-cohort sensitivity shows that some previously observed cross-population rank reversals disappear. [M5]

The manuscript NCATS cohort is the `cycle08_ncats` 4,764 set. Earlier files containing 5,012 or 5,067 parent structures use different identity/exclusion rules and must not be mixed into its flow table. Historical submitted predictions provide additional context but have nonmatched training resources and selection histories; they are separate from the controlled local representation comparisons. [M3, M4]

## Suggested concise Methods text

We retrospectively evaluated the released Octant CYP2C9 direct-inhibition panel using 1,285 training compounds with complete, ordered pIC50 fit intervals and all 750 fixed test candidates. The other 3,620 structures in the multi-assay training registry lacked CYP2C9 measurements. An interval-supported hit required its lower bound to be at least 6; an upper bound below 6 defined a non-hit, and remaining outcomes were indeterminate. All candidates, including those with indeterminate outcomes or unavailable frozen embeddings, remained in the ranking population. Supplementary analyses distinguished PXR models using continuous interval-qualified training data from earlier methods restricted to threshold-resolved labels, and evaluated transfer to an independently produced NCATS CYP2C9 panel after removing training-structure overlap. Continuous-error sensitivity on NCATS was restricted to a common finite-potency cohort for both error and shortlist metrics.
