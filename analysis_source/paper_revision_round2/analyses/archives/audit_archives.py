from collections import Counter
from hashlib import sha256
from itertools import combinations
from pathlib import Path
import json

import numpy as np
import pandas as pd

out = Path(__file__).resolve().parent
root = out.parents[2]
cyp_cols = [f"CYP{x}_pIC50_direct_inhibition" for x in ("1A2", "2C9", "2D6", "3A4")]
cyp_panel = pd.read_csv(root / "cycle05_data/cyp-challenge-TEST-BLINDED.csv")
old = pd.read_csv(root / "cycle08_benchmark/frozen_versions.csv")
other = json.loads((root / "cycle08_independent/prediction_manifest.json").read_text(encoding="utf-8"))
seen = set()
manifest = []


def archive(owner, repo, commit, date, path, source, expected):
    assert date < "2026-11-04"
    assert sha256(source.read_bytes()).hexdigest() == expected
    data = pd.read_csv(source)
    assert len(data) == 750 and data.Molecule_Name.is_unique
    assert set(data.Molecule_Name) == set(cyp_panel.Molecule_Name)
    assert np.isfinite(data[cyp_cols]).all().all()
    if expected in seen:
        return
    seen.add(expected)
    manifest.append(dict(owner=owner, commit=commit, commit_utc=date,
                         source_path=path, fixed_url=f"https://github.com/{repo}/blob/{commit}/{path}",
                         sha256=expected, candidate_count=len(data), evaluated_name=source.stem))


for r in old.itertuples():
    archive("adlvdl", "adlvdl/cyp_competition", r.commit, r.committed_at, r.source_path,
            root / "cycle08_benchmark" / r.file, r.sha256)
for r in other:
    if r.get("blob_hash_matches") and r.get("rows") == 750 and r["commit_date"] < "2026-11-04":
        source = root / "cycle08_independent" / r["local_file"]
        if source.suffix == ".csv":
            archive(r["owner"], r["repository"], r["commit_sha"], r["commit_date"],
                    r["path"], source, r["sha256"])
assert len(old) == 10 and len(other) == 57
assert Counter(r["owner"] for r in manifest) == {"adlvdl": 9, "gashawmg": 49, "Scigantic": 4}
combined = pd.DataFrame(manifest)
source = pd.read_csv(root / "cycle08_benchmark/multiteam_scores.csv")
archived_scores = source.query("owner != 'local'")
for _, group in archived_scores.groupby("project"):
    assert set(group.name) == set(combined.evaluated_name)
assert len(archived_scores) == 248
combined.to_csv(out / "cyp_prediction_manifest.csv", index=False)

pxr_source = {r["team"]:r for r in json.loads((root / "cycle05/prediction_sources.json").read_text(encoding="utf-8"))}
pxr = pd.read_csv(root / "cycle05/external_validation.csv")
pxr_manifest = []
for team in pxr.team:
    r = pxr_source[team]
    assert r["status"] == 200 and r["n"] == 513
    assert sha256((root / f"cycle05/{team}_final.csv").read_bytes()).hexdigest() == r["sha256"]
    commit = json.loads((root / f"cycle08_benchmark/pxr_{team}_commit_history.json").read_text(encoding="utf-8"))[0]
    time = commit["commit"]["committer"]["date"]
    assert time < "2026-07-03"
    url = r["url"].replace("raw.githubusercontent.com", "github.com").replace("/main/", "/blob/"+commit["sha"]+"/", 1)
    pxr_manifest.append(dict(team=team, fixed_url=url, commit_utc=time, sha256=r["sha256"]))
pd.DataFrame(pxr_manifest).to_csv(out / "pxr_prediction_manifest.csv", index=False)

examples = pd.read_csv(root / "cycle08_benchmark/cross_benchmark_examples.csv")
assert len(examples) == 21 and len(examples[["project","a","b"]].drop_duplicates()) == 7
observed_reverse = examples.loss_a.sub(examples.loss_b).mul(examples.net).gt(0)
finite_reverse = observed_reverse & examples.lower_net.mul(examples.upper_net).gt(0)
assert observed_reverse.sum() == 11 and finite_reverse.sum() == 8
pairs = pd.read_csv(root / "cycle08_independent/ncats_metric_vs_hit_pairs.csv")
assert len(pairs) == 6 and pairs.net_lower.lt(0).all() and pairs.net_upper.gt(0).all()
same = pd.read_csv(root / "cycle09/external/same_cohort_pairs.csv")
old_ncats = pd.read_csv(root / "cycle08_independent/ncats_qualified_median_targets.csv")
matched = pd.read_csv(root / "cycle09/external/same_cohort_candidate_results.csv")
gone = set(old_ncats.candidate_id).difference(matched.candidate_id)
assert len(old_ncats) == 1766 and len(matched) == 1765 and gone == {"SID170464933"}
curves = pd.read_csv(root / "cycle09/external/ncats_qualified_curve_window_audit.csv", low_memory=False)
excluded = curves[curves.analysis_id.isin(gone)][["candidate_id", "analysis_id", "PUBCHEM_CID", "pIC50",
                                                  "min_included_uM", "max_included_uM", "fitted_AC50_uM", "window_reason"]]
assert len(excluded) == 1 and excluded.window_reason.iat[0] == "outside_observed_window"
assert excluded.fitted_AC50_uM.iat[0] > excluded.max_included_uM.iat[0]
excluded.to_csv(out / "ncats_window_exclusion.csv", index=False)
for row in same.itertuples():
    a = matched[f"selected_{row.method_x}"]
    b = matched[f"selected_{row.method_y}"]
    net = int((a & matched.hit.eq(1)).sum()) - int((b & matched.hit.eq(1)).sum())
    u = matched.hit.isna()
    assert (net, net - (b & ~a & u).sum(), net + (a & ~b & u).sum()) == (
        row.observed_hit_x_minus_y, row.unknown_lower, row.unknown_upper)
    assert np.isclose(row.MAE_x_minus_y, (matched[row.method_x] - matched.median_pIC50).abs().mean() -
                      (matched[row.method_y] - matched.median_pIC50).abs().mean())

rules = pd.read_csv(root / "cycle03/endpoint_sensitivity.csv").query("phase == 'all'")
reversed_rules = [(x.method, y.method) for x, y in combinations(rules.itertuples(), 2)
                  if (x.estimated_positive-y.estimated_positive)*(x.confirmed_positive-y.confirmed_positive) < 0]
assert reversed_rules == [("Feature_NN", "AI_NN")]
print("Verified files: CYP 62, PXR 3; selected archive contrasts 21; NCATS exclusion SID170464933; PXR local eligible-rule reversal 1/28.")
