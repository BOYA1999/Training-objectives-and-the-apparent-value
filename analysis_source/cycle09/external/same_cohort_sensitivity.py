from pathlib import Path
from itertools import combinations
import hashlib
import json
import numpy as np
import pandas as pd

out = Path(__file__).parent
root = out.parents[1]
source = root / 'cycle08_ncats'
raw = root / 'cycle08_data/NCATS_CYP2C9_fixed_curve_labels.csv'
description = root / 'cycle08_data/NCATS_AID1645842_description.json'
mapping_path = source / 'ncats_candidate_identity.csv'
mapping = pd.read_csv(mapping_path)[['candidate_id', 'analysis_id']]
curves = mapping.merge(pd.read_csv(raw, low_memory=False), on='candidate_id', validate='one_to_many')
curves = curves[curves.curve_reason.str.startswith('good_inhibition_', na=False)].copy()
fields = json.loads(description.read_text(encoding='utf-8'))['PC_AssayContainer'][0]['assay']['descr']['results']
dosemap = {f['name'].split('-Replicate_')[0]: f['tc']['concentration']
           for f in fields if 'tc' in f and f['name'].endswith('Replicate_1')}
cols, doses = list(dosemap), np.array(list(dosemap.values()))
lo, hi = [], []
for _, row in curves.iterrows():
    tested = doses[row[cols].notna().to_numpy()]
    exclusions = np.array(str(row.Excluded_Points).split(), int)
    included = tested[exclusions == 0] if len(tested) == len(exclusions) else np.array([])
    lo.append(included.min() if len(included) else np.nan)
    hi.append(included.max() if len(included) else np.nan)
curves['min_included_uM'], curves['max_included_uM'] = lo, hi
curves['fitted_AC50_uM'] = 10 ** (6 - curves.pIC50)
curves['in_dose_window'] = curves.fitted_AC50_uM.between(curves.min_included_uM, curves.max_included_uM)
curves['window_reason'] = np.where(curves.min_included_uM.isna(), 'exclusion_mask_ambiguous',
                                  np.where(curves.in_dose_window, 'within_observed_window', 'outside_observed_window'))
curves.to_csv(out / 'ncats_qualified_curve_window_audit.csv', index=False)
curves = curves[curves.in_dose_window]
targets = curves.groupby('analysis_id').pIC50.agg(['median', 'size', 'min', 'max'])
targets.columns = ['median_pIC50', 'n_supported_curves', 'min_pIC50', 'max_pIC50']
frames = {}
inputs = [out / 'ncats_qualified_curve_window_audit.csv', source / 'ncats_adjudicated_labels.csv', raw, description, mapping_path]
for name in ['mol2mol', 'chemgpt']:
    path = source / name / 'continuous_locked_scores.csv'
    manifest = path.parent / 'continuous_score_manifest.json'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == json.loads(manifest.read_text())['score_hash']
    frames[name] = pd.read_csv(path).set_index('candidate_id')
    inputs.extend([path, manifest])
a, b = frames['mol2mol'], frames['chemgpt']
pd.testing.assert_frame_equal(a[['identity', 'cont_fp_point', 'cont_fp_interval']], b[['identity', 'cont_fp_point', 'cont_fp_interval']])
d = a[['identity', 'cont_fp_point', 'cont_fp_interval']].copy()
d['mol2mol'] = a.cont_fused_interval
d['chemgpt'] = b.cont_fused_interval
d = d.join(pd.read_csv(inputs[1]).set_index('candidate_id'), validate='one_to_one').join(targets, how='inner', validate='one_to_one')
d.index.name = 'candidate_id'
d['tie'] = d.identity.map(lambda s: hashlib.sha256(s.encode()).hexdigest())
assert len(d) == 1765 and len(curves) == 1907
methods = ['cont_fp_point', 'cont_fp_interval', 'mol2mol', 'chemgpt']
assert np.isfinite(d[methods + ['median_pIC50']].to_numpy()).all()
k = len(d) * 20 // 100
selected = {m: d.index.isin(d.sort_values([m, 'tie'], ascending=[False, True]).head(k).index) for m in methods}
mae = {m: float((d[m] - d.median_pIC50).abs().mean()) for m in methods}
rows = [dict(method=m, n_cohort=len(d), MAE=mae[m], slots=k,
             hits=int((selected[m] & d.hit.eq(1)).sum()),
             unknown=int((selected[m] & d.hit.isna()).sum())) for m in methods]
pairs = []
for x, y in combinations(methods, 2):
    delta = selected[x].astype('int8') - selected[y].astype('int8')
    net = int((delta * d.hit.eq(1)).sum())
    lower = net - int(((delta == -1) & d.hit.isna()).sum())
    upper = net + int(((delta == 1) & d.hit.isna()).sum())
    pairs.append(dict(method_x=x, method_y=y, MAE_x_minus_y=mae[x]-mae[y],
                      observed_hit_x_minus_y=net, unknown_lower=lower, unknown_upper=upper,
                      descriptive_reversal=(mae[x]-mae[y])*net > 0))
pd.DataFrame(rows).to_csv(out / 'same_cohort_summary.csv', index=False)
pd.DataFrame(pairs).to_csv(out / 'same_cohort_pairs.csv', index=False)
for m in methods:
    d['selected_' + m] = selected[m]
d.to_csv(out / 'same_cohort_candidate_results.csv')
audit = dict(n_cohort=len(d), n_curves=len(curves), budget_fraction=0.2, slots=k,
             label_counts=d.hit.value_counts(dropna=False).to_json(),
             inputs=[dict(path=f.relative_to(root).as_posix(), sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in inputs])
(out / 'same_cohort_provenance.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
print(pd.DataFrame(rows).to_string(index=False))
print(pd.DataFrame(pairs).to_string(index=False))
print('label counts:', audit['label_counts'])
