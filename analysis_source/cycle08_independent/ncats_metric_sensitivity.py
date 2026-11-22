from pathlib import Path
from itertools import combinations
import hashlib
import json
import numpy as np
import pandas as pd

out = Path(__file__).resolve().parent
base = out.parent
src = base / 'cycle08_ncats'
inputs = [base / 'cycle08_data/NCATS_CYP2C9_fixed_curve_labels.csv', src / 'ncats_candidate_identity.csv', src / 'ncats_adjudicated_labels.csv', src / 'ncats_external_metrics.csv']
curves = pd.read_csv(inputs[0], low_memory=False)
mapping = pd.read_csv(inputs[1])
assert mapping.candidate_id.is_unique
linked = mapping[['candidate_id', 'analysis_id']].merge(curves, on='candidate_id', validate='one_to_many')
good = linked[linked.curve_reason.str.startswith('good_inhibition_', na=False)].copy()
assert good.Fit_CurveClass.isin([-1.1, -1.2, -2.1, -2.2]).all()
assert np.isfinite(good.pIC50).all() and good.Efficacy.gt(65).all()
targets = good.groupby('analysis_id').pIC50.agg(['median', 'size', 'min', 'max'])
targets.columns = ['median_pIC50', 'n_qualified_curves', 'min_pIC50', 'max_pIC50']
scores = {}
for model in ['mol2mol', 'chemgpt']:
    path = src / model / 'continuous_locked_scores.csv'
    manifest = path.parent / 'continuous_score_manifest.json'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == json.loads(manifest.read_text())['score_hash']
    scores[model] = pd.read_csv(path).set_index('candidate_id')
    inputs += [path, manifest]
a, b = scores['mol2mol'], scores['chemgpt']
pd.testing.assert_frame_equal(a[['identity', 'cont_fp_point', 'cont_fp_interval']], b[['identity', 'cont_fp_point', 'cont_fp_interval']])
d = a[['identity', 'cont_fp_point', 'cont_fp_interval']].copy()
d['mol2mol'] = a.cont_fused_interval
d['chemgpt'] = b.cont_fused_interval
d = d.join(pd.read_csv(inputs[2]).set_index('candidate_id'), validate='one_to_one').join(targets, validate='one_to_one')
assert set(d.index) == set(mapping.analysis_id) and d.identity.is_unique
d['tie'] = d.identity.map(lambda s: hashlib.sha256(s.encode()).hexdigest())
k = len(d) * 20 // 100
methods = ['cont_fp_point', 'cont_fp_interval', 'mol2mol', 'chemgpt']
assert np.isfinite(d[methods].to_numpy()).all()
selected = {m: d.index.isin(d.sort_values([m, 'tie'], ascending=[False, True]).head(k).index) for m in methods}
saved = pd.read_csv(inputs[3]).query('budget_percent == 20').set_index('method')
rows, pairs = [], []
mae = {m: float((d[m] - d.median_pIC50).abs().mean()) for m in methods}
for m in methods:
    picked = selected[m]
    delta = picked.astype('int8') - selected['cont_fp_interval'].astype('int8')
    hits, unknown = int((picked & d.hit.eq(1)).sum()), int((picked & d.hit.isna()).sum())
    net = int((delta * d.hit.eq(1)).sum())
    lower, upper = net - int(((delta == -1) & d.hit.isna()).sum()), net + int(((delta == 1) & d.hit.isna()).sum())
    assert [hits, unknown, net, lower, upper] == saved.loc[m, ['hits', 'unknown', 'net', 'net_lower', 'net_upper']].astype(int).tolist()
    rows.append(dict(method=m, mae=mae[m], n_mae=int(d.median_pIC50.notna().sum()), slots=k, hits=hits, unknown=unknown, net_vs_fp_interval=net, net_lower=lower, net_upper=upper))
for x, y in combinations(methods, 2):
    delta = selected[x].astype('int8') - selected[y].astype('int8')
    net = int((delta * d.hit.eq(1)).sum())
    lower, upper = net - int(((delta == -1) & d.hit.isna()).sum()), net + int(((delta == 1) & d.hit.isna()).sum())
    pairs.append(dict(method_x=x, method_y=y, mae_x_minus_y=mae[x]-mae[y], observed_hit_x_minus_y=net, net_lower=lower, net_upper=upper, strict_metric_hit_reversal=(mae[x]-mae[y])*net > 0, unknown_allows_tie=lower <= 0 <= upper))
pd.DataFrame(rows).to_csv(out / 'ncats_metric_vs_hit_summary.csv', index=False)
pd.DataFrame(pairs).to_csv(out / 'ncats_metric_vs_hit_pairs.csv', index=False)
d.loc[d.median_pIC50.notna(), ['median_pIC50', 'n_qualified_curves', 'min_pIC50', 'max_pIC50', 'hit'] + methods].to_csv(out / 'ncats_qualified_median_targets.csv')
audit = dict(n_candidates=len(d), n_qualified_curves=len(good), n_qualified_compounds=len(targets), n_class4_in_mae=int(good.Fit_CurveClass.eq(4).sum()), median_pIC50_range=[float(targets.median_pIC50.min()), float(targets.median_pIC50.max())], qualified_compound_hit_counts=d.loc[d.median_pIC50.notna(), 'hit'].value_counts(dropna=False).to_json(), inputs=[dict(path=p.relative_to(base).as_posix(), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in inputs])
(out / 'ncats_metric_vs_hit_provenance.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
print(json.dumps(audit, indent=2))
print(pd.DataFrame(rows).to_string(index=False))
print(pd.DataFrame(pairs).to_string(index=False))
