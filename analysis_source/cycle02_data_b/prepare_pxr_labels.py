import json
from pathlib import Path
import numpy as np
import pandas as pd

b = Path(__file__).resolve().parent
out = b / 'labels'
out.mkdir(exist_ok=True)
assert (b / 'PXR_DATA_CONTRACT.md').exists()
summary = {}
for split in ['train', 'test']:
    d = pd.read_csv(b / f'pxr_prepared_{split}.csv').rename(columns={'compound_id': 'candidate_id'})
    if 'phase' not in d:
        d['phase'] = 0
    p, lo, hi = d.pec50, d.pec50_ci_lower, d.pec50_ci_upper
    a, z = d.drc_min_observed_pconc, d.drc_max_observed_pconc
    finite = np.isfinite(d[['pec50', 'pec50_ci_lower', 'pec50_ci_upper', 'drc_min_observed_pconc', 'drc_max_observed_pconc']]).all(axis=1)
    valid = finite & (a <= z) & (lo <= p) & (p <= hi)
    inside = valid & (lo >= a) & (hi <= z)
    yes, no = inside & (lo >= 6), inside & (hi < 6)
    d['hit'] = pd.Series(pd.NA, index=d.index, dtype='Int64')
    d.loc[yes, 'hit'], d.loc[no, 'hit'] = 1, 0
    d['endpoint_status'] = np.select([~finite, ~valid, ~inside, yes, no], ['unknown_missing', 'unknown_invalid_interval', 'unknown_outside_dose_window', 'quantified_threshold_hit', 'quantified_below_threshold'], default='unknown_threshold_overlap')
    cols = ['candidate_id', 'hit', 'endpoint_status', 'phase', 'pec50', 'pec50_ci_lower', 'pec50_ci_upper', 'pec50_se', 'drc_min_observed_pconc', 'drc_max_observed_pconc', 'emax_relative', 'emax_relative_ci_lower', 'emax_relative_ci_upper']
    d[cols].to_csv(out / f'pxr_{split}_labels.csv', index=False)
    assert d.candidate_id.is_unique and not (yes & no).any()
    counts = lambda x: {'n': len(x), 'hit': int(x.hit.eq(1).sum()), 'nonhit': int(x.hit.eq(0).sum()), 'unknown': int(x.hit.isna().sum())}
    summary[split] = counts(d)
    summary[split]['statuses'] = d.endpoint_status.value_counts().to_dict()
    summary[split]['phases'] = {str(k): counts(v) for k, v in d.groupby('phase')}
    if split == 'test':
        blind = pd.read_csv(b / 'structures' / 'pxr_candidates.csv')
        assert set(d.candidate_id) == set(blind.candidate_id)
        mapped = blind.loc[blind.parent_id.notna(), 'candidate_id']
        summary[split]['mapped_parent_subset'] = counts(d[d.candidate_id.isin(mapped)])
(b / 'pxr_label_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(summary))
