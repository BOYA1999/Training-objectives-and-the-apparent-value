import sys
sys.dont_write_bytecode = True
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
lock = json.loads((HERE/'prediction_lock.json').read_text())
score_path = HERE/'locked_predictions.csv'
assert hashlib.sha256(score_path.read_bytes()).hexdigest()==lock['sha256']
scores = pd.read_csv(score_path).set_index('candidate_id')
raw_path = ROOT/'cycle05_data/cyp-challenge-TEST-UNBLINDED_inhibition.csv'
raw = pd.read_csv(raw_path).set_index('Molecule_Name')
assert raw.index.is_unique and scores.index.is_unique and set(scores.index)==set(raw.index) and len(scores)==750
raw = raw.loc[scores.index]
stem = 'CYP2C9_pIC50_direct_inhibition'
point,lower,upper = raw[stem],raw[stem+'_conf_low'],raw[stem+'_conf_high']
valid = lower.le(point)&point.le(upper)
hit = pd.Series(np.select([valid&lower.ge(6),valid&upper.lt(6)],[1.,0.],default=np.nan),index=scores.index)
previous_labels = pd.read_csv(ROOT/'cycle06/cyp2c9/late_labels.csv').set_index('candidate_id').loc[scores.index]
np.testing.assert_equal(hit.to_numpy(),previous_labels.hit.to_numpy())
tie = scores.identity.map(lambda s:hashlib.sha256(s.encode()).hexdigest())
cells,selected = [],{}
old = json.loads((ROOT/'cycle09/verification.json').read_text())['cyp2c9_hits_from_raw_public_labels']
for objective in ['point_l1','point_l2','interval']:
    for j,representation in enumerate(['FP','Mol2Mol','ChemGPT']):
        name = f'{representation}_{objective}'
        ranking = pd.DataFrame({'score':scores[name],'tie':tie}).sort_values(['score','tie'],ascending=[False,True])
        selected[name] = ranking.index[:150]
        h = hit.loc[selected[name]]
        cells.append(dict(objective=objective,representation=representation,slots=150,
            confirmed=int(h.eq(1).sum()),point_hits=int(point.loc[h.index].ge(6).sum()),unknown=int(h.isna().sum()),
            confirmed_feature_fraction_07=old[objective][j],boundary_tie=bool(ranking.score.iloc[149]==ranking.score.iloc[150])))
        ranking.head(150).assign(hit=h).to_csv(HERE/f'selected_{name}.csv')
old_pairs = pd.read_csv(ROOT/'cycle09/comparison_matrix.csv')
comparison_names={'point_l1':'unmatched_L1_point','point_l2':'unmatched_L2_point','interval':'matched_interval'}
pairs=[]
for model in ['Mol2Mol','ChemGPT']:
    for comparator in ['point_l1','point_l2','interval']:
        a,b=selected[f'{model}_interval'],selected[f'FP_{comparator}']
        ha,hb=int(hit.loc[a].eq(1).sum()),int(hit.loc[b].eq(1).sum())
        ua,ub=int(hit.loc[a.difference(b)].isna().sum()),int(hit.loc[b.difference(a)].isna().sum())
        oldrow=old_pairs[(old_pairs.assay=='cyp2c9')&(old_pairs.model==model)&(old_pairs.budget_percent==20)&(old_pairs.endpoint=='assay_supported')&(old_pairs.comparison==comparison_names[comparator])].iloc[0]
        pairs.append(dict(model=model,fp_comparator=comparator,model_hits=ha,fp_hits=hb,net=ha-hb,
            lower=ha-hb-ub,upper=ha-hb+ua,model_only_unknown=ua,fp_only_unknown=ub,exclusive_each=len(a.difference(b)),
            old_net=int(oldrow.delta),old_lower=int(oldrow.lower),old_upper=int(oldrow.upper)))
cells,pairs=pd.DataFrame(cells),pd.DataFrame(pairs)
assert not cells.boundary_tie.any()
cells.to_csv(HERE/'nine_cells.csv',index=False)
pairs.to_csv(HERE/'interval_fusion_pairs.csv',index=False)
summary=dict(evaluated_utc=datetime.now(timezone.utc).isoformat(),prediction_lock_utc=lock['locked_utc'],
    predictions_sha256=lock['sha256'],raw_labels_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
    labels_match_prior=True,slots=150,unique_candidates=750,unique_structures=int(scores.identity.nunique()),
    raw_supported=int(hit.eq(1).sum()),raw_nonhit=int(hit.eq(0).sum()),raw_unknown=int(hit.isna().sum()),
    no_boundary_ties=True,only_training_parameter_changed='feature_fraction:0.7->1.0',
    bounds_interpretation='fixed-catalogue unresolved-label partial-identification bounds; not confidence intervals')
(HERE/'evaluation_manifest.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(cells.to_string(index=False))
print(pairs.to_string(index=False))
