from pathlib import Path
import numpy as np
import pandas as pd

root=Path(__file__).resolve().parent
ranked=pd.read_csv(root/'evaluated_candidates.csv')
lab=pd.read_csv(root.parent/'cycle02_data_b'/'labels'/'pxr_test_labels.csv')
d=ranked.merge(lab[['candidate_id','pec50','pec50_ci_lower','pec50_ci_upper',
                   'drc_min_observed_pconc','drc_max_observed_pconc']],
               on='candidate_id',validate='one_to_one')
point=d.pec50.ge(6)
ci_support=d.pec50_ci_lower.ge(6)
point_in_window=d.pec50.between(d.drc_min_observed_pconc,d.drc_max_observed_pconc)
interval_in_window=d.pec50_ci_lower.ge(d.drc_min_observed_pconc)&d.pec50_ci_upper.le(d.drc_max_observed_pconc)

recipes={'point_ge6':point,'point_ge6_in_window':point&point_in_window,
         'CI_above6':point&ci_support,'CI_above6_inside_dose_window':point&ci_support&interval_in_window}
assert recipes['CI_above6_inside_dose_window'].equals(d.hit.eq(1))
rows=[]
for phase,g in [('all',d),*d.groupby('phase')]:
    for rule, positives in recipes.items():
        for model in ['AI_NN','Feature_NN','fingerprint_NN','Tanimoto_SVM','permuted_AI_NN']:
            selected=g[model+'_chosen']
            rows.append(dict(phase=phase,rule=rule,models=model,
                             all_signals=int(positives[g.index].sum()),selected=int(selected.sum()),
                             positives_in_shortlist=int(positives[g.index][selected].sum())))
table=pd.DataFrame(rows)
table.to_csv(root/'assay_label_decision_audit.csv',index=False)
print(table[table.phase.eq('all')].pivot(index='rule',columns='models',values='positives_in_shortlist').to_string())
for k,mask in recipes.items():
    print(k, int(mask.sum()))
