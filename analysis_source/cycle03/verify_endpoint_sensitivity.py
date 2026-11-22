from pathlib import Path
import pandas as pd
from sklearn.metrics import average_precision_score

root = Path(__file__).resolve().parent
d = pd.read_csv(root/'evaluated_candidates.csv')
lab = pd.read_csv(root.parent/'cycle02_data_b'/'labels'/'pxr_test_labels.csv')
d = d.merge(lab[['candidate_id','pec50']],on='candidate_id',validate='one_to_one')
d['point_label'] = d.pec50.ge(6).where(d.pec50.notna())
rows = []
for group, s in [('all',d),*d.groupby('phase')]:
    for name in [x.removesuffix('_chosen') for x in d if x.endswith('_chosen')]:
        selected=s[s[name+'_chosen']]
        known=s[s.hit.notna()]
        point=s[s.point_label.notna()]
        rows.append(dict(phase=group,method=name,
                         estimated_positive=int(selected.point_label.eq(True).sum()),
                         confirmed_positive=int(selected.hit.eq(1).sum()),
                         point_based_only=int((selected.point_label.eq(True)&selected.hit.isna()).sum()),
                         confirmed_AP=average_precision_score(known.hit,known[name]),
                         point_AP=average_precision_score(point.point_label,point[name])))
out=pd.DataFrame(rows)
out.to_csv(root/'endpoint_sensitivity.csv',index=False)
print(out.to_string(index=False))
print('All full point positive',int(d.point_label.eq(True).sum()),'confirmed',int(d.hit.eq(1).sum()))
