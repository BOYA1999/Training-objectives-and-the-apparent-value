import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import average_precision_score

root = Path(__file__).resolve().parent
path = root/'locked_scores.csv'
lock = json.loads((root/'score_lock.json').read_text())
assert hashlib.sha256(path.read_bytes()).hexdigest()==lock['score_sha256']
score = pd.read_csv(path)
labels = pd.read_csv(root.parent/'cycle02_data_b'/'labels'/'pxr_test_labels.csv')
d = score.merge(labels[['candidate_id','phase','hit','endpoint_status']],on='candidate_id',validate='one_to_one')
assert len(d)==len(score)
d['scaffold'] = d.identity.map(lambda s:MurckoScaffold.MurckoScaffoldSmiles(smiles=s) or s)
d['tiebreak'] = d.identity.map(lambda s:hashlib.sha256(s.encode()).hexdigest())
d['confirmed'] = d.hit.eq(1)
d['unknown'] = d.hit.isna()
names=[m for m in score if m not in ('candidate_id','smiles','identity')]
for name in names:
    ordered=d.sort_values([name,'tiebreak'],ascending=[False,True])
    d[name+'_chosen']=(ordered.groupby('phase').cumcount()<ordered.groupby('phase').hit.transform('size')//5).reindex(d.index)

metrics=[]
for phase,g in [('all',d),*d.groupby('phase')]:
    for name in names:
        sel=g[g[name+'_chosen']]
        determined=g.dropna(subset=['hit'])
        metrics.append(dict(phase=phase,model=name,n=len(g),positive_n=int(g.confirmed.sum()),unknown_n=int(g.unknown.sum()),
                         slots=len(sel),confirmed=int(sel.confirmed.sum()),unknown=int(sel.unknown.sum()),
                         AP=average_precision_score(determined.hit,determined[name])))
pd.DataFrame(metrics).to_csv(root/'metrics.csv',index=False)
d.to_csv(root/'evaluated_candidates.csv',index=False)
pairwise=[]
for reference in ('fingerprint_NN','ECFP_NN','Feature_NN','Tanimoto_SVM','ECFP_RF','permuted_AI_NN'):
    act = d.AI_NN_chosen
    ctl = d[reference+'_chosen']
    diff = act.astype(int)-ctl.astype(int)
    paired = pd.DataFrame({'scaffold':d.scaffold,'net':diff*d.confirmed,'slots':act.astype(int)})
    groups=paired.groupby('scaffold')[['net','slots']].sum().to_numpy()
    rng=np.random.default_rng(20261120)
    z=groups[rng.integers(0,len(groups),(20000,len(groups)))].sum(axis=1)
    interval=np.quantile(100*z[:,0]/z[:,1],[.025,.975])
    net=int(paired.net.sum())
    pairwise.append(dict(reference=reference,delta_hits=net,slots=int(act.sum()),scaffold_groups=len(groups),
                         delta_pct=100*net/act.sum(),ci_low_pct=interval[0],ci_high_pct=interval[1],
                         delta_min_unknown=net-int((diff.eq(-1)&d.unknown).sum()),
                         delta_max_unknown=net+int((diff.eq(1)&d.unknown).sum())))
pd.DataFrame(pairwise).to_csv(root/'comparisons.csv',index=False)
print(pd.DataFrame(metrics).to_string(index=False))
print(pd.DataFrame(pairwise).to_string(index=False))
