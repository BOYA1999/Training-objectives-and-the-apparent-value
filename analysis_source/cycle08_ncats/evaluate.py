from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from rdkit.Chem.Scaffolds import MurckoScaffold

here = Path(__file__).resolve().parent
base = here.parent
curves = pd.read_csv(base/'cycle08_data'/'NCATS_CYP2C9_fixed_curve_labels.csv',
                     usecols=['candidate_id','curve_label_pIC50_6'])
mapping = pd.read_csv(here/'ncats_candidate_identity.csv')
linked = mapping.merge(curves, on='candidate_id',validate='one_to_many')
assert len(linked.analysis_id.unique())==mapping.analysis_id.nunique()
labels = linked.groupby('analysis_id').curve_label_pIC50_6.agg(['size','count','nunique','first'])
labels['hit']=labels['first'].where(labels['size'].eq(labels['count'])&labels['nunique'].eq(1))
scores={}
for model in ['mol2mol','chemgpt']:
    path=here/model/'continuous_locked_scores.csv'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==json.loads((path.parent/'continuous_score_manifest.json').read_text())['score_hash']
    scores[model]=pd.read_csv(path).set_index('candidate_id')
a,b=scores['mol2mol'],scores['chemgpt']
pd.testing.assert_frame_equal(a[['identity','cont_fp_interval','cont_fp_point']], b[['identity','cont_fp_interval','cont_fp_point']])
d=a[['identity','cont_fp_interval','cont_fp_point']].join(labels.hit,validate='one_to_one')
assert len(d)==len(labels)
d=d.assign(mol2mol=a.cont_fused_interval,chemgpt=b.cont_fused_interval,
           mol2mol_rank=a.cont_rank_fusion,chemgpt_rank=b.cont_rank_fusion,
           tie=d.identity.map(lambda s:hashlib.sha256(s.encode()).hexdigest()),
           scaffold=d.identity.map(lambda s:MurckoScaffold.MurckoScaffoldSmiles(smiles=s) or s))
print('NCATS strict retained source observed counts:',d.hit.value_counts(dropna=False).to_dict())
d[['hit']].to_csv(here/'ncats_adjudicated_labels.csv')
d['positive']=d.hit.eq(1)
d['unknown']=d.hit.isna()
out=[]
for p in (10,20,30):
    k=len(d)*p//100
    methods=['cont_fp_point','cont_fp_interval','mol2mol','chemgpt','mol2mol_rank','chemgpt_rank']
    chosen={method:d.index.isin(d.sort_values([method,'tie'],ascending=[False,True]).head(k).index) for method in methods}
    for method in methods:
        v=chosen[method]
        q=chosen['cont_fp_interval']
        swap=v.astype('int8')-q.astype('int8')
        delta=int((swap*d.positive).sum())
        bins=pd.DataFrame({'scaffold':d.scaffold,'d':swap*d.positive,'b':v}).groupby('scaffold').sum().to_numpy()
        rng=np.random.default_rng(20261120)
        totals=bins[rng.integers(len(bins),size=(8000,len(bins)))].sum(axis=1)
        lo,hi=np.quantile(100*totals[:,0]/totals[:,1],[.025,.975])
        out.append(dict(budget_percent=p,slots=k,method=method,hits=int((d.positive&v).sum()),unknown=int((d.unknown&v).sum()),
                        net=delta,net_lower=delta-int((d.unknown&(swap==-1)).sum()),
                        net_upper=delta+int((d.unknown&(swap==1)).sum()),bootstrap_low_pct=lo,bootstrap_high_pct=hi))
x=pd.DataFrame(out)
x.to_csv(here/'ncats_external_metrics.csv',index=False)
print(x.query('budget_percent==20').round(3).to_string(index=False))

