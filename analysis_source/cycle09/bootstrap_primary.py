from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from rdkit.Chem.Scaffolds import MurckoScaffold

here=Path(__file__).resolve().parent
base=here.parent
rows=[]
for assay in ('cyp2c9','pxr'):
    s=base/'cycle06'/assay
    x=pd.read_csv(s/'continuous_locked_scores.csv')[['candidate_id','identity','cont_fp_point','cont_fp_interval','cont_fused_interval']]
    c=base/'cycle07'/'chemgpt'/assay/'continuous_locked_scores.csv'
    x=x.merge(pd.read_csv(c)[['candidate_id','cont_fused_interval']].rename(columns={'cont_fused_interval':'ChemGPT'}),validate='one_to_one')
    x=x.merge(pd.read_csv(here/f'{assay}_locked_point_l2.csv')[['candidate_id','fp_point_l2']],validate='one_to_one')
    x=x.merge(pd.read_csv(s/'late_labels.csv'),validate='one_to_one')
    x['tie']=x.identity.map(lambda z:hashlib.sha256(z.encode()).hexdigest())
    x['scaffold']=x.identity.map(lambda z:MurckoScaffold.MurckoScaffoldSmiles(smiles=z) or z)
    hit=x.hit.eq(1).astype(int)
    choices={}
    for col in ('cont_fp_point','fp_point_l2','cont_fp_interval','cont_fused_interval','ChemGPT'):
        v=x.sort_values([col,'tie'],ascending=[False,True])
        group=v.groupby('phase').phase
        chosen=v.index[group.cumcount()<group.transform('size')//5]
        choices[col]=pd.Series(x.index.isin(chosen).astype(int),index=x.index)
    pairs=[('fp_point_l2','cont_fp_point'),('cont_fp_interval','fp_point_l2')]
    pairs.extend((model,control) for model in ('cont_fused_interval','ChemGPT') for control in ('fp_point_l2','cont_fp_interval'))
    coalesced=pd.DataFrame({k:(choices[a]-choices[b])*hit for k,(a,b) in enumerate(pairs)}).groupby(x.scaffold).sum()
    rng=np.random.default_rng(20261120)
    frequency=rng.multinomial(len(coalesced),np.full(len(coalesced),1/len(coalesced)),size=15000)
    deltas=frequency@coalesced.to_numpy()
    k=int(choices['cont_fp_point'].sum())
    assert all(v.sum()==k for v in choices.values())
    q=np.quantile(100*deltas/k,[.025,.975],axis=0)
    rows.extend(dict(assay=assay,method=a,comparator=b,budget=k,net=int(coalesced[kp].sum()),
                     low_pct=q[0,kp],high_pct=q[1,kp],scaffolds=len(coalesced)) for kp,(a,b) in enumerate(pairs))
out=pd.DataFrame(rows)
out.to_csv(here/'scaffold_bootstrap.csv',index=False)
print(out.round(4).to_string(index=False))

