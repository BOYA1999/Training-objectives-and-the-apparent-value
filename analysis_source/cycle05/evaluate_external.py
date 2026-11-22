import hashlib
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.metrics import average_precision_score

root=Path(__file__).resolve().parent
data=root.parent/'cycle02_data_b'
lab=pd.read_csv(data/'labels'/'pxr_test_labels.csv').query('phase == 2').copy()
structures=pd.read_csv(data/'pxr-challenge_TEST_BLINDED.csv')
structures=structures.rename(columns={'Molecule Name':'candidate_id','SMILES':'blind_smiles'})
lab=lab.merge(structures,on='candidate_id',validate='one_to_one')
assert len(lab)==260
canon=lambda s:Chem.MolToSmiles(Chem.MolFromSmiles(s))
lab['tiebreak']=lab.blind_smiles.map(lambda s:hashlib.sha256(canon(s).encode()).hexdigest())
metrics=[]
selected=[]
for f in sorted(root.glob('*_final.csv')):
    d=pd.read_csv(f).rename(columns={'Molecule Name':'candidate_id','pEC50':'prediction'})
    assert d.candidate_id.is_unique and d.prediction.notna().all()
    d=lab.merge(d,on='candidate_id',validate='one_to_one')
    assert len(d)==260
    assert d.blind_smiles.map(canon).equals(d.SMILES.map(canon))
    d=d.sort_values(['prediction','tiebreak'],ascending=[False,True])
    top=d.head(52).copy()
    top['team']=f.stem.removesuffix('_final')
    selected.append(top[['team','candidate_id','prediction','pec50','hit','endpoint_status']])
    k=d.hit.notna()
    metrics.append(dict(team=top.team.iat[0],n=len(d),slots=len(top),
                        mae=float((d.prediction-d.pec50).abs().mean()),
                        ap_point=average_precision_score(d.pec50.ge(6),d.prediction),
                        ap_determined=average_precision_score(d.loc[k,'hit'],d.loc[k,'prediction']),
                        point_hits=int(top.pec50.ge(6).sum()),confirmed_hits=int(top.hit.eq(1).sum()),
                        unknown=int(top.hit.isna().sum())))
out=pd.DataFrame(metrics)
pairs=[]
d=pd.concat(selected)
for x,y in combinations(out.to_dict('records'),2):
    a=d[d.team.eq(x['team'])].set_index('candidate_id')
    b=d[d.team.eq(y['team'])].set_index('candidate_id')
    delta=x['confirmed_hits']-y['confirmed_hits']
    raw=x['point_hits']-y['point_hits']
    lower=delta-int(b.loc[~b.index.isin(a.index),'hit'].isna().sum())
    upper=delta+int(a.loc[~a.index.isin(b.index),'hit'].isna().sum())
    pairs.append(dict(team_a=x['team'],team_b=y['team'],delta_point=raw,delta_confirmed=delta,
                      strict_flip=bool(raw*delta<0),tie_change=bool((raw==0)!=(delta==0)),lower=lower,upper=upper))
out.to_csv(root/'external_validation.csv',index=False)
d.to_csv(root/'selected_external.csv',index=False)
pair=pd.DataFrame(pairs)
pair.to_csv(root/'external_pairs.csv',index=False)
print(out.to_string(index=False)); print(pair.to_string(index=False))
