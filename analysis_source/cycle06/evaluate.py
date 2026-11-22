import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import average_precision_score

root=Path(__file__).resolve().parent

def evaluate(project):
    base=root/project
    score=base/'locked_scores.csv'
    manifest=json.loads((base/'score_manifest.json').read_text())
    assert hashlib.sha256(score.read_bytes()).hexdigest()==manifest['score_hash']
    pred=pd.read_csv(score)
    late=pd.read_csv(base/'late_labels.csv')
    d=pred.merge(late,on='candidate_id',validate='one_to_one')
    assert len(d)==len(pred)
    d['scaffold']=d.identity.map(lambda x:MurckoScaffold.MurckoScaffoldSmiles(smiles=x) or x)
    d['tie']=d.identity.map(lambda x:hashlib.sha256(x.encode()).hexdigest())
    d['confirmed']=d.hit.eq(1)
    d['unknown']=d.hit.isna()
    methods=pred.columns[2:].tolist()
    metrics=[]
    comparisons=[]
    shells={'all_candidates':d}
    lookup=root.parent/'cycle02'/'chembl28_structure_overlap.csv'
    if lookup.exists() and (project=='pxr' or project.startswith('cyp')):
        era=pd.read_csv(lookup)
        ref='pxr' if project=='pxr' else 'cyp2c9'
        exposed=set(era.loc[era.project.eq(ref)&era.chembl28_connectivity_ids.notna(),'candidate_id'])
        shells['no_chembl28_connectivity']=d[~d.candidate_id.isin(exposed)].copy()
    for mode,sub in shells.items():
        picks={}
        for method in methods:
            order=sub.sort_values([method,'tie'],ascending=[False,True])
            picks[method]=(order.groupby('phase').cumcount()<order.groupby('phase').phase.transform('size')//5).reindex(sub.index)
            chosen=sub[picks[method]]
            known=sub[~sub.unknown]
            metrics.append(dict(mode=mode,method=method,candidates=len(sub),slots=len(chosen),
                                confirmed_hits=int(chosen.confirmed.sum()),unknown=int(chosen.unknown.sum()),
                                ap=float(average_precision_score(known.confirmed,known[method])),
                                measured_candidates=len(known),confirmed_pool=int(known.confirmed.sum())))
        a=picks['fused_soft']
        n=int(a.sum())
        for baseline in [m for m in methods if m!='fused_soft']:
            b=picks[baseline]
            diff=a.astype(int)-b.astype(int)
            net=int((diff*sub.confirmed).sum())
            grouped=pd.DataFrame({'slots':a.astype(int),'net':diff*sub.confirmed,
                                  'scaffold':sub.scaffold}).groupby('scaffold').sum().to_numpy()
            rng=np.random.default_rng(20261120)
            boot=grouped[rng.integers(0,len(grouped),(10000,len(grouped)))].sum(axis=1)
            lo,hi=np.quantile(100*boot[:,1]/boot[:,0],[.025,.975])
            comparisons.append(dict(mode=mode,baseline=baseline,slots=n,net=net,delta_pct=100*net/n,
                                    ci_low=lo,ci_high=hi,
                                    unknown_min=net-int((sub.unknown&diff.eq(-1)).sum()),
                                    unknown_max=net+int((sub.unknown&diff.eq(1)).sum())))
        if mode=='all_candidates':
            for method, mask in picks.items():
                d['select_'+method]=mask
    pd.DataFrame(metrics).to_csv(base/'metrics.csv',index=False)
    pd.DataFrame(comparisons).to_csv(base/'comparisons.csv',index=False)
    d.to_csv(base/'evaluated.csv',index=False)
    print(project,pd.DataFrame(metrics).query("mode=='all_candidates'")[['method','slots','confirmed_hits','unknown','ap']]
          .to_string(index=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('projects',nargs='+')
    for project in p.parse_args().projects:
        evaluate(project)
