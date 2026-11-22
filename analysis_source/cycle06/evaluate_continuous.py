import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from rdkit.Chem.Scaffolds import MurckoScaffold

root=Path(__file__).resolve().parent
prefix=root.parent/'cycle02'
references=pd.read_csv(prefix/'chembl28_structure_overlap.csv')
def evaluate(name):
    out=root/name
    f=out/'continuous_locked_scores.csv'
    manifest=json.loads((out/'continuous_score_manifest.json').read_text())
    assert hashlib.sha256(f.read_bytes()).hexdigest()==manifest['score_hash']
    cols=pd.read_csv(f)
    previous=pd.read_csv(out/'evaluated.csv')[['candidate_id','fp_soft','ecfp_rf','tanimoto_svm']]
    d=cols.merge(previous,on='candidate_id',validate='one_to_one').merge(
        pd.read_csv(out/'late_labels.csv'),on='candidate_id',validate='one_to_one')
    assert len(d)==len(cols)
    d['scaffold']=d.identity.map(lambda x:MurckoScaffold.MurckoScaffoldSmiles(smiles=x) or x)
    d['tie']=d.identity.map(lambda x:hashlib.sha256(x.encode()).hexdigest())
    d['verified']=d.hit.eq(1)
    d['unknown']=d.hit.isna()
    methods=cols.columns[2:].tolist()+['fp_soft','ecfp_rf','tanimoto_svm']
    ref='pxr' if name=='pxr' else 'cyp2c9'
    hits=set(references.loc[references.project.eq(ref)&references.chembl28_connectivity_ids.notna(),'candidate_id'])
    modes={'all':d,'no_chembl28_connectivity':d[~d.candidate_id.isin(hits)].copy()}
    rows=[]
    pairs=[]
    for space,block in modes.items():
        m=block[~block.unknown]
        selections={}
        for method in methods:
            ordered=block.sort_values([method,'tie'],ascending=[False,True])
            keep=(ordered.groupby('phase').cumcount()<ordered.groupby('phase').phase.transform('size')//5).reindex(block.index)
            selections[method]=keep
            z=block[keep]
            rows.append(dict(mode=space,method=method,slots=len(z),confirmed=int(z.verified.sum()),unknown=int(z.unknown.sum()),
                             ap=average_precision_score(m.verified,m[method])))
        for method in methods:
            if method=='cont_fused_interval': continue
            a=selections['cont_fused_interval']
            diff=a.astype(int)-selections[method].astype(int)
            g=pd.DataFrame(dict(scaffold=block.scaffold,delta=diff*block.verified,budget=a.astype(int))).groupby('scaffold').sum().to_numpy()
            rng=np.random.default_rng(20261120)
            draw=g[rng.integers(0,len(g),(10000,len(g)))].sum(axis=1)
            lo,hi=np.quantile(100*draw[:,0]/draw[:,1],[.025,.975])
            net=int((diff*block.verified).sum())
            pairs.append(dict(mode=space,baseline=method,slots=int(a.sum()),net=net,
                              ci_low_pct=lo,ci_high_pct=hi,lower_unknown=net-int((block.unknown&diff.eq(-1)).sum()),
                              upper_unknown=net+int((block.unknown&diff.eq(1)).sum())))
    pd.DataFrame(rows).to_csv(out/'continuous_metrics.csv',index=False)
    pd.DataFrame(pairs).to_csv(out/'continuous_comparisons.csv',index=False)
    print(name,pd.DataFrame(rows).query("mode=='all'")[['method','confirmed','slots','unknown','ap']]
          .to_string(index=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('projects',nargs='+')
    for name in p.parse_args().projects:
        evaluate(name)
