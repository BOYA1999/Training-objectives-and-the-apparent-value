from pathlib import Path
import hashlib, json, sys
import numpy as np
import pandas as pd
from rdkit.Chem.Scaffolds import MurckoScaffold

here=Path(__file__).resolve().parent
base=here.parent
sys.path.insert(0,str(base/'cycle06'))
from prepare import standardize

selection = {
 'pxr': [('gashaw','saglam')],
 'cyp1a2': [('05_auxiliary_data__20260915','07_placement__20260916_pubchem_r2'),
            ('05_auxiliary_data__20260915','31_interval_loss__20261026')],
 'cyp2c9': [('07_placement__20260916_pubchem_r2','03_methods__20260914')],
 'cyp3a4': [('22_pooled_seed_ensemble__20261023','31_interval_loss__20261026_gated'),
            ('Scigantic__data__submissions__R25','gashawmg__artifacts__submission_splice1a2'),
            ('31_interval_loss__20261026_gated','gashawmg__artifacts__submission_raw3a4')],
}
cyp_candidates = pd.read_csv(base/'cycle05_data'/'cyp-challenge-TEST-BLINDED.csv').rename(columns={'Molecule_Name':'candidate_id'})
pxr_candidates = pd.read_csv(base/'cycle02_data_b'/'pxr-challenge_TEST_BLINDED.csv').rename(columns={'Molecule Name':'candidate_id'})
rows=[]
for project, pairs in selection.items():
    d=pd.read_csv(base/'cycle06'/project/'late_labels.csv').merge(
        pxr_candidates if project=='pxr' else cyp_candidates,on='candidate_id',validate='one_to_one')
    if project=='pxr':
        d=d[d.phase.eq(2)]
    d=d.reset_index(drop=True)
    d['identity']=d.SMILES.map(standardize)
    assert d.identity.is_unique
    d['scaffold']=d.identity.map(lambda s:MurckoScaffold.MurckoScaffoldSmiles(smiles=s) or s)
    d['tie']=d.identity.map(lambda s:hashlib.sha256(s.encode()).hexdigest())
    d=d.set_index('candidate_id')
    for a,b in pairs:
        def read(name):
            if project=='pxr':
                return pd.read_csv(base/'cycle05'/f'{name}_final.csv').set_index('Molecule Name').pEC50.reindex(d.index)
            local=here/f'{name}.csv'
            source=local if local.is_file() else base/'cycle08_independent'/local.name
            return pd.read_csv(source).set_index('Molecule_Name')[project.upper()+'_pIC50_direct_inhibition'].reindex(d.index)
        y,z=read(a),read(b)
        assert y.notna().all() and z.notna().all()
        if project=='pxr':
            fit=d.point.notna()
            la=(y-d.point).abs()
            lb=(z-d.point).abs()
            loss_base=pd.Series(1.,index=d.index)
        else:
            fit=d[['point','lower','upper']].notna().all(axis=1)
            la=(y-d.upper).clip(lower=0)+(d.lower-y).clip(lower=0)
            lb=(z-d.upper).clip(lower=0)+(d.lower-z).clip(lower=0)
            m=d.loc[fit,'point'].mean()
            loss_base=(m-d.upper).clip(lower=0)+(d.lower-m).clip(lower=0)
        for percent in (10,20,30):
            k=len(d)*percent//100
            y_list=set(d.assign(score=y).sort_values(['score','tie'],ascending=[False,True]).head(k).index)
            z_list=set(d.assign(score=z).sort_values(['score','tie'],ascending=[False,True]).head(k).index)
            yl=pd.Series(d.index.isin(y_list),index=d.index)
            zl=pd.Series(d.index.isin(z_list),index=d.index)
            swap=yl.astype('int8')-zl.astype('int8')
            verified=d.hit.eq(1)
            unresolved=d.hit.isna()
            h=swap*verified
            group=pd.DataFrame({'scaffold':d.scaffold,'loss':(la-lb).where(fit,0),
                                'ref':loss_base.where(fit,0),'gain':h,'budget':yl.astype(int)}).groupby('scaffold').sum()
            rng=np.random.default_rng(20261120)
            draws=group.to_numpy()[rng.integers(len(group),size=(20000,len(group)))].sum(axis=1)
            lerr=np.quantile(draws[:,0]/draws[:,1],[.025,.975])
            herr=np.quantile(100*draws[:,2]/draws[:,3],[.025,.975])
            net=int(h.sum())
            rows.append(dict(project=project,a=a,b=b,candidate_n=len(d),clusters=len(group),percent=percent,slots=k,
                             loss_a=float(la[fit].sum()/loss_base[fit].sum()),loss_b=float(lb[fit].sum()/loss_base[fit].sum()),
                             lower_loss_diff=lerr[0],upper_loss_diff=lerr[1],hits_a=int((yl&verified).sum()),hits_b=int((zl&verified).sum()),
                             net=net,lower_net=net-int((unresolved&swap.eq(-1)).sum()),upper_net=net+int((unresolved&swap.eq(1)).sum()),
                             low_hit_diff_per100=herr[0],high_hit_diff_per100=herr[1]))
out=pd.DataFrame(rows)
out.to_csv(here/'cross_benchmark_examples.csv',index=False)
print(out.query("percent==20").round(4).to_string(index=False))

