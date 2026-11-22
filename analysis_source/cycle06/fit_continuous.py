import argparse
import hashlib
import json
import time
from pathlib import Path

from fit import FrozenMol2MolEncoder, Chem, descriptors, fcfp, lgb, morgan, np, params, pd

root=Path(__file__).resolve().parent
def fit_project(name,encoder):
    out=root/name
    start=time.perf_counter()
    struct=pd.read_csv(out/'structures.csv')
    early=pd.read_csv(out/'early_labels.csv')
    train=struct.query("split == 'train'").merge(early,on='candidate_id',validate='one_to_one')
    test=struct.query("split == 'test'").reset_index(drop=True)
    allmols=pd.concat([train,test],ignore_index=True)
    mols=[Chem.MolFromSmiles(s) for s in allmols.identity]
    fp=np.concatenate((np.array([morgan.GetFingerprintAsNumPy(m) for m in mols]),
                       np.array([fcfp.GetFingerprintAsNumPy(m) for m in mols]),
                       np.array([[f(m) for f in descriptors] for m in mols])),axis=1,dtype=np.float32)
    emb,track=encoder.encode(allmols.rename(columns={'identity':'smiles'}))
    assert track.candidate_id.equals(allmols.candidate_id)
    vec=emb['mean']
    nz=len(train)
    rng=np.random.default_rng(20261120)
    perm=np.concatenate([rng.permutation(nz),nz+rng.permutation(len(test))])
    mu=float(train.point.mean())
    y=train.point.to_numpy()-mu
    lo=train.lower.to_numpy()-mu
    hi=train.upper.to_numpy()-mu
    def objective(raw,dataset):
        err=np.where(raw<lo,raw-lo,np.maximum(raw-hi,0))
        return err+0.1*(raw-y), 0.1+((raw<lo)|(raw>hi)).astype(float)
    modes=[
        ('cont_fp_point',fp,'regression_l1'),
        ('cont_fused_point',np.hstack([fp,vec]),'regression_l1'),
        ('cont_fp_interval',fp,objective),
        ('cont_latent_interval',vec,objective),
        ('cont_fused_interval',np.hstack([fp,vec]),objective),
        ('cont_shuffled_interval',np.hstack([fp,vec[perm]]),objective),
    ]
    output=test[['candidate_id','identity']].copy()
    for method,x,obj in modes:
        model=lgb.train({**params,'objective':obj}, lgb.Dataset(x[:nz],label=y),num_boost_round=300)
        output[method]=model.predict(x[nz:],num_threads=2,raw_score=True)+mu
    output['cont_rank_fusion']=(output.cont_fp_interval.rank(pct=True)+output.cont_latent_interval.rank(pct=True))/2
    assert np.isfinite(output.iloc[:,2:].to_numpy()).all()
    dest=out/'continuous_locked_scores.csv'
    output.to_csv(dest,index=False)
    log=dict(project=name,n_train=nz,n_test=len(test),elapsed=time.perf_counter()-start,
             score_hash=hashlib.sha256(dest.read_bytes()).hexdigest())
    (out/'continuous_score_manifest.json').write_text(json.dumps(log,indent=2),encoding='utf-8')
    print(json.dumps(log),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('projects',nargs='+')
    names=p.parse_args().projects
    encoder=FrozenMol2MolEncoder()
    for name in names:
        fit_project(name,encoder)
