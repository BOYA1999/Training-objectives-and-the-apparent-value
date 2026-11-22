import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

root=Path(__file__).resolve().parent
sys.path.insert(0,str(root.parent/'cycle06_model'))
from frozen_encoder import FrozenMol2MolEncoder
import lightgbm as lgb
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from rdkit import DataStructs

params=dict(objective='cross_entropy',verbosity=-1,num_threads=2,num_leaves=15,
            learning_rate=0.05,min_data_in_leaf=12,feature_fraction=0.7,
            seed=20261120,deterministic=True,force_col_wise=True)
morgan=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
fcfp=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=1024,includeChirality=True,
                                               atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen())
descriptors=[Descriptors.MolWt,Descriptors.MolLogP,Descriptors.TPSA,Descriptors.NumHDonors,
             Descriptors.NumHAcceptors,Descriptors.NumRotatableBonds,Descriptors.RingCount,
             Descriptors.NumAromaticRings,Descriptors.FractionCSP3,Descriptors.HeavyAtomCount]

def fit(project,encoder):
    start=time.perf_counter()
    out=root/project
    structure=pd.read_csv(out/'structures.csv')
    early=pd.read_csv(out/'early_labels.csv')
    assert structure.candidate_id.is_unique and early.candidate_id.is_unique
    train=structure.query("split == 'train'").merge(early,on='candidate_id',validate='one_to_one')
    test=structure.query("split == 'test'").reset_index(drop=True)
    merged=pd.concat([train,test],ignore_index=True)
    mols=[Chem.MolFromSmiles(s) for s in merged.identity]
    conventional=np.concatenate((
        np.array([morgan.GetFingerprintAsNumPy(m) for m in mols]),
        np.array([fcfp.GetFingerprintAsNumPy(m) for m in mols]),
        np.array([[f(m) for f in descriptors] for m in mols]),
    ),axis=1,dtype=np.float32)
    n=len(train)
    features,emap=encoder.encode(merged.rename(columns={'identity':'smiles'}))
    assert emap.candidate_id.equals(merged.candidate_id)
    latent=features['mean']
    rng=np.random.default_rng(20261120)
    permutation=np.concatenate([rng.permutation(n),n+rng.permutation(len(test))])
    shuffled=latent[permutation]
    emap[['candidate_id','status','token_count_with_specials']].to_csv(out/'model_coverage.csv',index=False)
    results=test[['candidate_id','identity']].copy()
    runs=[
        ('fp_hard',conventional,'hard'),('fp_soft',conventional,'soft'),
        ('fused_hard',np.hstack([conventional,latent]),'hard'),
        ('fused_soft',np.hstack([conventional,latent]),'soft'),
        ('shuffled_soft',np.hstack([conventional,shuffled]),'soft'),
        ('latent_soft',latent,'soft'),
    ]
    for name,features,y in runs:
        model=lgb.train(params,lgb.Dataset(features[:n],label=train[y]),num_boost_round=300)
        results[name]=model.predict(features[n:],num_threads=2)
    baseline=lgb.train({**params,'objective':'regression_l1'},
                       lgb.Dataset(conventional[:n],label=train.point),num_boost_round=300)
    results['fp_point_regression']=baseline.predict(conventional[n:],num_threads=2)
    rf=RandomForestClassifier(n_estimators=300,min_samples_leaf=2,class_weight='balanced',
                              random_state=20261120,n_jobs=2)
    rf.fit(conventional[:n,:2048],train.hard)
    results['ecfp_rf']=rf.predict_proba(conventional[n:,:2048])[:,1]
    fp=[morgan.GetFingerprint(m) for m in mols]
    x=fp[:n]
    model=SVC(C=10,kernel='precomputed',class_weight='balanced')
    model.fit(np.array([DataStructs.BulkTanimotoSimilarity(a,x) for a in x]),train.hard)
    results['tanimoto_svm']=model.decision_function(np.array([DataStructs.BulkTanimotoSimilarity(a,x) for a in fp[n:]]))
    assert np.isfinite(results.iloc[:,2:].to_numpy()).all()
    path=out/'locked_scores.csv'
    results.to_csv(path,index=False)
    log=dict(project=project,n_train=n,n_test=len(test),hard_train_positive=int(train.hard.sum()),
             train_uncertain=int(train.soft.between(0,1,inclusive='neither').sum()),
             model_coverage=emap.status.value_counts().to_dict(),elapsed=time.perf_counter()-start,
             structure_hash=hashlib.sha256((out/'structures.csv').read_bytes()).hexdigest(),
             score_hash=hashlib.sha256(path.read_bytes()).hexdigest())
    (out/'score_manifest.json').write_text(json.dumps(log,indent=2),encoding='utf-8')
    print(json.dumps(log),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('projects',nargs='+')
    plan=parser.parse_args()
    encoder=FrozenMol2MolEncoder()
    for name in plan.projects:
        fit(name,encoder)
