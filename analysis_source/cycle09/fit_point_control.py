import hashlib
import json
from pathlib import Path
import sys
import torch

here=Path(__file__).resolve().parent
base=here.parent
sys.path[:0]=[str(base/'cycle06'),str(base/'cycle07_model')]
from fit import FrozenMol2MolEncoder, Chem, descriptors, fcfp, lgb, morgan, np, params, pd
from frozen_chemgpt import FrozenChemGPTEncoder

torch.set_num_threads(2)
encoders={'Mol2Mol':FrozenMol2MolEncoder(),'ChemGPT':FrozenChemGPTEncoder()}
for assay in ('cyp2c9','pxr'):
    src=base/'cycle06'/assay
    registry=pd.read_csv(src/'structures.csv')
    early=pd.read_csv(src/'early_labels.csv')
    train=registry.query("split=='train'").merge(early,validate='one_to_one')
    test=registry.query("split=='test'")[['candidate_id','identity']].reset_index(drop=True)
    cat=pd.concat([train,test],ignore_index=True)
    mol=[Chem.MolFromSmiles(s) for s in cat.identity]
    fp=np.concatenate((np.array([morgan.GetFingerprintAsNumPy(m) for m in mol]),
                       np.array([fcfp.GetFingerprintAsNumPy(m) for m in mol]),
                       np.array([[f(m) for f in descriptors] for m in mol])),axis=1,dtype=np.float32)
    n=len(train)
    mu=float(train.point.mean())
    y=train.point.to_numpy()-mu
    def squared(raw,dataset):
        return 1.1*(raw-y), np.full_like(raw,1.1)
    def fitted(features):
        model=lgb.train({**params,'objective':squared},lgb.Dataset(features[:n],label=y),num_boost_round=300)
        return model.predict(features[n:],num_threads=2,raw_score=True)+mu
    scores=test.copy()
    scores['fp_point_l2']=fitted(fp)
    for name,encoder in encoders.items():
        lat,record=encoder.encode(cat.rename(columns={'identity':'smiles'}))
        assert record.candidate_id.equals(cat.candidate_id)
        scores[name+'_point_l2']=fitted(np.column_stack([fp,lat['mean']]))
    assert np.isfinite(scores.iloc[:,2:]).all().all()
    path=here/f'{assay}_locked_point_l2.csv'
    scores.to_csv(path,index=False)
    (here/f'{assay}_point_l2_manifest.json').write_text(json.dumps(
      {'train':n,'test':len(test),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()},indent=2))
    print(assay,len(test),flush=True)

