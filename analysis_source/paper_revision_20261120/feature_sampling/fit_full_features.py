import sys
sys.dont_write_bytecode = True
from pathlib import Path
import hashlib
import json
import time
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
reads = set()
def audit_read(event, args):
    if event == 'open' and isinstance(args[0], str):
        path = args[0].replace('\\', '/').lower()
        assert not any(token in path for token in ['late_labels.csv','test-unblinded','pxr_test_labels.csv'])
        opened = Path(args[0]).resolve()
        if opened.is_relative_to(ROOT) and isinstance(args[1], str) and 'r' in args[1]:
            reads.add(opened.relative_to(ROOT).as_posix())
sys.addaudithook(audit_read)

import torch
sys.path[:0] = [str(ROOT/'cycle06'), str(ROOT/'cycle07_model')]
from fit import FrozenMol2MolEncoder, Chem, descriptors, fcfp, lgb, morgan, np, params, pd
from frozen_chemgpt import FrozenChemGPTEncoder

started = time.perf_counter()
source = ROOT/'cycle06/cyp2c9'
registry = pd.read_csv(source/'structures.csv')
early = pd.read_csv(source/'early_labels.csv')
train = registry.query("split == 'train'").merge(early, on='candidate_id', validate='one_to_one')
test = registry.query("split == 'test'")[['candidate_id','identity']].reset_index(drop=True)
all_rows = pd.concat([train,test],ignore_index=True)
assert len(train)==1285 and len(test)==750 and all_rows.candidate_id.is_unique
assert train[['point','lower','upper']].notna().all().all()
assert (train.lower.le(train.point)&train.point.le(train.upper)).all()
settings = {**params, 'feature_fraction':1.0}
assert {k:(params[k],settings[k]) for k in params if params[k]!=settings[k]} == {'feature_fraction':(0.7,1.0)}
inputs = [source/'structures.csv',source/'early_labels.csv',ROOT/'cycle06/fit.py',ROOT/'cycle06/fit_continuous.py',ROOT/'cycle09/fit_point_control.py',ROOT/'cycle06_model/frozen_encoder.py',ROOT/'cycle07_model/frozen_chemgpt.py']
weights = [ROOT/'models_outputs/mol2mol_similarity.prior',ROOT/'cycle07_model/checkpoint/pytorch_model.bin']
digests = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs+weights}
plan = dict(started_utc=datetime.now(timezone.utc).isoformat(),train_rows=len(train),test_rows=len(test),
    input_sha256=digests,base_parameters=params,changed_parameters={'feature_fraction':{'old':0.7,'new':1.0}},
    rounds=300,objectives=['point_l1','point_l2','interval'],representations=['FP','Mol2Mol','ChemGPT'],
    slots=150,test_labels_forbidden_during_fit=True,retrospective_sensitivity=True)
(HERE/'fit_plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
molecules = [Chem.MolFromSmiles(s) for s in all_rows.identity]
fp = np.concatenate((np.array([morgan.GetFingerprintAsNumPy(m) for m in molecules]),
    np.array([fcfp.GetFingerprintAsNumPy(m) for m in molecules]),
    np.array([[f(m) for f in descriptors] for m in molecules])),axis=1,dtype=np.float32)
features = {'FP':fp}
coverage = {}
for name, cls in [('Mol2Mol',FrozenMol2MolEncoder),('ChemGPT',FrozenChemGPTEncoder)]:
    stamp = time.perf_counter()
    encoder = cls()
    network = encoder.model.network if name=='Mol2Mol' else encoder.model
    assert not network.training and not any(p.requires_grad for p in network.parameters())
    latent, rows = encoder.encode(all_rows.rename(columns={'identity':'smiles'}))
    assert rows.candidate_id.equals(all_rows.candidate_id)
    rows['split'] = ['train']*len(train)+['test']*len(test)
    rows.to_csv(HERE/f'{name}_coverage.csv',index=False)
    features[name] = np.hstack([fp,latent['mean']])
    coverage[name] = {'seconds':time.perf_counter()-stamp,'counts':rows.groupby(['split','status']).size().to_dict()}
    coverage[name]['counts'] = {'/'.join(k):int(v) for k,v in coverage[name]['counts'].items()}
    print('Encoded',name,coverage[name],flush=True)
    del encoder

n = len(train)
mu = float(train.point.mean())
y,lo,hi = [train[k].to_numpy()-mu for k in ['point','lower','upper']]
def squared(raw, dataset):
    return 1.1*(raw-y), np.full_like(raw,1.1)
def interval(raw, dataset):
    err = np.where(raw<lo,raw-lo,np.maximum(raw-hi,0))
    return err+0.1*(raw-y),0.1+((raw<lo)|(raw>hi)).astype(float)
output = test.copy()
timings = []
for objective, loss in [('point_l1','regression_l1'),('point_l2',squared),('interval',interval)]:
    for representation, x in features.items():
        stamp = time.perf_counter()
        fitted = lgb.train({**settings,'objective':loss},lgb.Dataset(x[:n],label=y),num_boost_round=300)
        name = f'{representation}_{objective}'
        output[name] = fitted.predict(x[n:],num_threads=2,raw_score=True)+mu
        fitted.save_model(str(HERE/f'{name}.txt'))
        timings.append(dict(model=name,seconds=time.perf_counter()-stamp,features=x.shape[1]))
        print('Fitted',timings[-1],flush=True)
assert np.isfinite(output.iloc[:,2:].to_numpy()).all()
assert all(hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest for path,digest in digests.items())
dest = HERE/'locked_predictions.csv'
output.to_csv(dest,index=False)
lock = dict(locked_utc=datetime.now(timezone.utc).isoformat(),sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
    rows=len(output),score_columns=output.columns[2:].tolist(),elapsed_seconds=time.perf_counter()-started,
    coverage=coverage,timings=timings,project_read_paths=sorted(reads),test_labels_read=False,
    parameters={**settings,'objective':'specified independently for each objective'},lightgbm_version=lgb.__version__)
(HERE/'prediction_lock.json').write_text(json.dumps(lock,indent=2),encoding='utf-8')
print('LOCKED',lock['sha256'],'elapsed',lock['elapsed_seconds'],flush=True)
