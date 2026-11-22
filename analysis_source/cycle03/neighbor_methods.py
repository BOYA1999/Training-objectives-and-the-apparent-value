import hashlib
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root.parent/'cycle02'))
from score_catalogue import Chem, DataStructs, conditional_scores, np, pd, rdFingerprintGenerator, standardize
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def score_neighbors(train, test, root):
    morgan = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
    feature = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True,
                  atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen())

    def fps(series, generator):
        return [generator.GetFingerprint(Chem.MolFromSmiles(s)) for s in series]

    def sims(a,b):
        return np.asarray([DataStructs.BulkTanimotoSimilarity(x,b) for x in a])

    tx, vx = fps(train.identity,morgan),fps(test.identity,morgan)
    m = sims(vx,tx)
    f = sims(fps(test.identity,feature),fps(train.identity,feature))
    rows = []
    for i,candidate in enumerate(test.itertuples(index=False)):
        neighbors = np.union1d(np.argsort(-m[i],kind='stable')[:4],np.argsort(-f[i],kind='stable')[:4])
        for j in neighbors:
            rows.append(dict(candidate_id=candidate.candidate_id, identity=candidate.identity,
                             ref_id=train.loc[j,'candidate_id'],ref=train.loc[j,'identity'],
                             early_hit=float(train.loc[j,'hit']),ECFP4=m[i,j],FeatureMorgan=f[i,j]))
    pairs = pd.DataFrame(rows)
    pairs['conditional'] = conditional_scores(pairs.ref.tolist(),pairs.identity.tolist())
    print('pairs scored',pairs.conditional.notna().sum(),len(pairs),flush=True)
    rng = np.random.default_rng(20261120)
    pred = []
    for cid,g in pairs.groupby('candidate_id',sort=False):
        morgan_rank = rankdata(g.ECFP4)/len(g)
        feature_rank = rankdata(g.FeatureMorgan)/len(g)
        distance_rank = rankdata((g.ECFP4+g.FeatureMorgan)/2)/len(g)
        model = np.full(len(g),0.5)
        valid = np.isfinite(g.conditional.to_numpy())
        if valid.any():
            model[valid] = rankdata(g.conditional.to_numpy()[valid])/valid.sum()
        hybrid = (distance_rank+model)/2
        permuted = (distance_rank+rng.permutation(model))/2
        ranks = [morgan_rank,feature_rank,distance_rank,hybrid,permuted,model]
        names = ['ECFP_NN','Feature_NN','fingerprint_NN','AI_NN','permuted_AI_NN','likelihood_only_NN']
        r = {k:float(np.average(g.early_hit,weights=np.exp(3*v))) for k,v in zip(names,ranks)}
        pred.append(dict(candidate_id=cid,**r))
    test = test.merge(pd.DataFrame(pred),on='candidate_id',validate='one_to_one',sort=False)

    svm = SVC(C=10,kernel='precomputed',class_weight='balanced')
    svm.fit(sims(tx,tx),train.hit)
    test['Tanimoto_SVM'] = svm.decision_function(m)
    rf = RandomForestClassifier(n_estimators=300,min_samples_leaf=2,class_weight='balanced',random_state=20261120,n_jobs=2)
    rf.fit(np.asarray(tx,dtype=np.uint8),train.hit)
    test['ECFP_RF'] = rf.predict_proba(np.asarray(vx,dtype=np.uint8))[:,1]

    test.to_csv(root/'locked_scores.csv',index=False)
    pairs.to_csv(root/'neighbor_records.csv',index=False)
    lock=dict(n_train=len(train),positive_train=int(train.hit.sum()),n_candidates=len(test),
              n_scored_pairs=int(pairs.conditional.notna().sum()),n_pairs=len(pairs),
              score_sha256=hashlib.sha256((root/'locked_scores.csv').read_bytes()).hexdigest())
    (root/'score_lock.json').write_text(json.dumps(lock,indent=2),encoding='utf-8')
    print(lock,flush=True)
