from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import sys

import numpy as np
import pandas as pd
from rdkit.Chem.Scaffolds import MurckoScaffold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SLOTS = 150
OBJECTIVES = ['point_l1', 'point_l2', 'interval']
MODELS = ['FP', 'Mol2Mol', 'ChemGPT']
CORE_SEED = 20261120


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_original():
    folder = ROOT/'cycle06/cyp2c9'
    aim = ROOT/'cycle07/chemgpt/cyp2c9'
    combined = []
    for model, path, manifest in [
        ('Mol2Mol', folder/'continuous_locked_scores.csv', folder/'continuous_score_manifest.json'),
        ('ChemGPT', aim/'continuous_locked_scores.csv', aim/'continuous_score_manifest.json'),
    ]:
        info = json.loads(manifest.read_text())
        assert sha(path) == info['score_hash']
        table = pd.read_csv(path).set_index('candidate_id')
        fields = {'cont_fused_point': f'{model}_point_l1', 'cont_fused_interval': f'{model}_interval'}
        if model == 'Mol2Mol':
            fields.update({'identity':'identity', 'cont_fp_point':'FP_point_l1', 'cont_fp_interval':'FP_interval'})
        combined.append(table[list(fields)].rename(columns=fields))
    path = ROOT/'cycle09/cyp2c9_locked_point_l2.csv'
    assert sha(path) == json.loads((ROOT/'cycle09/cyp2c9_point_l2_manifest.json').read_text())['sha256']
    l2 = pd.read_csv(path).set_index('candidate_id')
    assert l2['identity'].reindex(combined[0].index).equals(combined[0]['identity'])
    combined.append(l2.drop(columns='identity').rename(columns={'fp_point_l2':'FP_point_l2'}))
    scores = pd.concat(combined,axis=1)
    assert scores.index.is_unique and len(scores) == 750 and scores.notna().all().all()
    return scores


def read_locked(path, lock):
    info = json.loads(lock.read_text())
    assert sha(path) == info['sha256']
    return pd.read_csv(path).set_index('candidate_id')


def assays(scores):
    rawpath = ROOT/'cycle05_data/cyp-challenge-TEST-UNBLINDED_inhibition.csv'
    raw = pd.read_csv(rawpath).set_index('Molecule_Name')
    assert set(raw.index) == set(scores.index) and raw.index.is_unique
    raw = raw.loc[scores.index]
    stem = 'CYP2C9_pIC50_direct_inhibition'
    point, lo, hi = (raw[k] for k in (stem, stem+'_conf_low', stem+'_conf_high'))
    valid = lo.le(point) & point.le(hi)
    labels = pd.Series(np.select([valid & lo.ge(6), valid & hi.lt(6)],[1.0,0.0],default=np.nan),index=raw.index)
    previous = pd.read_csv(ROOT/'cycle06/cyp2c9/late_labels.csv').set_index('candidate_id').reindex(raw.index).hit
    np.testing.assert_equal(labels.to_numpy(),previous.to_numpy())
    assert (labels.eq(1).sum(),labels.eq(0).sum(),labels.isna().sum()) == (96,612,42)
    return labels, rawpath


def evaluate(scores, target, setting, seed):
    target = target.reindex(scores.index)
    hit, unknown = target.eq(1), target.isna()
    tie = scores.identity.map(lambda key:hashlib.sha256(key.encode()).hexdigest())
    selections = {}
    counts = []
    for train in OBJECTIVES:
        for model in MODELS:
            col = f'{model}_{train}'
            ranked = scores.assign(tie=tie).sort_values([col,'tie'],ascending=[False,True])[col]
            choose = pd.Series(0,index=scores.index,dtype=np.int8)
            choose.loc[ranked.index[:SLOTS]] = 1
            selections[col] = choose
            counts.append(dict(setting=setting,seed=seed,objective=train,representation=model,
                hits=int(choose[hit].sum()),unresolved=int(choose[unknown].sum()),
                boundary_tie=bool(ranked.iloc[SLOTS-1] == ranked.iloc[SLOTS])))
    result, weights = [], {}
    for model in MODELS[1:]:
        for train in OBJECTIVES:
            left = f'{model}_{train}'
            for comparator in OBJECTIVES:
                right = f'FP_{comparator}'
                w = selections[left]-selections[right]
                name = f'{left}_vs_{right}'
                weights[name] = w
                net = int(w[hit].sum())
                result.append(dict(setting=setting,seed=seed,model=model,model_objective=train,
                    fp_objective=comparator,same_objective=train==comparator,
                    model_hits=int(selections[left][hit].sum()),fp_hits=int(selections[right][hit].sum()),
                    net=net,lower=net+int(w[unknown].clip(upper=0).sum()),
                    upper=net+int(w[unknown].clip(lower=0).sum()),
                    fused_only_unknown=int((w[unknown]>0).sum()),fp_only_unknown=int((w[unknown]<0).sum()),
                    contrast=name))
    return pd.DataFrame(counts),pd.DataFrame(result),selections,weights


def bootstrap(settings, labels, previous):
    hit, unknown = labels.eq(1), labels.isna()
    scaffold = previous.identity.map(lambda key: MurckoScaffold.MurckoScaffoldSmiles(smiles=key) or key)
    rows = []
    for name, (counts, pairs, selections, weights) in settings.items():
        delta_a = selections['FP_point_l2']-selections['FP_point_l1']
        delta_b = selections['FP_interval']-selections['FP_point_l2']
        objective_weights = {'FP_point_l2_vs_FP_point_l1':delta_a,'FP_interval_vs_FP_point_l2':delta_b,
                            'difference_of_FP_gains':delta_a-delta_b}
        contrast_weights = {**objective_weights, **weights}
        contribution = pd.DataFrame(contrast_weights).mul(hit,axis=0).groupby(scaffold).sum()
        size = len(contribution)
        assert size == 546
        frequency = np.random.default_rng(CORE_SEED).multinomial(size,np.full(size,1/size),size=15000)
        delta = frequency.astype(float) @ contribution.to_numpy(dtype=float)
        ci = np.quantile(100*delta/SLOTS,[.025,.975],axis=0)
        for i,(contrast,w) in enumerate(contrast_weights.items()):
            net = int(w[hit].sum())
            rows.append(dict(setting=name,contrast=contrast,hit_difference=net,
                unresolved_lower=net+int(w[unknown].clip(upper=0).sum()),
                unresolved_upper=net+int(w[unknown].clip(lower=0).sum()),
                boot_low_per100=ci[0,i],boot_high_per100=ci[1,i],boot_repetitions=15000,
                boot_seed=CORE_SEED,scaffold_groups=size,boot_denominator=SLOTS))
    return pd.DataFrame(rows)


def verify_new_baseline(original, run):
    assert run.identity.reindex(original.index).equals(original.identity)
    cols = original.columns.difference(['identity'])
    diff = (original[cols]-run.loc[original.index,cols]).abs().max()
    assert diff.max() < 1e-12, diff
    return diff.to_dict()


def main(seeds_only):
    original = load_original()
    if seeds_only:
        all_scores = read_locked(HERE/'locked_predictions.csv',HERE/'prediction_lock.json')
        maxdiff = verify_new_baseline(original,all_scores.query('seed == @CORE_SEED').drop(columns='seed'))
        labels,raw = assays(original)
        data = [evaluate(frame.drop(columns='seed'),labels,'reseed_0.7',int(s))
                for s,frame in all_scores.groupby('seed')]
        pd.concat([elem[0] for elem in data]).to_csv(HERE/'seed_cells.csv',index=False)
        pairs = pd.concat([elem[1] for elem in data])
        pairs.to_csv(HERE/'seed_pairs.csv',index=False)
        assert len(pairs)==90 and not pd.concat([elem[0] for elem in data]).boundary_tie.any()
        summary = []
        for name,g in pairs.groupby('contrast',sort=False):
            summary.append(dict(contrast=name,model=g.model.iloc[0],model_objective=g.model_objective.iloc[0],
                fp_objective=g.fp_objective.iloc[0],same_objective=g.same_objective.iloc[0],
                n_seeds=len(g),net_min=g.net.min(),net_max=g.net.max(),lower_min=g.lower.min(),
                upper_max=g.upper.max(),negative=int(g.net.lt(0).sum()),zero=int(g.net.eq(0).sum()),
                positive=int(g.net.gt(0).sum()),entirely_negative=int(g.upper.lt(0).sum()),
                entirely_positive=int(g.lower.gt(0).sum()),overlap_zero=int((g.lower.le(0)&g.upper.ge(0)).sum())))
        summary = pd.DataFrame(summary)
        summary.to_csv(HERE/'seed_range.csv',index=False)
        print(summary.to_string(index=False))
        check = dict(evaluated_utc=datetime.now(timezone.utc).isoformat(),per_column_original_seed_max_abs_score_difference=maxdiff,
                     original_seed_matches=True,raw_labels_sha256=sha(raw),
                     seed_predictions_sha256=sha(HERE/'locked_predictions.csv'),no_boundary_ties=True)
        (HERE/'seed_verification.json').write_text(json.dumps(check,indent=2),encoding='utf-8')
        return
    prev = ROOT/'paper_revision_20261120/feature_sampling'
    full = read_locked(prev/'locked_predictions.csv',prev/'prediction_lock.json').loc[original.index]
    assert full.identity.equals(original.identity)
    labels,raw = assays(original)
    settings = {key:evaluate(frame,labels,key,CORE_SEED) for key,frame in [('original_0.7',original),('full_1.0',full)]}
    cells = pd.concat([elem[0] for elem in settings.values()])
    pairs = pd.concat([elem[1] for elem in settings.values()])
    assert len(cells)==18 and len(pairs)==36 and not cells.boundary_tie.any()
    assert cells.hits.to_list() == [50,40,45,65,59,58,68,57,60,50,40,47,64,61,59,67,57,62]
    cells.to_csv(HERE/'objective_cells.csv',index=False)
    pairs.to_csv(HERE/'objective_pairs.csv',index=False)
    intervals = bootstrap(settings,labels,original)
    previous = pd.read_csv(ROOT/'cycle09/scaffold_bootstrap.csv').query('assay == "cyp2c9"')
    names = ['FP_point_l2_vs_FP_point_l1','FP_interval_vs_FP_point_l2',
             'Mol2Mol_interval_vs_FP_point_l2','Mol2Mol_interval_vs_FP_interval',
             'ChemGPT_interval_vs_FP_point_l2','ChemGPT_interval_vs_FP_interval']
    current = intervals.query('setting == "original_0.7"').set_index('contrast').loc[names]
    np.testing.assert_allclose(current[['boot_low_per100','boot_high_per100']],previous[['low_pct','high_pct']],atol=1e-12,rtol=0)
    np.testing.assert_array_equal(current.hit_difference,previous.net)
    intervals.to_csv(HERE/'objective_intervals.csv',index=False)
    aligned = settings['original_0.7'][2]
    tie = original.identity.map(lambda key:hashlib.sha256(key.encode()).hexdigest())
    permutation = []
    for model, folder in [('Mol2Mol','cycle06'),('ChemGPT','cycle07/chemgpt')]:
        pp = pd.read_csv(ROOT/f'{folder}/cyp2c9/continuous_locked_scores.csv').set_index('candidate_id')
        indexes = pp.assign(tie=tie).sort_values(['cont_shuffled_interval','tie'],ascending=[False,True]).head(SLOTS).index
        member = original.index.isin(indexes).astype(int)
        a = aligned[f'{model}_interval']
        w = a-member
        unknown, hit = labels.isna(), labels.eq(1)
        net = int(w[hit].sum())
        permutation.append(dict(model=model,slots=SLOTS,aligned_hits=int(a[hit].sum()),
            permuted_hits=int(member[hit].sum()),net=net,lower=net+int(w[unknown].clip(upper=0).sum()),
            upper=net+int(w[unknown].clip(lower=0).sum())))
    pd.DataFrame(permutation).to_csv(HERE/'aligned_vs_permuted.csv',index=False)
    print(pairs[['setting','model','model_objective','fp_objective','net','lower','upper']].to_string(index=False))
    print(intervals[intervals.contrast.isin(names[:2]+['difference_of_FP_gains'])].to_string(index=False))
    check = dict(evaluated_utc=datetime.now(timezone.utc).isoformat(),archived_bootstrap_agreement_rows=len(previous),
                 original_original_sha256=sha(ROOT/'cycle06/cyp2c9/continuous_locked_scores.csv'),
                 original_chemgpt_sha256=sha(ROOT/'cycle07/chemgpt/cyp2c9/continuous_locked_scores.csv'),
                 original_l2_sha256=sha(ROOT/'cycle09/cyp2c9_locked_point_l2.csv'),
                 full_feature_sha256=sha(prev/'locked_predictions.csv'),raw_labels_sha256=sha(raw),
                 raw_labels_count=[96,612,42],no_boundary_ties=True)
    (HERE/'objective_verification.json').write_text(json.dumps(check,indent=2),encoding='utf-8')


if __name__ == '__main__':
    assert len(sys.argv) == 1 or sys.argv[1:] == ['seeds']
    main(sys.argv[1:] == ['seeds'])

