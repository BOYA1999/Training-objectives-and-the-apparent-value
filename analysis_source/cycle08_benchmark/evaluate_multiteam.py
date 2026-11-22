from pathlib import Path
from itertools import combinations
import hashlib
import functools
import json
import sys
import numpy as np
import pandas as pd

here = Path(__file__).resolve().parent
base = here.parent
sys.path[:0] = [str(base/'cycle06_novelty'), str(base/'cycle06')]
from official_custom_scoring_functions import rae_soft_threshold_absolute_error
from prepare import standardize

versions = pd.read_csv(here/'frozen_versions.csv')
for row in versions.itertuples():
    assert hashlib.sha256((here/row.file).read_bytes()).hexdigest()==row.sha256
    assert row.committed_at < '2026-11-04'
archives = {r.file[:-4]:pd.read_csv(here/r.file).rename(columns={'Molecule_Name':'candidate_id'})
            for r in versions.drop_duplicates('sha256').itertuples()}
other = base/'cycle08_independent'
extra = json.loads((other/'prediction_manifest.json').read_text())
owners = {name:'adlvdl' for name in archives}
seen = set(versions.sha256)
for row in extra:
    if not row.get('blob_hash_matches') or row.get('rows') != 750 or row['commit_date'] >= '2026-11-04':
        continue
    file = other/row['local_file']
    if file.suffix != '.csv' or row['sha256'] in seen:
        continue
    assert hashlib.sha256(file.read_bytes()).hexdigest() == row['sha256']
    seen.add(row['sha256'])
    archives[file.stem] = pd.read_csv(file).rename(columns={'Molecule_Name':'candidate_id'})
    owners[file.stem] = row['owner']
print(pd.Series(owners).value_counts().to_dict())
canonical = functools.lru_cache(maxsize=5000)(standardize)
projects = ('cyp1a2','cyp2c9','cyp2d6','cyp3a4')
rows, pairs = [], []
for project in projects:
    source = base/'cycle06'/project
    c = pd.read_csv(source/'continuous_locked_scores.csv')
    labels = pd.read_csv(source/'late_labels.csv')
    d = c.merge(labels, on='candidate_id', validate='one_to_one')
    d['tie'] = d.identity.map(lambda s:hashlib.sha256(s.encode()).hexdigest())
    stem = project.upper()+'_pIC50_direct_inhibition'
    score_cols = {'fp_point':d.cont_fp_point, 'fp_interval':d.cont_fp_interval,
                  'Mol2Mol_interval':d.cont_fused_interval,
                  'ChemGPT_interval':d[['candidate_id']].merge(
                      pd.read_csv(base/'cycle07'/'chemgpt'/project/'continuous_locked_scores.csv')[['candidate_id','cont_fused_interval']],
                      on='candidate_id',validate='one_to_one').cont_fused_interval}
    for name, archive in archives.items():
        assert set(archive.candidate_id)==set(d.candidate_id)
        a = d[['candidate_id','identity']].merge(archive,on='candidate_id',validate='one_to_one')
        assert a.identity.eq(a.SMILES.map(canonical)).all()
        score_cols[name]=a[stem]
    rank_df=pd.DataFrame(score_cols).assign(tie=d.tie)
    assert len(d)==750 and np.isfinite(rank_df.drop(columns='tie')).all().all()
    fit = d[['point','lower','upper']].notna().all(axis=1)
    assert (d.loc[fit,'lower']<=d.loc[fit,'upper']).all()
    metrics={}
    selection={}
    for name, score in score_cols.items():
        selection[name] = set(rank_df.sort_values([name,'tie'],ascending=[False,True]).head(150).index)
        top = d.loc[sorted(selection[name])]
        point = int(top.point.ge(6).sum())
        interval = int(top.hit.eq(1).sum())
        st = rae_soft_threshold_absolute_error(d.loc[fit,'point'],score[fit],d.loc[fit,'upper'],d.loc[fit,'lower'])
        metrics[name] = (st, point, interval)
        rows.append(dict(project=project,name=name,owner=owners.get(name,'local'),st_rae=st,spearman=d.point.corr(score,method='spearman'),
                         point_support=point,interval_support=interval,unknown=int(top.hit.isna().sum()),
                         n_scores=len(score),n_assay_ranges=int(fit.sum())))
    for a,b in combinations(score_cols,2):
        swapped_a=d.loc[sorted(selection[a]-selection[b])]
        swapped_b=d.loc[sorted(selection[b]-selection[a])]
        delta=metrics[a][2]-metrics[b][2]
        pairs.append(dict(project=project,a=a,b=b,st_rae_a_minus_b=metrics[a][0]-metrics[b][0],
                          point_diff=metrics[a][1]-metrics[b][1],interval_diff=delta,
                          low=delta-int(swapped_b.hit.isna().sum()),high=delta+int(swapped_a.hit.isna().sum()),
                          swap_n=len(swapped_a), owner_a=owners.get(a,"local"), owner_b=owners.get(b,"local"), both_archived=a in archives and b in archives))
out = pd.DataFrame(rows)
ps = pd.DataFrame(pairs)
out.to_csv(here/'multiteam_scores.csv',index=False)
ps.to_csv(here/'multiteam_pairs.csv',index=False)

ps=ps[ps.both_archived & ps.st_rae_a_minus_b.abs().gt(1e-12)].copy()
ps['cross_team']=ps.owner_a.ne(ps.owner_b)
ps['reverse']=ps.st_rae_a_minus_b*ps.interval_diff>0
ps['robust']=(ps.st_rae_a_minus_b.gt(0)&ps.low.gt(0))|(ps.st_rae_a_minus_b.lt(0)&ps.high.lt(0))
summary=ps.groupby(['project','cross_team'])[['reverse','robust']].agg(['sum','size'])
summary.to_csv(here/'multiteam_summary.csv')
print(summary.to_string())
for (owner,project),g in out[out.owner.ne('local')].groupby(['owner','project']):
    low=g.sort_values('st_rae').iloc[0]
    high=g.sort_values(['interval_support','st_rae'],ascending=[False,True]).iloc[0]
    print(owner,project,len(g),low['name'],round(low.st_rae,3),low.interval_support,high['name'],round(high.st_rae,3),high.interval_support)
