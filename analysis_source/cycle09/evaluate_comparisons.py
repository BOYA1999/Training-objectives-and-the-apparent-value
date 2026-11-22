from pathlib import Path
import hashlib
import json
import pandas as pd

here=Path(__file__).resolve().parent
base=here.parent
sources={'Mol2Mol':base/'cycle06','ChemGPT':base/'cycle07'/'chemgpt'}
rows=[]
for assay in ('pxr','cyp1a2','cyp2c9','cyp2d6','cyp3a4'):
    labels=pd.read_csv(base/'cycle06'/assay/'late_labels.csv')
    l2=here/f'{assay}_locked_point_l2.csv'
    if l2.exists():
        sha=json.loads((here/f'{assay}_point_l2_manifest.json').read_text())['sha256']
        assert hashlib.sha256(l2.read_bytes()).hexdigest()==sha
        point_l2=pd.read_csv(l2)
    for model,root in sources.items():
        p=root/assay/'continuous_locked_scores.csv'
        sha=json.loads((p.parent/'continuous_score_manifest.json').read_text())['score_hash']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==sha
        x=pd.read_csv(p).merge(labels,validate='one_to_one')
        assert len(x)==len(labels) and x.candidate_id.is_unique
        x['tie']=x.identity.map(lambda s:hashlib.sha256(s.encode()).hexdigest())
        pairs={
            'unmatched_L1_point':('cont_fused_interval','cont_fp_point'),
            'matched_L1_point':('cont_fused_point','cont_fp_point'),
            'matched_interval':('cont_fused_interval','cont_fp_interval'),
            'shuffled_interval':('cont_shuffled_interval','cont_fp_interval'),
        }
        if l2.exists():
            x=x.merge(point_l2,validate='one_to_one')
            assert len(x)==len(labels)
            pairs['matched_L2_point']=(model+'_point_l2','fp_point_l2')
            pairs['unmatched_L2_point']=('cont_fused_interval','fp_point_l2')
            pairs['same_FP_interval_vs_L2']=('cont_fp_interval','fp_point_l2')
            pairs['same_model_interval_vs_L2']=('cont_fused_interval',model+'_point_l2')
        x=x.set_index('candidate_id')
        point=x.point.ge(6).where(x.point.notna())
        for percent in (10,20,30):
            shortlist={}
            for col in set(sum((list(z) for z in pairs.values()),[])):
                ordered=x.sort_values([col,'tie'],ascending=[False,True])
                group=ordered.groupby('phase').phase
                shortlist[col]=ordered[group.cumcount()<group.transform('size')*percent//100].index
            for comparison,(test,control) in pairs.items():
                a,b=shortlist[test],shortlist[control]
                aonly,b_only=a.difference(b),b.difference(a)
                assert len(a)==len(b)
                for endpoint,truth in [('point',point),('assay_supported',x.hit)]:
                    ha=int(truth.loc[a].eq(1).sum())
                    hb=int(truth.loc[b].eq(1).sum())
                    net=ha-hb
                    ua=int(truth.loc[aonly].isna().sum())
                    ub=int(truth.loc[b_only].isna().sum())
                    rows.append(dict(assay=assay,model=model,budget_percent=percent,slots=len(a),
                                     comparison=comparison,endpoint=endpoint,a_hits=ha,b_hits=hb,delta=net,
                                     lower=net-ub,upper=net+ua,exclusive=len(aonly)))
d=pd.DataFrame(rows)
d.to_csv(here/'comparison_matrix.csv',index=False)
print(d.query("budget_percent==20 & endpoint=='assay_supported' & assay in ['cyp2c9','pxr']")
      [['assay','model','comparison','a_hits','b_hits','delta','lower','upper']].to_string(index=False))

