from pathlib import Path
import numpy as np
import pandas as pd

p = Path(__file__).parent
d = pd.read_csv(p / 'NCATS_CYP2C9_replicates.csv', low_memory=False)
cols = [c for c in d if c.startswith('Activity at')]
doses = np.array([float(c.split(' ')[2]) for c in cols])
d['max_tested_uM'] = np.nanmax(np.where(d[cols].notna(), doses, np.nan), axis=1)
good = d.Fit_CurveClass.isin([-1.1, -1.2, -2.1, -2.2]) & d.pIC50.notna() & d.Efficacy.gt(65)
inactive = d.Fit_CurveClass.eq(4) & d.max_tested_uM.ge(1)
d['curve_label_pIC50_6'] = np.where(good, np.where(d.pIC50.ge(6), 1, 0), np.where(inactive, 0, np.nan))
d['curve_reason'] = np.select(
    [good & d.pIC50.ge(6), good & d.pIC50.lt(6), inactive],
    ['good_inhibition_pIC50_ge6_efficacy_gt65', 'good_inhibition_pIC50_lt6_efficacy_gt65',
     'source_inactive_tested_at_or_above_1uM'], default='unknown_curve_or_efficacy')
rows = []
for parent, g in d[d.parent_smiles.notna()].groupby('parent_smiles', sort=True):
    values = g.curve_label_pIC50_6.dropna()
    conflict = values.nunique() > 1
    unknown = len(values) != len(g)
    rows.append(dict(
        candidate_id=min(g.candidate_id), smiles=parent, n_SIDs=g.candidate_id.nunique(),
        n_curves=len(g), source_SIDs=';'.join(sorted(g.candidate_id.unique())),
        source_CIDs=';'.join(sorted(set(g.PUBCHEM_CID.dropna().astype(int).astype(str)))),
        source_outcomes=';'.join(sorted(g.PUBCHEM_ACTIVITY_OUTCOME.unique())),
        label_pIC50_6=values.iloc[0] if len(values) and not conflict and not unknown else np.nan,
        repeat_conflict=conflict, has_unknown_curve=unknown,
        n_positive_curves=int(values.eq(1).sum()), n_negative_curves=int(values.eq(0).sum()),
        n_unknown_curves=int(g.curve_label_pIC50_6.isna().sum()),
        min_accepted_pIC50=g.loc[good.loc[g.index], 'pIC50'].min(),
        max_accepted_pIC50=g.loc[good.loc[g.index], 'pIC50'].max()))
c = pd.DataFrame(rows)
m = pd.read_csv(p / 'NCATS_Octant_CYP2C9_structure_mapping.csv')
train = set(m.loc[m.octant_split.eq('train'), 'parent_smiles'])
c['overlap_octant_train_parent'] = c.smiles.isin(train)
c['external_evaluation_eligible'] = ~c.overlap_octant_train_parent
d.to_csv(p / 'NCATS_CYP2C9_fixed_curve_labels.csv', index=False)
c.to_csv(p / 'NCATS_CYP2C9_fixed_parent_labels.csv', index=False)
c[['candidate_id', 'smiles', 'external_evaluation_eligible', 'overlap_octant_train_parent']].to_csv(
    p / 'NCATS_CYP2C9_fixed_candidates.csv', index=False)
e = c[c.external_evaluation_eligible]
assert (len(c), len(e), e.label_pIC50_6.eq(1).sum(), e.label_pIC50_6.eq(0).sum(),
        e.label_pIC50_6.isna().sum()) == (5067, 5012, 199, 3816, 997)
assert not c.candidate_id.duplicated().any()
print('Verified:', len(e), 'external parents;', e.label_pIC50_6.value_counts(dropna=False).to_dict())
