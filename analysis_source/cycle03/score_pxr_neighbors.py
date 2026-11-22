from pathlib import Path
from neighbor_methods import pd, standardize, score_neighbors

root=Path(__file__).resolve().parent
data = root.parent/'cycle02_data_b'
train = pd.read_csv(data/'pxr_prepared_train.csv')
train_lab = pd.read_csv(data/'labels'/'pxr_train_labels.csv').dropna(subset=['hit'])
train = train.merge(train_lab[['candidate_id','hit']],left_on='compound_id',right_on='candidate_id').sort_values('candidate_id')
train['identity'] = train.canonical_smiles.map(standardize)
group = train.groupby('identity').hit.nunique()
train = train[train.identity.isin(group[group.eq(1)].index)].drop_duplicates('identity').reset_index(drop=True)
test = pd.read_csv(data/'pxr-challenge_TEST_BLINDED.csv')
assert len(test)==513
test = test.rename(columns={test.columns[0]:'candidate_id',test.columns[1]:'smiles'})
test = test[['candidate_id','smiles']].sort_values('candidate_id').reset_index(drop=True)
assert test.candidate_id.is_unique
test['identity'] = test.smiles.map(standardize)
test = test[test.identity.notna() & ~test.identity.isin(set(train.identity))].copy()
assert test.identity.is_unique
score_neighbors(train,test,root)
