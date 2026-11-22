from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

root=Path(__file__).resolve().parent
RDLogger.DisableLog('rdApp.*')

def standardize(smiles):
    m=Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(rdMolStandardize.FragmentParent(m),isomericSmiles=True) if m else None

def setup(project, early, late, candidates):
    base=root/project
    base.mkdir(exist_ok=True)
    early=early.dropna(subset=['point','lower','upper']).copy()
    early=early[early.lower.le(early.point)&early.upper.ge(early.point)]
    early['identity']=early.smiles.map(standardize)
    early=early.dropna(subset=['identity']).sort_values('candidate_id')
    width=(early.upper-early.lower).to_numpy()
    early['hard']=early.point.ge(6).astype(float)
    early['soft']=np.clip(np.divide(early.upper-6,width,out=early.hard.to_numpy().copy(),where=width>0),0,1)
    counts=early.groupby('identity').size()
    early=early[early.identity.isin(counts[counts.eq(1)].index)].copy()
    candidates=candidates.copy()
    assert candidates.candidate_id.is_unique and late.candidate_id.is_unique
    candidates['identity']=candidates.smiles.map(standardize)
    candidates=candidates[candidates.identity.notna()&~candidates.identity.isin(set(early.identity))].drop_duplicates('identity')
    ids=early[['candidate_id','identity']].assign(split='train')
    ids=pd.concat([ids,candidates[['candidate_id','identity']].assign(split='test')],ignore_index=True)
    assert ids.candidate_id.is_unique
    ids.to_csv(base/'structures.csv',index=False)
    early[['candidate_id','point','lower','upper','hard','soft']].to_csv(base/'early_labels.csv',index=False)
    late=late[late.candidate_id.isin(set(candidates.candidate_id))].copy()
    assert len(late)==len(candidates)
    late.to_csv(base/'late_labels.csv',index=False)
    print(project,'early',len(early),'ambiguous kept',int(early.soft.between(0,1,inclusive='neither').sum()),
          'later',len(candidates),flush=True)

def pxr():
    folder=root.parent/'cycle02_data_b'
    raw=pd.read_csv(folder/'pxr_prepared_train.csv')
    names={'compound_id':'candidate_id','pec50':'point','pec50_ci_lower':'lower','pec50_ci_upper':'upper'}
    raw=raw.rename(columns=names)
    early=raw[raw.lower.ge(raw.drc_min_observed_pconc)&raw.upper.le(raw.drc_max_observed_pconc)]
    late=pd.read_csv(folder/'labels'/'pxr_test_labels.csv').rename(columns=names)
    b=pd.read_csv(folder/'pxr-challenge_TEST_BLINDED.csv').rename(columns={'Molecule Name':'candidate_id','SMILES':'smiles'})
    setup('pxr',early,late[['candidate_id','phase','hit','point','lower','upper']],b)

def cyp():
    folder=root.parent/'cycle05_data'
    names={'Molecule_Name':'candidate_id','SMILES':'smiles'}
    original=pd.read_csv(folder/'cyp-challenge-TRAIN_inhibition.csv').rename(columns=names)
    b=pd.read_csv(folder/'cyp-challenge-TEST-BLINDED.csv').rename(columns=names)
    raw_test=pd.read_csv(folder/'cyp-challenge-TEST-UNBLINDED_inhibition.csv').rename(columns=names)
    assert set(raw_test.candidate_id)==set(b.candidate_id)
    for protein in ['CYP1A2','CYP2C9','CYP2D6','CYP3A4']:
        stem=protein+'_pIC50_direct_inhibition'
        cols={stem:'point',stem+'_conf_low':'lower',stem+'_conf_high':'upper'}
        early=original.rename(columns=cols)
        late=raw_test.rename(columns=cols).copy()
        late['phase']=1
        valid=late.lower.le(late.point)&late.upper.ge(late.point)
        late['hit']=np.where(valid&late.lower.ge(6),1,np.where(valid&late.upper.lt(6),0,np.nan))
        setup(protein.lower(),early,late[['candidate_id','phase','hit','point','lower','upper']],b)

if __name__=='__main__':
    pxr()
    cyp()
