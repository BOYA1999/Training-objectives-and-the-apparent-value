from pathlib import Path
import sys
import torch
import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

here = Path(__file__).resolve().parent
base = here.parent
sys.path[:0] = [str(base/'cycle06'), str(base/'cycle07_model')]
import fit_continuous
from fit import FrozenMol2MolEncoder
from frozen_chemgpt import FrozenChemGPTEncoder
from prepare import standardize

def connectivity(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToInchiKey(rdMolStandardize.ChargeParent(m)).split('-')[0] if m else ''

source = base/'cycle06'/'cyp2c9'
registry = pd.read_csv(source/'structures.csv').query("split=='train'")
early = pd.read_csv(base/'cycle05_data'/'cyp-challenge-TRAIN_inhibition.csv')
exclude = set(early.SMILES.map(connectivity))
assert set(registry.identity.map(connectivity)).issubset(exclude)
raw = pd.read_csv(base/'cycle08_data'/'NCATS_CYP2C9_candidates.csv').sort_values('candidate_id')
raw['identity'] = raw.smiles.map(standardize, na_action='ignore')
raw['connectivity'] = raw.identity.map(connectivity, na_action='ignore').fillna('')
ok = raw.identity.notna() & raw.connectivity.ne('') & ~raw.connectivity.isin(exclude)
live = raw[ok].copy()
live['analysis_id'] = live.groupby('identity').candidate_id.transform('first')
test = live[['analysis_id','identity']].drop_duplicates('analysis_id').rename(columns={'analysis_id':'candidate_id'}).assign(split='test')
assert test.identity.is_unique and not set(registry.identity)&set(test.identity)
live[['candidate_id','analysis_id','identity']].to_csv(here/'ncats_candidate_identity.csv',index=False)
raw[~ok][['candidate_id','identity','connectivity']].to_csv(here/'ncats_excluded_candidate.csv',index=False)
print('raw SIDs', len(raw),'eligible SIDs', len(live),'eligible molecules',len(test),'Octant training', len(registry),flush=True)
fit_continuous.root=here
for name, enc in [('mol2mol',FrozenMol2MolEncoder()),('chemgpt',FrozenChemGPTEncoder())]:
    out=here/name
    out.mkdir(exist_ok=True)
    pd.concat([registry,test]).to_csv(out/'structures.csv',index=False)
    (out/'early_labels.csv').write_bytes((source/'early_labels.csv').read_bytes())
    fit_continuous.fit_project(name,enc)



