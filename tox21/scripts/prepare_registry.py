import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog('rdApp.*')
home = Path(__file__).resolve().parents[1]
root = home.parent / 'analysis_source'
assay = home / 'source/aid1671198.csv'


def table(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        yield from csv.DictReader(handle)


def identify(s):
    mol = Chem.MolFromSmiles(s) if s else None
    if mol is None:
        return None
    fragment = rdMolStandardize.FragmentParent(mol)
    if not any(atom.GetAtomicNum() == 6 for atom in fragment.GetAtoms()):
        return None
    key = Chem.MolToInchiKey(rdMolStandardize.ChargeParent(fragment)).split('-')[0]
    if not key:
        return None
    return Chem.MolToSmiles(fragment, isomericSmiles=True), key


paths = {
    'octant_cyp': (root/'cycle06/cyp2c9/structures.csv', 'identity'),
    'octant_pxr': (root/'cycle06/pxr/structures.csv', 'identity'),
    'earlier_ncats': (root/'cycle08_data/NCATS_CYP2C9_candidates.csv', 'smiles'),
}
previous = {}
for label, (path, column) in paths.items():
    ids = [identify(r[column]) for r in table(path)]
    previous[label] = {x[1] for x in ids if x is not None}

groups = defaultdict(list)
counts = defaultdict(int)
for row in table(assay):
    sid = row['PUBCHEM_SID']
    if not sid.isdecimal():
        continue
    counts['source_rows'] += 1
    ident = identify(row['PUBCHEM_EXT_DATASOURCE_SMILES'])
    if ident is None or not row['PUBCHEM_CID'].isdecimal():
        counts['unusable_structure_rows'] += 1
        continue
    groups[ident[1]].append((int(sid), int(row['PUBCHEM_CID']), ident[0]))
rows = []
for key, records in sorted(groups.items()):
    exclusions = [label for label, ids in previous.items() if key in ids]
    for item in exclusions:
        counts['overlap_'+item] += 1
    if exclusions:
        counts['any_overlap'] += 1
        continue
    sids = sorted({sid for sid, _, _ in records})
    representatives = sorted({s for _, _, s in records})
    rows.append(dict(candidate_id=key,identity=representatives[0],
                     source_sids=';'.join(map(str,sids)),structure_variants=len(representatives)))
counts['unique_connectivities'] = len(groups)
counts['retained_candidates'] = len(rows)
counts['tested_budget'] = len(rows)//5
assert len(rows) >= 750
assert len({sid for row in rows for sid in row['source_sids'].split(';')}) == sum(len(row['source_sids'].split(';')) for row in rows)
hashfile = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
with (home/'external_structures.csv').open('w',encoding='utf-8',newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
counts['input_sha256'] = {s:hashfile(path) for s,(path,_) in paths.items()}
counts['assay_sha256'] = hashfile(assay)
counts['registry_sha256'] = hashfile(home/'external_structures.csv')
counts['input_fields_used'] = ['PUBCHEM_SID','PUBCHEM_CID','PUBCHEM_EXT_DATASOURCE_SMILES']
(home/'structure_qc.json').write_text(json.dumps(counts,indent=2)+'\n',encoding='utf-8')
print(json.dumps(counts,indent=2))
