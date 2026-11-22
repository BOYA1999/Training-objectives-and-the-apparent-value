import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root.parent / 'models_outputs'))
from mol2mol_cpu import load_model
from reinvent.models.transformer.core.dataset.paired_dataset import PairedDataset

RDLogger.DisableLog('rdApp.*')
torch.set_num_threads(2)

def standardize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = rdMolStandardize.FragmentParent(mol)
    if not any(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms()):
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True)

def conditional_scores(source, target):
    model, _ = load_model()
    model.network.eval()
    vocab = model.vocabulary
    tokenize = model.tokenizer.tokenize
    pairs = [(i, x, y, tokenize(y)) for i, (x, y) in enumerate(zip(source, target))
             if all(token in vocab for token in tokenize(x) + tokenize(y))
             and max(len(tokenize(x)), len(tokenize(y))) <= model.max_sequence_length]
    scores = np.full(len(source), np.nan)
    ds = PairedDataset([x for _, x, _, _ in pairs], [y for _, _, y, _ in pairs], vocab, model.tokenizer)
    assert len(ds) == len(pairs)
    with torch.inference_mode():
        for offset in range(0, len(ds), 16):
            stop = min(len(ds), offset + 16)
            batch = ds.collate_fn([ds[i] for i in range(offset, stop)])
            values = model.likelihood(batch.input, batch.input_mask, batch.output, batch.output_mask)
            for (i, _, _, tokens), value in zip(pairs[offset:stop], values.tolist()):
                scores[i] = -value / (len(tokens) - 1)
    return scores

def rank(d, column):
    return d.groupby('parent_id')[column].rank(pct=True, method='average')

def score(project):
    start = time.perf_counter()
    path = root / f'{project}_structures.csv'
    d = pd.read_csv(path).sort_values('candidate_id')
    assert d.candidate_id.is_unique
    d['parent'] = d.parent_smiles.map(standardize)
    d['candidate'] = d.smiles.map(standardize)
    parents = d.dropna(subset=['parent'])[['parent_id','parent']].drop_duplicates()
    assert parents.parent_id.is_unique
    raw_n = len(d)
    d = d.dropna(subset=['parent','candidate'])
    n_valid = len(d)
    d = d[d.parent != d.candidate].copy()
    n_nonseed = len(d)
    d['same_parent_duplicate'] = d.duplicated(['parent_id','candidate'])
    d[['candidate_id','parent_id','candidate','same_parent_duplicate']].to_csv(
        root / f'{project}_structure_dedup.csv', index=False)
    d = d[~d.same_parent_duplicate].copy()
    n_unique = len(d)
    d = d[d.groupby('parent_id').candidate.transform('size') >= 5].copy()
    parent_fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    parent_fp = {s: parent_fp_gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in parents.parent.unique()}
    alt = {}
    for s in parent_fp:
        options = sorted(set(parent_fp) - {s})
        alt[s] = max(options, key=lambda x: DataStructs.TanimotoSimilarity(parent_fp[s], parent_fp[x]))
    d['replacement'] = d.parent.map(alt)
    d['series_scaffold'] = d.parent.map(
        lambda s: MurckoScaffold.MurckoScaffoldSmiles(smiles=s) or s)
    sources = d.parent.to_list() + d.replacement.to_list()
    values = conditional_scores(sources, d.candidate.to_list() * 2)
    d['true_NLL'] = values[:len(d)]
    d['replacement_NLL'] = values[len(d):]
    d['contrast_NLL'] = d.true_NLL - d.replacement_NLL

    p, q, a = [[Chem.MolFromSmiles(s) for s in d[c]] for c in ['parent','candidate','replacement']]
    fp = {
        'ECFP4': parent_fp_gen.GetFingerprint,
        'FeatureMorgan': rdFingerprintGenerator.GetMorganGenerator(
            radius=2, fpSize=2048, atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen(),
            includeChirality=True).GetFingerprint,
        'AtomPair': rdFingerprintGenerator.GetAtomPairGenerator(fpSize=2048, includeChirality=True).GetFingerprint,
        'MACCS': MACCSkeys.GenMACCSKeys,
    }
    for name, fn in fp.items():
        d[name] = [DataStructs.TanimotoSimilarity(fn(x), fn(y)) for x, y in zip(p, q)]
    d['ECFP4_contrast'] = d.ECFP4 - np.array([
        DataStructs.TanimotoSimilarity(fp['ECFP4'](x), fp['ECFP4'](y)) for x, y in zip(a,q)])
    d['fingerprint_consensus'] = pd.concat([rank(d, k) for k in fp], axis=1).mean(axis=1)
    for name, signal in {
        'AI_contrast': 'contrast_NLL',
        'AI_no_contrast': 'true_NLL',
        'AI_replaced_parent': 'replacement_NLL',
        'fingerprint_contrast': 'ECFP4_contrast',
    }.items():
        aux = rank(d, signal).fillna(d.fingerprint_consensus)
        d[name] = (d.fingerprint_consensus + aux) / 2
    d['budget'] = (d.groupby('parent_id')['candidate'].transform('size') // 5).clip(lower=1)
    methods = list(fp) + ['fingerprint_consensus','AI_contrast','AI_no_contrast',
                         'AI_replaced_parent','fingerprint_contrast']
    for method in methods:
        ordered = d.sort_values([method, 'candidate_id'], ascending=[False,True])
        d['select_' + method] = (
            ordered.groupby('parent_id').cumcount() < ordered.budget).reindex(d.index)
    assert d.groupby('parent_id')[['select_' + m for m in methods]].sum().nunique(axis=1).eq(1).all()
    output = root / f'{project}_locked_scores.csv'
    d.drop(columns=['smiles','parent_smiles']).to_csv(output, index=False)
    log = dict(project=project, raw=raw_n, valid=n_valid, nonself=n_nonseed, unique_within_parent=n_unique,
               eligible=len(d), parent_series=int(d.parent_id.nunique()), independent_scaffold_groups=int(d.series_scaffold.nunique()),
               selected=int(d.select_AI_contrast.sum()), model_pair_coverage=int(d.contrast_NLL.notna().sum()),
               input_hash=hashlib.sha256(path.read_bytes()).hexdigest(),
               score_hash=hashlib.sha256(output.read_bytes()).hexdigest(), seconds=time.perf_counter()-start)
    (root / f'{project}_score_manifest.json').write_text(json.dumps(log, indent=2),encoding='utf-8')
    print(json.dumps(log, indent=2), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('project')
    score(parser.parse_args().project)
