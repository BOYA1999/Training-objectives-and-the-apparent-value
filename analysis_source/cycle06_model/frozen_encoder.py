import sys
sys.dont_write_bytecode = True
from pathlib import Path
import time
import hashlib
import json
import argparse
import torch
import numpy as np
import pandas as pd
from rdkit import Chem

ROOT = Path(__file__).resolve().parent
MODEL_ROOT = ROOT.parent / 'models_outputs'
sys.path.insert(0, str(MODEL_ROOT))
from mol2mol_cpu import load_model


class FrozenMol2MolEncoder:
    def __init__(self):
        torch.set_num_threads(2)
        self.model, _ = load_model()
        self.model.network.eval().requires_grad_(False)
        self.dimension = self.model.network.model_dimension

    def encode(self, table, batch_size=32, canonicalize=True):
        rows = table[['candidate_id', 'smiles']].copy().reset_index(drop=True)
        assert rows.candidate_id.notna().all() and rows.candidate_id.is_unique
        tokens, positions, encoded_smiles, states, lengths = [], [], [], [], []
        vocab = self.model.vocabulary
        for i, smi in enumerate(rows.smiles):
            mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
            text = Chem.MolToSmiles(mol, isomericSmiles=True) if mol is not None and canonicalize else smi
            seq = self.model.tokenizer.tokenize(text) if mol is not None and mol.GetNumAtoms() else []
            status = 'invalid_smiles' if not seq else 'ok'
            if seq and any(t not in vocab for t in seq):
                status = 'out_of_vocabulary'
            if len(seq) > self.model.max_sequence_length:
                status = 'over_max_sequence_length'
            encoded_smiles.append(text)
            states.append(status)
            lengths.append(len(seq))
            if status == 'ok':
                positions.append(i)
                tokens.append(torch.tensor([vocab[t] for t in seq], dtype=torch.long))
        mean = np.full((len(rows), self.dimension), np.nan, dtype=np.float32)
        eos = np.full_like(mean, np.nan)
        with torch.inference_mode():
            for start in range(0, len(tokens), batch_size):
                seqs = tokens[start:start + batch_size]
                src = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0)
                mask = src.ne(0)
                hidden = self.model.network.encode(src, mask[:, None, :])
                content = mask & src.ne(vocab['^']) & src.ne(vocab['$'])
                pooled = (hidden * content[:, :, None]).sum(1) / content.sum(1)[:, None]
                final = hidden[torch.arange(len(seqs)), torch.tensor([len(x) - 1 for x in seqs])]
                idx = positions[start:start + batch_size]
                mean[idx], eos[idx] = pooled.cpu().numpy(), final.cpu().numpy()
        rows['encoded_smiles'], rows['status'], rows['token_count_with_specials'] = encoded_smiles, states, lengths
        return {'mean': mean, 'eos': eos}, rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    assert Path(args.name).name == args.name
    start = time.perf_counter()
    encoder = FrozenMol2MolEncoder()
    load_seconds = time.perf_counter() - start
    table = pd.read_csv(args.input, usecols=['candidate_id', 'smiles'], dtype={'candidate_id': str})
    start = time.perf_counter()
    features, rows = encoder.encode(table)
    encode_seconds = time.perf_counter() - start
    prefix = ROOT / args.name
    np.savez_compressed(str(prefix) + '.npz', candidate_id=rows.candidate_id.to_numpy(dtype=str), **features)
    rows.to_csv(str(prefix) + '_rows.csv', index=False)
    report = dict(input_sha256=hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
                  prior_sha256=hashlib.sha256((MODEL_ROOT / 'mol2mol_similarity.prior').read_bytes()).hexdigest(),
                  n=len(rows), status_counts=rows.status.value_counts().to_dict(), dimension=encoder.dimension,
                  load_seconds=load_seconds, encode_seconds=encode_seconds, threads=torch.get_num_threads(),
                  pooling='final_layer_norm_encoder_content_token_mean_and_eos', canonicalize=True)
    (ROOT / (args.name + '_manifest.json')).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))
