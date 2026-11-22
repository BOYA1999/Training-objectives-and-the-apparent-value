import sys, json, hashlib, time, argparse
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path[:0] = [str(BASE / 'reinvent_minimal'), str(BASE / 'local_packages')]
import torch
from rdkit import Chem
from reinvent.models.transformer.mol2mol.mol2mol import Mol2MolModel
from reinvent.models.transformer.core.dataset.dataset import Dataset
import reinvent.models.transformer.core.vocabulary as vocabulary_module
sys.modules['reinvent.models.mol2mol.models.vocabulary'] = vocabulary_module


def load_model():
    path = BASE / 'mol2mol_similarity.prior'
    assert hashlib.md5(path.read_bytes()).hexdigest() == '451893c2d1818d9164bc73ea5a5bd7ef'
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model = Mol2MolModel.create_from_dict(checkpoint, 'inference', torch.device('cpu'))
    return model, checkpoint


def generate(model, smiles, n=16, seed=20261120, batch_size=16):
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    rows = []
    for parent_index, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        assert mol is not None, smi
        canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
        for offset in range(0, n, batch_size):
            size = min(batch_size, n-offset)
            ds = Dataset([canonical] * size, model.vocabulary, model.tokenizer)
            assert len(ds) == size
            src, mask = Dataset.collate_fn([ds[i] for i in range(size)])
            ins, outs, nlls = model.sample(src, mask, 'multinomial')
            for i, (inp, out, nll) in enumerate(zip(ins, outs, nlls)):
                product = Chem.MolFromSmiles(out)
                rows.append(dict(parent_index=parent_index, draw_index=offset+i, input_smiles=inp, output_smiles=out, canonical_output=Chem.MolToSmiles(product, isomericSmiles=True) if product else None, nll=float(nll), valid=product is not None))
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--n', type=int, default=16)
    parser.add_argument('--seed', type=int, default=20261120)
    args = parser.parse_args()
    start = time.perf_counter()
    model, checkpoint = load_model()
    smiles = [s.strip().split()[0] for s in Path(args.input).read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    rows = generate(model, smiles, args.n, args.seed)
    Path(args.output).write_text(json.dumps(dict(rows=rows, elapsed_seconds=time.perf_counter()-start, seed=args.seed, n=args.n, metadata=checkpoint['metadata'].as_dict()), ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(draws=len(rows), valid=sum(r['valid'] for r in rows), elapsed_seconds=time.perf_counter()-start)))

