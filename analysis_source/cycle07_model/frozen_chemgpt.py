import sys
sys.dont_write_bytecode = True
from pathlib import Path
import importlib.util
import hashlib
import json
import torch
import numpy as np
import pandas as pd
from rdkit import Chem
from transformers import GPTNeoConfig, GPTNeoModel, PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / 'checkpoint'
spec = importlib.util.spec_from_file_location('chemgpt_selfies104', ROOT / 'vendor/selfies/__init__.py',
                                             submodule_search_locations=[str(ROOT / 'vendor/selfies')])
sf = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = sf
spec.loader.exec_module(sf)


class FrozenChemGPTEncoder:
    def __init__(self):
        assert hashlib.sha256((CHECKPOINT / 'pytorch_model.bin').read_bytes()).hexdigest() == 'a8a39c5ea1d50c450632e003d21c90f847d815a801934b03a12b02010b74d21b'
        torch.set_num_threads(2)
        config = json.loads((CHECKPOINT / 'config.json').read_text())
        config.update(bos_token_id=2, eos_token_id=3, pad_token_id=1)
        self.model, self.loading_info = GPTNeoModel.from_pretrained(CHECKPOINT, config=GPTNeoConfig(**config),
            local_files_only=True, attn_implementation='eager', output_loading_info=True)
        assert not self.loading_info['missing_keys']
        assert all(k.endswith('.attn.attention.bias') or k.endswith('.attn.attention.masked_bias')
                   for k in self.loading_info['unexpected_keys'])
        self.model.eval().requires_grad_(False)
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(CHECKPOINT / 'tokenizer.json'),
            pad_token='[PAD]', unk_token='[UNK]', cls_token='[CLS]', sep_token='[SEP]', model_max_length=512)
        self.dimension = self.model.config.hidden_size

    def encode(self, table, batch_size=16):
        rows = table[['candidate_id', 'smiles']].copy().reset_index(drop=True)
        assert rows.candidate_id.notna().all() and rows.candidate_id.is_unique
        positions, sequences, audit = [], [], []
        for i, smi in enumerate(rows.smiles):
            mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
            canonical = Chem.MolToSmiles(mol, isomericSmiles=True) if mol is not None else ''
            text = sf.encoder(canonical) if canonical else None
            status = 'invalid_smiles' if not canonical else 'selfies_conversion_failed' if not text else 'ok'
            ids = self.tokenizer(text, add_special_tokens=True)['input_ids'] if text else []
            if text:
                decoded = sf.decoder(text)
                back = Chem.MolFromSmiles(decoded) if decoded else None
                if back is None or Chem.MolToSmiles(back, isomericSmiles=True) != canonical:
                    status = 'selfies_roundtrip_mismatch'
                elif self.tokenizer.unk_token_id in ids:
                    status = 'unknown_token'
                elif len(ids) > 512:
                    status = 'over_max_sequence_length'
            audit.append((canonical, text, status, len(ids)))
            if status == 'ok':
                positions.append(i)
                sequences.append(torch.tensor(ids, dtype=torch.long))
        result = np.full((len(rows), self.dimension), np.nan, dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, len(sequences), batch_size):
                seqs = sequences[start:start + batch_size]
                src = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=1)
                mask = src.ne(1)
                hidden = self.model(input_ids=src, attention_mask=mask, use_cache=False).last_hidden_state
                pooled = hidden[torch.arange(len(seqs)), mask.sum(1) - 1]
                result[positions[start:start + batch_size]] = pooled.cpu().numpy()
        rows[['encoded_smiles', 'selfies', 'status', 'token_count_with_specials']] = pd.DataFrame(audit)
        return {'mean': result}, rows
