from pathlib import Path
import csv
import hashlib
import subprocess
import sys

root = Path(__file__).resolve().parent
with (root/'FILE_MANIFEST.csv').open(encoding='utf-8-sig', newline='') as source:
    manifest = list(csv.DictReader(source))
for row in manifest:
    path = root/row['path']
    assert path.is_file(), row['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256'], row['path']
print(f'PASS: {len(manifest)} release files verified.', flush=True)
for replay in ('reproducibility/replay_decisions.py', 'tox21/replay_tox21.py'):
    subprocess.run([sys.executable, str(root/replay)], cwd=root, check=True)
