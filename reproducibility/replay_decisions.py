from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

here = Path(__file__).resolve().parent


def read(name):
    with (here / name).open(encoding='utf-8-sig', newline='') as source:
        return list(csv.DictReader(source))


def compare(a, b, labels):
    ha = sum(labels[c] == '1.0' for c in a)
    hb = sum(labels[c] == '1.0' for c in b)
    ua = sum(labels[c] == '' for c in a - b)
    ub = sum(labels[c] == '' for c in b - a)
    return ha, hb, ha - hb, ha - hb - ub, ha - hb + ua, ua, ub


lock = json.loads((here / 'objectives/prediction_lock.json').read_text(encoding='utf-8'))
assert hashlib.sha256((here / 'objectives/locked_predictions.csv').read_bytes()).hexdigest() == lock['sha256']
labels = {row['candidate_id']: row['hit'] for row in read('late_labels.csv')}
scores_by_seed = defaultdict(list)
for row in read('objectives/locked_predictions.csv'):
    row['tie'] = hashlib.sha256(row['identity'].encode()).hexdigest()
    scores_by_seed[row['seed']].append(row)
assert set(scores_by_seed) == {str(seed) for seed in range(20261120, 20261125)}
lists = {}
for seed, rows in scores_by_seed.items():
    assert len(rows) == len(labels) == 750
    for score in lock['score_columns']:
        top = sorted(rows, key=lambda row: (-float(row[score]), row['tie']))[:150]
        lists[seed, score] = {r['candidate_id'] for r in top}
matches = []
for row in read('objectives/seed_pairs.csv'):
    a = lists[row['seed'], row['model'] + '_' + row['model_objective']]
    b = lists[row['seed'], 'FP_' + row['fp_objective']]
    values = compare(a, b, labels)
    keys = ['model_hits','fp_hits','net','lower','upper','fused_only_unknown','fp_only_unknown']
    assert values == tuple(int(row[k]) for k in keys)
    if row['same_objective'] == 'True':
        matches.append(values)
assert len(matches) == 30 and all(x[2] < 0 for x in matches)
assert sum(x[4] < 0 for x in matches) == 29

members = defaultdict(dict)
for row in read('selected_memberships.csv'):
    key = (row['endpoint'], row['model'], row['budget_percent'], row['method'])
    members[key][row['candidate_id']] = row['supported_outcome']
directions = Counter()
for row in read('cross_endpoint_matched_interval.csv'):
    key = (row['endpoint'], row['model'], row['budget_percent'])
    a, b = (members[key + (method,)] for method in ('embedding','FP'))
    assert len(a) == len(b) == int(row['slots'])
    assert all(a[c] == b[c] for c in a.keys() & b.keys())
    values = compare(set(a), set(b), a | b)
    keys = ['embedding_supported','fp_supported','delta','net_lower','net_upper',
            'embedding_only_unknown','fp_only_unknown']
    assert values == tuple(int(row[k]) for k in keys)
    _, _, _, lo, hi, _, _ = values
    directions['positive' if lo > 0 else 'negative' if hi < 0 else 'zero' if lo == hi == 0 else 'unresolved'] += 1
assert directions == {'positive': 5, 'negative': 10, 'zero': 1, 'unresolved': 14}
archive = read('archives/cyp_prediction_manifest.csv')
assert len(archive) == len({row['sha256'] for row in archive}) == 62
assert len(read('archives/pxr_prediction_manifest.csv')) == 3
assert all(int(r['net_lower']) < 0 < int(r['net_upper']) for r in read('archives/ncats_full_catalogue_pairs.csv'))
print('PASS: 45 locked shortlists, 90 seeded contrasts, 30 endpoint contrasts, archive inventories and NCATS bounds.')
