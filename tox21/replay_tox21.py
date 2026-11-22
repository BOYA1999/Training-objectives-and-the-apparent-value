from collections import Counter, defaultdict
import csv
import hashlib
import json
from math import isfinite
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OBJECTIVES = ('point_l1', 'point_l2', 'interval')
REPS = ('FP', 'Mol2Mol', 'ChemGPT')
METHODS = tuple(f'{rep}_{obj}' for obj in OBJECTIVES for rep in REPS)
POLICIES = ('primary_label', 'cyp_only_label', 'three_record_label')
CONTRASTS = [(f'{rep}_{a}', f'FP_{b}') for rep in REPS[1:] for a in OBJECTIVES for b in OBJECTIVES]
CONTRASTS.extend(('FP_' + a, 'FP_' + b) for a, b in zip(OBJECTIVES[1:], OBJECTIVES))
SEEDS = range(20261120, 20261125)


def rows(path):
    with (ROOT / path).open(encoding='utf-8-sig', newline='') as source:
        return list(csv.DictReader(source))


def read_json(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8-sig'))


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


for name, expected in read_json('SHA256.json').items():
    assert digest(name) == expected, name

lock = read_json('fit/prediction_lock.json')
summary = read_json('repaired/evaluation/summary.json')
for name, expected in (
    ('protocol.json', lock['protocol_sha256']),
    ('fit/locked_predictions.csv', lock['sha256']),
    ('repaired/external_outcomes.csv', summary['outcome_sha256']),
):
    assert digest(name) == expected, name

outcome_rows = rows('repaired/external_outcomes.csv')
labels = {r['candidate_id']: r for r in outcome_rows}
assert len(outcome_rows) == len(labels) == summary['candidates'] == 4428
hit_sets, unknown_sets = {}, {}
for policy in POLICIES:
    classes = Counter(r[policy] for r in outcome_rows)
    assert set(classes) <= {'0', '1', ''}
    assert dict(supported_hit=classes['1'], supported_nonhit=classes['0'], unresolved=classes['']) == summary['outcomes'][policy]
    hit_sets[policy] = {key for key, row in labels.items() if row[policy] == '1'}
    unknown_sets[policy] = {key for key, row in labels.items() if row[policy] == ''}

score_rows = defaultdict(list)
for row in rows('fit/locked_predictions.csv'):
    score_rows[int(row['seed'])].append(row)
assert sorted(score_rows) == list(SEEDS)
identity = None
cell_checks, contrast_checks, replication_checks = {}, {}, {}
for seed, batch in score_rows.items():
    observed = {r['candidate_id']: r['identity'] for r in batch}
    assert len(batch) == len(observed) == len(labels) and set(observed) == set(labels)
    assert identity is None or observed == identity
    identity = observed
    tie = {key: hashlib.sha256(val.encode('utf-8')).hexdigest() for key, val in identity.items()}
    values = {method: {r['candidate_id']: float(r[method]) for r in batch} for method in METHODS}
    assert all(isfinite(v) for by_method in values.values() for v in by_method.values())
    ranks = {method: sorted(identity, key=lambda key: (-values[method][key], tie[key], key)) for method in METHODS}
    for budget, slots in summary['budgets'].items():
        selected = {method: set(rank[:slots]) for method, rank in ranks.items()}
        for policy in POLICIES:
            key = (policy, seed, budget, slots)
            hits = {method: len(panel & hit_sets[policy]) for method, panel in selected.items()}
            for method, panel in selected.items():
                edge = ranks[method][slots-1:slots+1]
                boundary_tie = values[method][edge[0]] == values[method][edge[1]]
                cell_checks[(*key, method)] = (hits[method], len(panel & unknown_sets[policy]), str(boundary_tie))
            for a, b in CONTRASTS:
                only_a, only_b = selected[a]-selected[b], selected[b]-selected[a]
                ua, ub = len(only_a & unknown_sets[policy]), len(only_b & unknown_sets[policy])
                d = hits[a] - hits[b]
                contrast_checks[(*key, f'{a}_vs_{b}')] = (hits[a], hits[b], d, d-ub, d+ua, ua, ub, len(only_a))
            row = {}
            for rep in REPS[1:]:
                for kind, baseline in (('cross', 'point_l1'), ('matched', 'interval')):
                    record = contrast_checks[(*key, f'{rep}_interval_vs_FP_{baseline}')]
                    row.update({f'{rep}_{kind}_{field}': v for field, v in zip(('net', 'lower', 'upper'), record[2:5])})
                row[f'{rep}_observed_sign_pattern'] = row[f'{rep}_cross_net'] > 0 and row[f'{rep}_matched_net'] < 0
                row[f'{rep}_strong_bounded_pattern'] = row[f'{rep}_cross_lower'] > 0 and row[f'{rep}_matched_upper'] < 0
            for name in ('observed_sign_pattern', 'strong_bounded_pattern'):
                row[f'{name}_both'] = all(row[f'{rep}_{name}'] for rep in REPS[1:])
            replication_checks[key] = row


def context(row):
    return row['label_policy'], int(row['seed']), row['budget'], int(row['slots'])


actual_cells = {(*context(r), r['method']): (int(r['hits']), int(r['unresolved']), r['boundary_tie'])
                for r in rows('repaired/evaluation/cells.csv')}
actual_contrasts = {(*context(r), r['contrast']): tuple(int(r[field]) for field in (
    'a_hits', 'b_hits', 'net', 'lower', 'upper', 'a_only_unknown', 'b_only_unknown', 'exclusive_each'))
    for r in rows('repaired/evaluation/contrasts.csv')}
assert actual_cells == cell_checks, 'cell table differs'
assert actual_contrasts == contrast_checks, 'contrast table differs'
actual_replication = {context(r): {field: r[field] for field in replication_checks[context(r)]}
                      for r in rows('repaired/evaluation/replication.csv')}
assert actual_replication == {k: {f: str(v) for f, v in r.items()} for k, r in replication_checks.items()}
primary = context(summary['primary_replication'])
assert all(summary['primary_replication'][f] == v for f, v in replication_checks[primary].items())

print(f'PASS: {len(score_rows)} seeds, {sum(map(len, score_rows.values())):,} scored rows, '
      f'{len(cell_checks):,} shortlist cells, {len(contrast_checks):,} paired contrasts, '
      f'{len(replication_checks)} sign-pattern checks.')
print('Primary outcome counts:', summary['outcomes']['primary_label'])
print('Primary contrasts:', replication_checks[primary])
