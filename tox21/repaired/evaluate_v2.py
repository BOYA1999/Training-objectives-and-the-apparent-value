from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit.Chem.Scaffolds import MurckoScaffold

ROOT = Path(__file__).resolve().parent.parent
OBJECTIVES = ('point_l1', 'point_l2', 'interval')
REPRESENTATIONS = ('FP', 'Mol2Mol', 'ChemGPT')
SCORE_COLS = [f'{rep}_{objective}' for objective in OBJECTIVES for rep in REPRESENTATIONS]
LABEL_COLS = ['primary_label', 'cyp_only_label', 'three_record_label']
PRIMARY_SEED = 20261120
SEEDS = tuple(range(PRIMARY_SEED, PRIMARY_SEED + 5))
BOOT_REPS = 15000


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def read_locked_scores(scores_path, lock_path, protocol_path):
    lock = read_json(lock_path)
    if sha(protocol_path) != lock['protocol_sha256'] or sha(scores_path) != lock['sha256']:
        raise ValueError('Protocol or predictions changed since lock.')
    scores = pd.read_csv(scores_path, dtype={'candidate_id': str, 'identity': str})
    if scores[['candidate_id', 'identity', 'seed']].isna().any().any():
        raise ValueError('Missing candidate, identity or seed in predictions.')
    if tuple(sorted(scores.seed.unique())) != SEEDS:
        raise ValueError('Predictions do not contain the five frozen seeds.')
    if not np.isfinite(scores[SCORE_COLS].to_numpy(dtype=float)).all():
        raise ValueError('Nonfinite locked scores.')
    by_seed = {int(seed): frame.set_index('candidate_id', verify_integrity=True).sort_index()
               for seed, frame in scores.groupby('seed', sort=True)}
    first = by_seed[PRIMARY_SEED].identity
    if not all(frame.identity.equals(first) for frame in by_seed.values()):
        raise ValueError('Seed-specific candidate IDs or identities differ.')
    return by_seed, lock


def read_outcomes(path, candidate_ids):
    labels = pd.read_csv(path, dtype=str, keep_default_na=False).set_index('candidate_id', verify_integrity=True)
    if set(labels.index) != set(candidate_ids):
        raise ValueError('The adjudication table must contain every ranked candidate exactly once.')
    labels = labels.loc[candidate_ids, LABEL_COLS].apply(lambda col: col.str.strip().replace('', np.nan))
    labels = labels.apply(pd.to_numeric, errors='raise')
    if not labels.isin([0, 1]).where(labels.notna(), True).all().all():
        raise ValueError('Outcome values must be 1, 0 or blank.')
    return labels


def rank_memberships(scores, slots):
    # scores is already sorted by candidate ID; keep the fallback explicit.
    keys = scores.identity.map(lambda s: hashlib.sha256(s.encode('utf-8')).hexdigest()).to_numpy(str)
    ids = scores.index.to_numpy(str)
    choices, tied = {}, {}
    for method in SCORE_COLS:
        values = scores[method].to_numpy(dtype=float)
        order = np.lexsort((ids, keys, -values))
        select = np.zeros(len(scores), dtype=np.int8)
        select[order[:slots]] = 1
        choices[method] = select
        tied[method] = bool(slots < len(scores) and values[order[slots-1]] == values[order[slots]])
    return choices, tied


def compare(label, choices, tied, context):
    hit = label.eq(1).to_numpy()
    unknown = label.isna().to_numpy()
    tips = {method: int(v[hit].sum()) for method, v in choices.items()}
    cells = [dict(**context, method=method, hits=tips[method], unresolved=int(v[unknown].sum()),
                  boundary_tie=tied[method])
             for method, v in choices.items()]
    specs = [(rep, obj, fp, 'representation', f'{rep}_{obj}', f'FP_{fp}')
             for rep in REPRESENTATIONS[1:] for obj in OBJECTIVES for fp in OBJECTIVES]
    specs.extend(('FP', obj, fp, 'fp_objective_step', f'FP_{obj}', f'FP_{fp}')
                 for obj, fp in zip(OBJECTIVES[1:], OBJECTIVES))
    rows, contributions = [], {}
    for rep, obj, fp, kind, a, b in specs:
        name = f'{a}_vs_{b}'
        diff = choices[a] - choices[b]
        d = tips[a] - tips[b]
        unknown_a = int((diff[unknown] == 1).sum())
        unknown_b = int((diff[unknown] == -1).sum())
        lo, hi = d - unknown_b, d + unknown_a
        rows.append(dict(**context, contrast=name, kind=kind, representation=rep, model_objective=obj,
                         fp_objective=fp, same_objective=obj == fp, a_hits=tips[a], b_hits=tips[b],
                         net=d, lower=lo, upper=hi, a_only_unknown=unknown_a, b_only_unknown=unknown_b,
                         exclusive_each=int((diff == 1).sum())))
        contributions[name] = diff * hit
    return cells, rows, contributions


def boot_intervals(identity, contributions, slots):
    groups = identity.map(lambda s: MurckoScaffold.MurckoScaffoldSmiles(smiles=s) or s)
    grouped = pd.DataFrame(contributions, index=identity.index).groupby(groups).sum()
    contributions_by_group = grouped.to_numpy(dtype=float)
    n = len(grouped)
    probabilities = np.full(n, 1 / n)
    rng = np.random.default_rng(PRIMARY_SEED)
    batches = []
    for offset in range(0, BOOT_REPS, 256):
        m = min(256, BOOT_REPS - offset)
        frequencies = rng.multinomial(n, probabilities, size=m)
        batches.append(frequencies.astype(float) @ contributions_by_group)
    intervals = np.quantile(100 * np.concatenate(batches, axis=0) / slots, [0.025, 0.975], axis=0)
    return pd.DataFrame(dict(contrast=grouped.columns, boot_low_per100=intervals[0], boot_high_per100=intervals[1],
                             clusters=n, replicates=BOOT_REPS, bootstrap_seed=PRIMARY_SEED, denominator=slots))


def seed_summary(pairs):
    keys = ['label_policy', 'budget', 'slots', 'contrast']
    return (pairs.assign(observed_negative=pairs.net.lt(0), observed_zero=pairs.net.eq(0),
                         observed_positive=pairs.net.gt(0), strict_negative=pairs.upper.lt(0),
                         strict_positive=pairs.lower.gt(0), exact_zero=pairs.lower.eq(0) & pairs.upper.eq(0),
                         contains_zero=pairs.lower.le(0) & pairs.upper.ge(0))
                 .groupby(keys, sort=False).agg(n_seeds=('seed', 'size'), min_net=('net', 'min'),
                     max_net=('net', 'max'), min_bound=('lower', 'min'), max_bound=('upper', 'max'),
                     n_observed_negative=('observed_negative', 'sum'), n_observed_zero=('observed_zero', 'sum'),
                     n_observed_positive=('observed_positive', 'sum'), n_strict_negative=('strict_negative', 'sum'),
                     n_strict_positive=('strict_positive', 'sum'), n_exact_zero=('exact_zero', 'sum'),
                     n_contains_zero=('contains_zero', 'sum')).reset_index())


def replication(pairs):
    reports = []
    for (policy, budget, seed), rows in pairs.groupby(['label_policy', 'budget', 'seed'], sort=False):
        row = dict(label_policy=policy, budget=budget, seed=int(seed), slots=int(rows.slots.iloc[0]))
        for rep in REPRESENTATIONS[1:]:
            indexed = rows.set_index('contrast')
            cross = indexed.loc[f'{rep}_interval_vs_FP_point_l1']
            matched = indexed.loc[f'{rep}_interval_vs_FP_interval']
            fields = [('cross', cross), ('matched', matched)]
            for name, pair in fields:
                for value in ('net', 'lower', 'upper'):
                    row[f'{rep}_{name}_{value}'] = int(pair[value])
            row[f'{rep}_observed_sign_pattern'] = bool(cross.net > 0 and matched.net < 0)
            row[f'{rep}_strong_bounded_pattern'] = bool(cross.lower > 0 and matched.upper < 0)
        row['observed_sign_pattern_both'] = all(row[f'{rep}_observed_sign_pattern'] for rep in REPRESENTATIONS[1:])
        row['strong_bounded_pattern_both'] = all(row[f'{rep}_strong_bounded_pattern'] for rep in REPRESENTATIONS[1:])
        reports.append(row)
    return pd.DataFrame(reports)


def main():
    score_path = ROOT/'fit/locked_predictions.csv'
    lock_path = ROOT/'fit/prediction_lock.json'
    protocol_path = ROOT/'protocol.json'
    outcome_path = ROOT/'repaired/external_outcomes.csv'
    scores, lock = read_locked_scores(score_path, lock_path, protocol_path)
    catalogue = scores[PRIMARY_SEED].identity
    n = len(catalogue)
    budgets = {'20%': n // 5, '10%': n // 10, '30%': n * 3 // 10, 'fixed150': 150}
    if any(not 0 < slots <= n for slots in budgets.values()):
        raise ValueError('A frozen budget falls outside the catalogue.')
    outcomes_hash = sha(outcome_path)
    outcomes = read_outcomes(outcome_path, catalogue.index)
    cell_rows, contrast_rows = [], []
    for seed, frame in scores.items():
        for budget, slots in budgets.items():
            choices, tied = rank_memberships(frame, slots)
            for policy in LABEL_COLS:
                context = dict(label_policy=policy, seed=seed, budget=budget, candidates=n, slots=slots)
                cells, rows, contributions = compare(outcomes[policy], choices, tied, context)
                cell_rows.extend(cells)
                contrast_rows.extend(rows)
                if (seed, budget, policy) == (PRIMARY_SEED, '20%', 'primary_label'):
                    boot = boot_intervals(catalogue, contributions, slots)
    pairs = pd.DataFrame(contrast_rows)
    if len(pairs) != 20 * len(SEEDS) * len(budgets) * len(LABEL_COLS):
        raise ValueError('A frozen contrast is missing.')
    signs = replication(pairs)
    primary = pairs.query('seed == @PRIMARY_SEED and budget == "20%" and label_policy == "primary_label"')
    primary = primary.merge(boot, on='contrast', validate='one_to_one')
    primary_replication = signs.query('seed == @PRIMARY_SEED and budget == "20%" and label_policy == "primary_label"').iloc[0]
    if sha(outcome_path) != outcomes_hash or sha(protocol_path) != lock['protocol_sha256'] or sha(score_path) != lock['sha256']:
        raise ValueError('Input changed while evaluating.')
    destination = ROOT/'repaired/evaluation'
    destination.mkdir(exist_ok=True)
    for name, table in {'cells': pd.DataFrame(cell_rows), 'contrasts': pairs, 'seed_summary': seed_summary(pairs),
                        'replication': signs, 'primary_scaffold_intervals': primary}.items():
        table.to_csv(destination/f'{name}.csv', index=False)
    summary = dict(evaluated_utc=datetime.now(timezone.utc).isoformat(), candidates=n, budgets=budgets,
        score_sha256=lock['sha256'], protocol_sha256=lock['protocol_sha256'], outcome_sha256=outcomes_hash,
        tied_boundaries=int(pd.DataFrame(cell_rows).query('label_policy == "primary_label"').boundary_tie.sum()),
        outcomes={policy:{'supported_hit': int(outcomes[policy].eq(1).sum()),
                          'supported_nonhit': int(outcomes[policy].eq(0).sum()),
                          'unresolved': int(outcomes[policy].isna().sum())} for policy in LABEL_COLS},
        primary_replication=json.loads(primary_replication.to_json()))
    (destination/'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
