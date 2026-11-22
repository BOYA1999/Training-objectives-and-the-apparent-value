from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re

home = Path(__file__).resolve().parents[1]
INHIBITORY = {-1.1, -1.2, -2.1, -2.2}
ACTIVATING = {1.1, 1.2, 2.1, 2.2}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(value):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def records(aid, sids, threshold, counter=False):
    path = home/f'source/aid{aid}_description.json'
    desc = json.loads(path.read_text(encoding='utf-8'))['PC_AssayContainer'][0]['assay']['descr']
    doses = defaultdict(list)
    for field in desc['results']:
        name = field['name']
        if name.startswith('Activity at ') and 'tc' in field:
            assert field['tc']['unit'] == 5
            slot = name.rsplit('-Replicate_', 1)[1]
            doses[slot].append((field['tc']['concentration'],name))
    for group in doses.values():
        group.sort()
    cut = math.log10(threshold) - 6
    data = defaultdict(list)
    diagnostics = Counter()
    with (home/f'source/aid{aid}.csv').open(encoding='utf-8-sig',newline='') as f:
        src = csv.DictReader(f)
        fields = set(src.fieldnames)
        names = []
        for slot, group in doses.items():
            cols = tuple(f'{name}-Replicate_{slot}' for name in
                         ('Fit_CurveClass','Fit_LogAC50','Efficacy','Excluded_Points'))
            if cols[0] not in fields:
                raise ValueError(f'Missing curve class {cols[0]}')
            if not {name for _,name in group}.issubset(fields):
                raise ValueError(f'Missing curve data in replicate {slot}')
            names.append((cols,group))
        for row in src:
            sid = row['PUBCHEM_SID']
            if sid not in sids:
                continue
            for (cc, ac, eff, mask_col), group in names:
                klass = number(row[cc])
                if klass is None:
                    continue
                diagnostics['recorded_classes'] += 1
                if klass in ACTIVATING:
                    data[sid].append('block')
                    diagnostics['activation'] += 1
                    continue
                dose = [d for d,name in group if d > 0 and number(row[name]) is not None]
                mask = (row.get(mask_col) or '').strip()
                if mask:
                    clean = re.sub(r'[\s,;\[\]]', '', mask)
                    if not re.fullmatch(r'[01]+',clean) or len(clean) != len(dose):
                        diagnostics['invalid_mask'] += 1
                        continue
                    dose = [d for d,excluded in zip(dose,clean) if excluded=='0']
                if not dose or not (dose[0] <= threshold <= dose[-1]):
                    diagnostics['no_threshold_span'] += 1
                    continue
                if klass == 4.0:
                    data[sid].append(0)
                    diagnostics['inactive_covered'] += 1
                elif klass in INHIBITORY:
                    log_ac = number(row.get(ac))
                    if log_ac is None or not (math.log10(dose[0])-6 <= log_ac <= math.log10(dose[-1])-6):
                        diagnostics['missing_or_out_of_window_fit'] += 1
                        continue
                    if counter:
                        status = 'block' if log_ac <= cut else 0
                    else:
                        efficacy = number(row.get(eff))
                        if efficacy is None or abs(efficacy) <= 65:
                            diagnostics['low_or_missing_efficacy'] += 1
                            continue
                        status = int(log_ac <= cut)
                    data[sid].append(status)
                    diagnostics['qualified_'+str(status)] += 1
                else:
                    diagnostics['other_curve_class'] += 1
    return data, diagnostics


def consensus(values, minimum):
    if len(values) >= minimum and len(set(values)) == 1 and values[0] in (0,1):
        return values[0]
    return None


def main():
    protocol = home/'protocol.json'
    frozen = home/'.aidd/candidates/tox21-cyp2c9/protocol.json'
    assert frozen.is_file()
    contract = json.loads(protocol.read_text(encoding='utf-8'))
    assert contract == json.loads(frozen.read_text(encoding='utf-8'))
    predlock = json.loads((home/'fit/prediction_lock.json').read_text(encoding='utf-8'))
    assert predlock['protocol_sha256'] == digest(protocol)
    assert predlock['sha256'] == digest(home/'fit/locked_predictions.csv')
    assert all(digest(home/name)==sha for name,sha in contract['fingerprints'].items())
    with (home/'external_structures.csv').open(encoding='utf-8',newline='') as f:
        candidates = list(csv.DictReader(f))
    sid_lists = [{s for s in r['source_sids'].split(';')} for r in candidates]
    sids = set.union(*sid_lists)
    cyp, cyp_qc = records(1671198, sids, 1.0)
    luc, luc_qc = records(1224835, sids, 20.0, counter=True)
    outcomes=[]
    for row,group in zip(candidates,sid_lists):
        same_structure = row['structure_variants']=='1'
        cyp2 = [consensus(cyp[s],2) for s in group]
        cyp3 = [consensus(cyp[s],3) for s in group]
        non_interfering2 = all(consensus(luc[s],2)==0 for s in group)
        non_interfering3 = all(consensus(luc[s],3)==0 for s in group)
        base = consensus(cyp2,len(group)) if same_structure else None
        strict_cyp = consensus(cyp3,len(group)) if same_structure else None
        outcomes.append({'candidate_id':row['candidate_id'],
            'primary_label':base if non_interfering2 else None,
            'cyp_only_label':base,
            'three_record_label':strict_cyp if non_interfering3 else None,
            'counter_clear_two':int(non_interfering2),'counter_clear_three':int(non_interfering3),
            'single_structure_variant':int(same_structure)})
    output = home/'external_outcomes.csv'
    with output.open('w',encoding='utf-8',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(outcomes[0]))
        writer.writeheader()
        writer.writerows(outcomes)
    categories = {}
    for key in ('primary_label','cyp_only_label','three_record_label','counter_clear_two','counter_clear_three',
                'single_structure_variant'):
        categories[key] = dict(Counter(str(r[key]) for r in outcomes))
    checks = {'classification':categories,'cyp_record_qc':dict(cyp_qc),
        'counter_screen_record_qc':dict(luc_qc),'source_sids':len(sids),
        'hashes':{'outcome_sha256':digest(output),'protocol_sha256':digest(protocol),
                  'prediction_sha256':predlock['sha256']}}
    (home/'adjudication_qc.json').write_text(json.dumps(checks,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(checks,indent=2))


if __name__=='__main__':
    main()
