"""Recompute voting endpoints from the bundled terminal-vote projection (stdlib only)."""
import csv
import gzip
import json
import math
import re

METHODS = ('BlindGuard', 'CASPIAN', 'DOMINANT', 'PREM', 'TAM', 'XG-Guard')
CONTROL = 'no_defense_baseline'
DATASETS = ('MMLU', 'MMLUPRO', 'CSQA', 'GSM8K')
GRAPHS = ('chain', 'random', 'star', 'tree')
METRICS = ('published', 'actual', 'disagreement', 'false_credit', 'missed_credit', 'perturbable', 'alerts')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def winner(votes):
    counts = {}
    for vote in votes:
        counts[vote] = counts.get(vote, 0) + 1
    require(bool(counts), 'Empty electorate')
    answer = max(counts, key=counts.get)
    unique = sum(n == counts[answer] for n in counts.values()) == 1
    return answer, unique


def correct(value, gold, dataset):
    return value == gold if dataset == 'GSM8K' else value.upper() == gold.upper()


def is_canonical(row, legacy=False):
    pattern = {'GSM8K': r'(0|-?[1-9][0-9]*)', 'MMLUPRO': r'[A-J]',
               'CSQA': r'[A-D]' if legacy else r'[A-E]'}.get(row['dataset'], r'[A-D]')
    return all(re.fullmatch(pattern, answer) is not None
               for answer in [row['gold'], *row['votes']])


def replacements(row):
    # Preserve the frozen stress test, including its original candidate set.
    values = set(row['votes']) | {row['gold']}
    if row['dataset'] == 'GSM8K':
        fresh = 0
        while str(fresh) in values:
            fresh += 1
        return sorted(values | {str(fresh)})
    alphabet = 'ABCDEFGHIJ' if row['dataset'] == 'MMLUPRO' else 'ABCD'
    fresh = next((letter for letter in alphabet if letter not in values), None)
    return sorted(values | ({fresh} if fresh is not None else set()))


def summary(rows):
    return {'debates': len(rows), **{key: sum(row[key] for row in rows) for key in METRICS}}


def strict_summary(rows, legacy):
    subset = [row for row in rows if row['method'] != CONTROL and is_canonical(row, legacy)
              and winner(row['votes'])[1]
              and winner([answer for answer, flag in zip(row['votes'], row['flagged']) if not flag])[1]]
    return {'debates': len(subset), 'discordant': sum(row['disagreement'] for row in subset),
            'false_credit': sum(row['false_credit'] for row in subset),
            'missed_credit': sum(row['missed_credit'] for row in subset)}


def check(projection, expected_dir, out):
    with gzip.open(projection, 'rt', encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream]
    expected = json.loads((expected_dir / 'GAMMAF_RESULTS.json').read_text())
    frozen = json.loads((expected_dir / 'MMLU_SELECTION.json').read_text())
    sql_receipt = json.loads((expected_dir / 'VOTE_VERIFICATION.json').read_text())
    ids = set()
    for row in rows:
        identity = (row['dataset'], row['method'], row['item'], row['graph'])
        require(identity not in ids, 'Duplicate terminal debate')
        ids.add(identity)
        require(row['dataset'] in DATASETS and row['method'] in (*METHODS, CONTROL)
                and row['graph'] in GRAPHS, 'Unknown debate identity')
        votes, flags = row['votes'], row['flagged']
        require(len(votes) == len(flags) == 10 and set(flags) <= {0, 1} and 0 in flags,
                'Invalid terminal electorate')
        whole = winner(votes)[0]
        delivered = winner([answer for answer, flag in zip(votes, flags) if not flag])[0]
        published, actual = (int(correct(answer, row['gold'], row['dataset']))
                             for answer in (whole, delivered))
        require(whole == row['all_answer'] and delivered == row['delivered'], 'Winner mismatch')
        sensitive = False
        for replacement in replacements(row):
            altered = [replacement if flag else answer for answer, flag in zip(votes, flags)]
            require(winner([answer for answer, flag in zip(altered, flags) if not flag])[0] == delivered,
                    'Excluded-vote edit changed delivery')
            sensitive |= correct(winner(altered)[0], row['gold'], row['dataset']) != bool(published)
        recomputed = {'published': published, 'actual': actual, 'disagreement': int(published != actual),
                      'false_credit': int(published and not actual),
                      'missed_credit': int(actual and not published),
                      'perturbable': int(sensitive), 'alerts': sum(flags)}
        require(all(row[key] == value for key, value in recomputed.items()),
                'Stored endpoint mismatch: ' + str(identity))
    require(len(rows) == expected['debates'] == sql_receipt['sql_final_decisions_reconciled'] == 11200,
            'Wrong number of terminal decisions')
    by_task = {}
    for dataset in DATASETS:
        by_task[dataset] = {}
        for method in (*METHODS, CONTROL):
            subset = [row for row in rows if row['dataset'] == dataset and row['method'] == method]
            require(len(subset) == 400 and len({row['item'] for row in subset}) == 100,
                    'Wrong repeated-question grid')
            by_task[dataset][method] = summary(subset)
            require(by_task[dataset][method] == expected['by_task_method'][dataset][method],
                    'Task-method summary differs')
            for graph in GRAPHS:
                cell = summary([row for row in subset if row['graph'] == graph])
                require(cell == expected['by_graph'][dataset][method][graph], 'Topology summary differs')
    counts = {method: by_task['MMLU'][method] for method in METHODS}
    selected = {}
    for endpoint in ('published', 'actual'):
        best = max(counts[method][endpoint] for method in METHODS)
        ties = sorted(method for method in METHODS if counts[method][endpoint] == best)
        selected[endpoint] = {'choice': ties[0], 'ties': ties, 'correct_of_400': best}
    require({'development': 'MMLU', 'counts': counts, 'selected': selected} == frozen,
            'Frozen development selection differs')
    legacy = strict_summary(rows, legacy=True)
    receipt = {key: sum(cell[source] for methods in sql_receipt['by_task_method'].values()
                        for method, cell in methods.items() if method != CONTROL)
               for key, source in [('debates', 'canonical_untied'), ('discordant', 'strict_discordant'),
                                   ('false_credit', 'strict_false_credit'), ('missed_credit', 'strict_missed_credit')]}
    require(legacy == receipt, 'Frozen SQL sensitivity check differs')
    alphabet_aware = strict_summary(rows, legacy=False)
    chosen = [selected[endpoint]['choice'] for endpoint in ('published', 'actual')]
    paired = []
    gains = {}
    for dataset in DATASETS[1:]:
        methods = [{(row['item'], row['graph']): row for row in rows
                    if row['dataset'] == dataset and row['method'] == method} for method in chosen]
        require(methods[0].keys() == methods[1].keys(), 'Unpaired selected detectors')
        diffs = []
        for item, graph in sorted(methods[0]):
            a, b = [method[item, graph] for method in methods]
            require(a['gold'] == b['gold'], 'Gold labels differ across selected detectors')
            difference = b['actual'] - a['actual']
            paired.append({'dataset': dataset, 'item': item, 'graph': graph,
                           'published_choice_correct': a['actual'],
                           'delivered_choice_correct': b['actual'], 'difference': difference})
            diffs.append(difference)
        gains[dataset] = sum(diffs) / len(diffs) * 100
        require(math.isclose(gains[dataset], expected['selection_outcome']['tasks'][dataset]['gain_pp'], abs_tol=1e-12),
                'Validation effect differs')
    with (out / 'paired_questions.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=paired[0].keys())
        writer.writeheader()
        writer.writerows(paired)
    cases = sorted([row for row in rows if row['disagreement']],
                   key=lambda row: (row['dataset'], row['method'], row['item'], row['graph']))
    (out / 'discordant_cases.json').write_text(json.dumps(cases, indent=2, sort_keys=True) + '\n')
    defended = [row for row in rows if row['method'] != CONTROL]
    controls = [row for row in rows if row['method'] == CONTROL]
    require(all(row['disagreement'] == row['perturbable'] == row['alerts'] == 0 for row in controls),
            'Disabled-detector negative control failed')
    result = {'status': 'PASS', 'decisions_recomputed': len(rows), 'defended': summary(defended),
              'disabled_controls': summary(controls), 'frozen_selection': selected,
              'validation_gain_pp': gains, 'mean_validation_gain_pp': sum(gains.values()) / len(gains),
              'frozen_narrow_alphabet_no_ties': legacy,
              'dataset_alphabet_no_ties': alphabet_aware,
              'bootstrap_intervals_recomputed': False,
              'note': 'Offline projection replay; source-byte and bootstrap verification require --from-source.'}
    (out / 'VOTING_CHECK.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    return result
