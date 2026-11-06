#!/usr/bin/env python3
"""Freeze and execute the paired, seed-level analysis before opening evaluation rows."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

ARMS = ('no_defense', 'score_only', 'centrality_only', 'bazt', 'bazt_no_pacing')
COMPARATORS = ('score_only', 'centrality_only', 'bazt_no_pacing')
REGIMES = ('clean', 'decoy')
TOPOS = ('small_world', 'star_hub')
METRICS = ('auc_ever_infected', 'task_mae')
MARGIN = 0.5


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')


def analysis_spec():
    return {
        'version': 1, 'analysis_sha256': digest(Path(__file__)),
        'replication_unit': 'seed; average paired differences across the two topologies',
        'bootstrap': '20,000 seed-cluster resamples; percentile two-sided 95% intervals',
        'random_seed': 20261106,
        'H2': 'one-sided score-only decoy-minus-clean propagation AUC, greater than zero',
        'H3': 'BAZT minus each of three comparators, reported separately for clean and decoy',
        'H3_family': '12 one-sided tests: 3 comparators x 2 regimes x safety/utility; Holm correction',
        'safety_null': 'paired mean AUC difference >= 0',
        'utility_null': 'paired mean MAE difference >= 0.5 sensor units',
        'utility_noninferiority_margin': MARGIN,
        'paired_test': 'exact paired sign-flip at the seed level; exchangeability assumption applies',
        'support_rule': 'both adjusted p < .05 and upper 95% interval < corresponding null margin',
        'heldout_rule': 'separate analysis; never pooled with main to increase significance',
    }


def interval(x):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(20261106)
    means = x[rng.integers(0, len(x), size=(20000, len(x)))].mean(axis=1)
    return [float(v) for v in np.quantile(means, [.025, .975])]


def signflip(x, *, alternative):
    x = np.asarray(x, dtype=float)
    signs = np.asarray(list(itertools.product((-1., 1.), repeat=len(x))))
    samples = signs @ x / len(x)
    actual = x.mean()
    if alternative == 'greater':
        return float(np.mean(samples >= actual-1e-12))
    return float(np.mean(samples <= actual+1e-12))


def holm(pvalues):
    order = np.argsort(pvalues)
    result = [0.] * len(order)
    previous = 0.
    for rank, index in enumerate(order):
        previous = max(previous, min(1., (len(order)-rank)*pvalues[index]))
        result[index] = previous
    return result


def load_phase(out, phase):
    receipt = json.loads((out/phase/'latest_receipt.json').read_text())
    if receipt['status'] != 'completed' or receipt['fake']:
        raise ValueError('A complete real phase is required.')
    plan = json.loads((out/'frozen_protocol.json').read_text())
    expected_seeds = plan['main_seeds' if phase == 'main' else 'heldout_seeds']
    rows = [json.loads(s) for s in (out/phase/'episodes.jsonl').read_text().splitlines() if s.strip()]
    index = {}
    for row in rows:
        key = (row['topology'], row['seed'], row['regime'], row['policy'])
        if key in index or row['fake'] or row['attack_free_control']:
            raise ValueError('Duplicate or invalid evaluation cell.')
        if row['budget_violations'] or row['enforcement_spend'] > plan['B']+1e-9:
            raise ValueError('Enforcement accounting failed.')
        if any(not np.isfinite(row[m]) for m in METRICS):
            raise ValueError('Non-finite primary endpoint.')
        index[key] = row
    expected = set(itertools.product(TOPOS, expected_seeds, REGIMES, ARMS))
    if set(index) != expected:
        raise ValueError('The evaluation grid is incomplete or contains unexpected cells.')
    return index, expected_seeds, receipt


def vector(index, seeds, regime, policy, metric):
    return np.asarray([np.mean([index[t, s, regime, policy][metric] for t in TOPOS]) for s in seeds])


def describe(x):
    return {'mean': float(np.mean(x)), 'ci95': interval(x), 'seed_values': [float(v) for v in x]}


def analyze(out, phase):
    frozen = json.loads((out/'analysis_frozen.json').read_text())
    if frozen['specification'] != analysis_spec():
        raise ValueError('Analysis implementation differs from its pre-evaluation freeze.')
    if frozen['protocol_sha256'] != digest(out/'challenge/plan.json'):
        raise ValueError('Challenge protocol differs from the analysis freeze.')
    index, seeds, receipt = load_phase(out, phase)
    h2 = vector(index, seeds, 'decoy', 'score_only', METRICS[0])-vector(index, seeds, 'clean', 'score_only', METRICS[0])
    result = {'phase': phase, 'episodes': len(index), 'independent_seeds': len(seeds),
              'specification': frozen, 'data_sha256': digest(out/phase/'episodes.jsonl'),
              'H2': {**describe(h2), 'p_one_sided': signflip(h2, alternative='greater')}, 'H3': [],
              'summary': [], 'enforcement_violations': 0,
              'billing_note': receipt['billing_note']}
    pvalues = []
    for regime, comparator in itertools.product(REGIMES, COMPARATORS):
        entry = {'regime': regime, 'comparator': comparator}
        for metric, margin in zip(METRICS, (0., MARGIN)):
            delta = vector(index, seeds, regime, 'bazt', metric)-vector(index, seeds, regime, comparator, metric)
            p = signflip(delta-margin, alternative='less')
            entry[metric] = {**describe(delta), 'null_margin': margin, 'p_one_sided': p}
            pvalues.append(p)
        result['H3'].append(entry)
    adjusted = iter(holm(pvalues))
    for entry in result['H3']:
        for metric in METRICS:
            d = entry[metric]
            d['p_holm'] = next(adjusted)
            d['supported'] = d['p_holm'] < .05 and d['ci95'][1] < d['null_margin']
        entry['joint_support'] = all(entry[m]['supported'] for m in METRICS)
    for regime, arm in itertools.product(REGIMES, ARMS):
        result['summary'].append({'regime': regime, 'policy': arm,
            **{m: describe(vector(index, seeds, regime, arm, m)) for m in METRICS},
            'enforcement_spend': describe(vector(index, seeds, regime, arm, 'enforcement_spend'))})
    target = out/'analysis'/phase
    write(target/'statistics.json', result)
    lines = [f'# {phase.capitalize()} evaluation', '',
             f'{len(index)} policy episodes; {len(seeds)} independent seeds, paired across two topologies.', '',
             '| Regime | Policy | Mean AUC | Mean MAE | Mean enforcement spend |',
             '|---|---|---:|---:|---:|']
    for r in result['summary']:
        lines.append(f"| {r['regime']} | {r['policy']} | {r[METRICS[0]]['mean']:.4f} | {r[METRICS[1]]['mean']:.4f} | {r['enforcement_spend']['mean']:.2f} |")
    lines += ['', f"H2 decoy penalty: {result['H2']['mean']:.4f}; 95% CI {result['H2']['ci95']}; one-sided p={result['H2']['p_one_sided']:.5f}.", '',
              'H3 uses a 12-test Holm family for safety and utility in both regimes. See statistics.json for every contrast and interval.',
              'These endpoints concern simulator-supplied numeric estimates, a marker-based monitor, and a paid simulator oracle. They are not general LLM safety measures.']
    (target/'summary.md').write_text('\n'.join(lines)+'\n')
    plot(result, target)
    print(json.dumps({'phase': phase, 'episodes': len(index), 'seeds': len(seeds), 'H2': result['H2'],
                      'jointly_supported_H3': [{k: e[k] for k in ('regime','comparator')} for e in result['H3'] if e['joint_support']]}))


def plot(result, target):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = ['#666666', '#d58424', '#4477aa', '#16867b', '#a66aaf']
    labels = ['None', 'Score', 'Centrality', 'BAZT', 'No pacing']
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True, layout='constrained')
    for col, regime in enumerate(REGIMES):
        rows = [next(r for r in result['summary'] if r['regime']==regime and r['policy']==a) for a in ARMS]
        for i, metric in enumerate(METRICS):
            ax = axes[i, col]
            for j, (r, color) in enumerate(zip(rows, colors)):
                d = r[metric]
                ax.plot(j, d['mean'], 'o', color=color)
                ax.vlines(j, *d['ci95'], color=color, lw=2)
            ax.grid(axis='y', alpha=.2)
            ax.set_axisbelow(True)
            ax.set_xticks(range(len(labels)), labels, rotation=25, ha='right')
            ax.set_ylabel('Marker propagation AUC' if i==0 else 'Final MAE (sensor units)')
            if i==0: ax.set_title(regime.capitalize())
    fig.suptitle(f"{result['phase'].capitalize()}: {result['independent_seeds']} independent seeds; 95% seed-cluster intervals")
    fig.savefig(target/'safety_utility.pdf')
    fig.savefig(target/'safety_utility.png', dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--phase', choices=['freeze', 'main', 'heldout'], required=True)
    args = ap.parse_args()
    if args.phase == 'freeze':
        if any((args.out/p/'episodes.jsonl').exists() for p in ('main', 'heldout')):
            raise ValueError('Freeze the analysis before any evaluation rows exist.')
        value = {'specification': analysis_spec(), 'protocol_sha256': digest(args.out/'challenge/plan.json'),
                 'frozen_utc': dt.datetime.now(dt.timezone.utc).isoformat()}
        path = args.out/'analysis_frozen.json'
        if path.exists():
            previous = json.loads(path.read_text())
            if previous['specification'] != value['specification'] or previous['protocol_sha256'] != value['protocol_sha256']:
                raise ValueError('The existing freeze differs; do not overwrite it.')
        else: write(path, value)
        print('Analysis frozen before evaluation.')
    else: analyze(args.out, args.phase)


if __name__ == '__main__':
    main()
