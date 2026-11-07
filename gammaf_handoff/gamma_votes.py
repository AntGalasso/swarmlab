"""Reconstruct the two voting populations from final executed GAMMAF records."""
import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np


HERE = Path(__file__).resolve().parent
SOURCES = json.loads((HERE / 'gamma_sources.json').read_text())
MODELS = ('BlindGuard', 'CASPIAN', 'DOMINANT', 'PREM', 'TAM', 'XG-Guard')
CONTROL = 'no_defense_baseline'
HELD = ('MMLUPRO', 'CSQA', 'GSM8K')
GRAPHS = ('chain', 'random', 'star', 'tree')


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def original(key, root, fetch):
    source = SOURCES['files'][key]
    file = root / source['path']
    if not file.exists() and fetch:
        url = f"https://raw.githubusercontent.com/{SOURCES['repo']}/{SOURCES['commit']}/{source['path']}"
        with urlopen(url, timeout=60) as response:
            raw = response.read()
        file.parent.mkdir(parents=True, exist_ok=True)
        temp = file.with_suffix('.download')
        verify_blob(raw, source)
        temp.write_bytes(raw)
        temp.replace(file)
    raw = file.read_bytes()
    verify_blob(raw, source)
    return json.loads(raw)


def verify_blob(raw, source):
    header = f'blob {len(raw)}\0'.encode()
    require(len(raw) == source['bytes'] and hashlib.sha1(header + raw).hexdigest() == source['blob'],
            'Unexpected public Git blob ' + source['path'])


def plurality(answers):
    # Counter preserves first occurrence on ties, matching both upstream voters.
    return collections.Counter(answers).most_common(1)[0][0]


def correct(value, gold, ds):
    return value == gold if ds == 'GSM8K' else value.upper() == gold.upper()


def alternatives(answers, gold, dataset):
    existing = set(answers) | {gold}
    if dataset == 'GSM8K':
        fresh = 0
        while str(fresh) in existing:
            fresh += 1
        return sorted(existing | {str(fresh)})
    options = 'ABCDEFGHIJ' if dataset == 'MMLUPRO' else 'ABCD'
    fresh = next((x for x in options if x not in existing and x != gold), None)
    return sorted(existing | ({fresh} if fresh is not None else set()))


def extract(records, ds, method, official):
    require(records['model'] == method and records['dataset_tag'] == ds, 'Source identity')
    debates = collections.defaultdict(list)
    for r in records['items']:
        require(r['included'], 'Unexpected excluded record')
        debates[r['loader_index'], r['topology']].append(r)
    result = []
    for (item, graph), rounds in sorted(debates.items()):
        rounds.sort(key=lambda r:r['round'])
        require([r['round'] for r in rounds] == [0, 1, 2], 'Incomplete three-round grid')
        real = [r for r in rounds if not r['padded']]
        require(real and real == rounds[:len(real)], 'Nonterminal imputation')
        r = real[-1]
        a, f, gold = r['agent_answers'], r['flags'], r['ground_truth']
        require(len(a) == len(f) == 10 and set(f) <= {0,1} and 0 in f, 'Unknown electorate')
        require(all(q['is_correct'] == r['is_correct'] and q['final_answer'] == r['final_answer']
                    and q['ground_truth'] == gold for q in rounds), 'Inconsistent final record')
        delivered = plurality([x for x, flag in zip(a, f) if not flag])
        whole = plurality(a)
        label = correct(whole, gold, ds)
        actual = correct(delivered, gold, ds)
        require(label == r['is_correct'] and delivered == r['final_answer'], 'Upstream replay differs')
        sensitive = False
        for replacement in alternatives(a, gold, ds):
            votes = [replacement if flag else x for x, flag in zip(a, f)]
            require(plurality([x for x, flag in zip(votes,f) if not flag]) == delivered,
                    'Excluded-vote perturbation changed delivered answer')
            sensitive |= correct(plurality(votes), gold, ds) != label
        result.append(dict(dataset=ds, method=method, item=item, graph=graph, executed_rounds=len(real),
                           gold=gold, votes=a, flagged=f, all_answer=whole, delivered=delivered,
                           published=int(label), actual=int(actual), disagreement=int(label != actual),
                           false_credit=int(label and not actual), missed_credit=int(actual and not label),
                           perturbable=int(sensitive), alerts=sum(f)))
    for graph in GRAPHS:
        sub = [r for r in result if r['graph'] == graph]
        pub = next(r for r in official['runs'][method]['datasets'][ds]['results'] if r['topology'] == graph)
        require(len(sub) == pub['total_questions'] == 100 and len({r['item'] for r in sub}) == 100,
                'Wrong graph denominator')
        require(sum(r['published'] for r in sub) == pub['correct_answers'], 'Public accuracy count differs')
    require(len(result) == 400 and len({r['item'] for r in result}) == 100, 'Broken repeated graph grid')
    if method == CONTROL:
        require(all(r['disagreement'] == r['perturbable'] == r['alerts'] == 0 for r in result),
                'Disabled control changes votes')
    return result


METRICS = ('published','actual','disagreement','false_credit','missed_credit','perturbable','alerts')


def summarize(rows):
    return {'debates':len(rows), **{k:sum(r[k] for r in rows) for k in METRICS}}


def paired_items(rows, method):
    sub = [r for r in rows if r['method'] == method]
    groups = collections.defaultdict(dict)
    for r in sub:
        require(r['graph'] not in groups[r['item']], 'Duplicate graph')
        groups[r['item']][r['graph']] = r
    require(all(set(graphs) == set(GRAPHS) for graphs in groups.values()), 'Unpaired topology')
    return groups


def select(rows):
    counts = {m:summarize([r for r in rows if r['method'] == m]) for m in MODELS}
    picked = {}
    for k in ('published', 'actual'):
        best = max(v[k] for v in counts.values())
        ties = sorted(m for m in MODELS if counts[m][k] == best)
        picked[k] = {'choice':ties[0], 'ties':ties, 'correct_of_400':best}
    return {'development':'MMLU', 'counts':counts, 'selected':picked}


def validate_selection(data, frozen):
    official = frozen['selected']['published']['choice']
    delivered = frozen['selected']['actual']['choice']
    rng = np.random.default_rng(202611071017)
    boot = np.zeros(20000)
    values = {}
    for ds in HELD:
        a, b = (paired_items(data[ds], m) for m in (official, delivered))
        require(a.keys() == b.keys(), 'Unmatched detector question IDs')
        diffs = np.array([np.mean([b[i][g]['actual'] - a[i][g]['actual'] for g in GRAPHS])
                          for i in sorted(a)]) * 100
        draw = rng.choice(diffs,(20000,len(diffs)),replace=True).mean(1)
        boot += draw / len(HELD)
        values[ds] = {'gain_pp':float(diffs.mean()),'ci95_pp':np.quantile(draw,[.025,.975]).tolist()}
    return {'release_selection':official, 'delivered_selection':delivered,'tasks':values,
            'mean_gain_pp':float(np.mean([v['gain_pp'] for v in values.values()])),
            'cluster_ci95_pp':np.quantile(boot,[.025,.975]).tolist()}


def save(path, obj):
    path.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['develop','validate'])
    p.add_argument('--raw',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--fetch',action='store_true',help='Fetch missing public, pinned JSON inputs')
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    official=original('summary',args.raw,args.fetch)
    selection_path=args.out/'MMLU_SELECTION.json'
    if args.phase=='develop':
        rows=[r for m in (*MODELS,CONTROL) for r in extract(original('MMLU/'+m,args.raw,args.fetch),'MMLU',m,official)]
        choices=select(rows)
        if selection_path.exists():
            require(json.loads(selection_path.read_text()) == choices,'Refusing to change frozen selection')
        else:
            save(selection_path,choices)
        print(json.dumps(choices['selected']))
        return
    frozen=json.loads(selection_path.read_text())
    data={}
    for ds in ('MMLU',*HELD):
        data[ds]=[r for m in (*MODELS,CONTROL) for r in
                  extract(original(ds+'/'+m,args.raw,args.fetch),ds,m,official)]
        src=paired_items(data[ds],CONTROL)
        for m in MODELS:
            other=paired_items(data[ds],m)
            require(src.keys()==other.keys() and all(src[i][g]['gold']==other[i][g]['gold']
                    for i in src for g in GRAPHS),'Gold/IDs not shared')
    require(select(data['MMLU'])==frozen,'Development selection drift')
    stats={ds:{m:summarize([r for r in rows if r['method']==m]) for m in (*MODELS,CONTROL)}
           for ds,rows in data.items()}
    by_graph={ds:{m:{g:summarize([r for r in rows if r['method']==m and r['graph']==g])
                     for g in GRAPHS} for m in (*MODELS,CONTROL)} for ds,rows in data.items()}
    cases=sorted([r for rows in data.values() for r in rows if r['disagreement']],
                 key=lambda r:(r['dataset'],r['method'],r['item'],r['graph']))
    raw=[r for rows in data.values() for r in rows]
    packed=''.join(json.dumps(r,separators=(',',':'))+'\n' for r in raw).encode()
    (args.out/'gamma_debates.jsonl.gz').write_bytes(gzip.compress(packed,mtime=0))
    output={'selection_outcome':validate_selection(data,frozen),'by_task_method':stats,
            'by_graph':by_graph,'first_observed_discordance':cases[0] if cases else None,
            'debates':len(raw),'source_summary_cells_checked':len(data)*7*4,
            'counts_are_method_debates_not_independent_questions':True}
    save(args.out/'GAMMAF_RESULTS.json',output)
    print(json.dumps(output['selection_outcome']))


if __name__=='__main__':
    main()
