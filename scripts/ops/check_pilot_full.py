import json, glob
f = glob.glob('results/experiments/*pilot*.jsonl')[0]
for i, line in enumerate(open(f)):
    d = json.loads(line)
    print(f'=== EPISODE {i+1} ===')
    for k in sorted(d.keys()):
        print(f'  {k}: {d[k]}')
