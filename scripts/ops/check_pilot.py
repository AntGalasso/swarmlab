import json, glob
files = glob.glob('results/experiments/*pilot*.jsonl')
if not files:
    print('Nessun file pilot trovato in results/experiments/')
else:
    print(f'--- PILOT REPORT ({files[0]}) ---')
    for line in open(files[0]):
        try:
            d = json.loads(line)
            if 'arm' in d:
                asr = d.get('asr_final', 0) * 100
                acc = d.get('accuracy', 0) * 100
                spent = d.get('intervention_spent', 0)
                print(f"{d.get('scenario', ''):<10} | {d.get('arm', ''):<20} | ASR: {asr:>5.1f}% | Acc: {acc:>5.1f}% | Spent: {spent}")
        except:
            pass
