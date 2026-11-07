"""Independent SQL count of both electorates, with a strict-format, no-tie sensitivity check."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import sqlite3


SQL = """
WITH votes AS (
  SELECT d.id, pop.which, v.answer, COUNT(*) AS supporters, MIN(v.agent) AS first_agent
  FROM debates d JOIN positions v ON d.id=v.id
  CROSS JOIN (SELECT 0 AS which UNION ALL SELECT 1) pop
  WHERE pop.which=0 OR v.flag=0
  GROUP BY d.id,pop.which,v.answer
), placed AS (
  SELECT *, ROW_NUMBER() OVER
      (PARTITION BY id,which ORDER BY supporters DESC, first_agent) AS position,
      LEAD(supporters) OVER
      (PARTITION BY id,which ORDER BY supporters DESC, first_agent) AS next_count
  FROM votes
)
SELECT d.ds,d.method,d.item,d.topology,d.gold,d.answer,d.label,d.canonical,
       a.answer,b.answer,
       (a.next_count IS NULL OR a.supporters>a.next_count) AND
       (b.next_count IS NULL OR b.supporters>b.next_count) AS untied
FROM debates d JOIN placed a ON a.id=d.id AND a.which=0 AND a.position=1
JOIN placed b ON b.id=d.id AND b.which=1 AND b.position=1
ORDER BY d.ds,d.method,d.item,d.topology
"""


def verify(root,primary_export,out):
    pins=json.loads((Path(__file__).parent/'gamma_sources.json').read_text())
    con=sqlite3.connect(':memory:')
    con.executescript("""
        CREATE TABLE debates(id INTEGER PRIMARY KEY, ds TEXT, method TEXT, item INTEGER,
             topology TEXT, gold TEXT, answer TEXT, label INTEGER, canonical INTEGER);
        CREATE TABLE positions(id INTEGER, agent INTEGER, flag INTEGER, answer TEXT,
             PRIMARY KEY(id,agent));
    """)
    count=0
    for key,pin in sorted(pins['files'].items()):
        if key=='summary':
            continue
        ds,method=key.split('/')
        blob=(root/pin['path']).read_bytes()
        assert hashlib.sha1(f'blob {len(blob)}\0'.encode()+blob).hexdigest()==pin['blob']
        data=json.loads(blob)
        last={}
        for r in data['items']:
            if not r['included'] or r['padded']:
                continue
            unit=(r['loader_index'],r['topology'])
            if unit not in last or r['round']>last[unit]['round']:
                last[unit]=r
        assert len(last)==400
        canonical={'GSM8K':r'(0|-?[1-9][0-9]*)','MMLUPRO':r'[A-J]'}.get(ds,r'[A-D]')
        for (item,graph),r in last.items():
            count+=1
            good=all(re.fullmatch(canonical,s) for s in [r['ground_truth'],*r['agent_answers']])
            con.execute('INSERT INTO debates VALUES(?,?,?,?,?,?,?,?,?)',
                 (count,ds,method,item,graph,r['ground_truth'],r['final_answer'],r['is_correct'],good))
            con.executemany('INSERT INTO positions VALUES(?,?,?,?)',
                [(count,a,f,s) for a,(f,s) in enumerate(zip(r['flags'],r['agent_answers'],strict=True))])
    with gzip.open(primary_export,'rt') as f:
        rows=[json.loads(line) for line in f]
    reference={(r['dataset'],r['method'],r['item'],r['graph']):r for r in rows}
    assert len(rows)==len(reference)==count==11200
    cells={}
    first=None
    observed=0
    for ds,m,item,g,gold,answer,label,canonical,full,delivered,untied in con.execute(SQL):
        ref=reference[ds,m,item,g]
        allow=lambda x: x==gold if ds=='GSM8K' else x.upper()==gold.upper()
        full_ok,delivered_ok=allow(full),allow(delivered)
        assert full_ok==label==ref['published'] and delivered==answer==ref['delivered']
        assert delivered_ok==ref['actual'] and full==ref['all_answer']
        observed+=1
        cell=cells.setdefault(ds,{}).setdefault(m,{'all':0,'canonical_untied':0,
                      'discordant':0,'strict_discordant':0,'strict_false_credit':0,'strict_missed_credit':0})
        cell['all']+=1
        cell['discordant']+=full_ok!=delivered_ok
        if canonical and untied:
            cell['canonical_untied']+=1
            if full_ok!=delivered_ok:
                cell['strict_discordant']+=1
                cell['strict_false_credit']+=int(full_ok)
                cell['strict_missed_credit']+=int(delivered_ok)
                if first is None:
                    first=ref
    assert observed==count
    result={'sql_final_decisions_reconciled':observed,'by_task_method':cells,
            'first_canonical_untied_discordance':first}
    out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print('SQL reconstruction matches',observed,'debates')
    print('Strict subgroup', {k:sum(c[k] for s in cells.values() for m,c in s.items()
          if m!='no_defense_baseline') for k in ('canonical_untied','strict_discordant',
                                                'strict_false_credit','strict_missed_credit')})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',type=Path,required=True)
    p.add_argument('--debates',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    verify(a.raw,a.debates,a.out)
