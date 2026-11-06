#!/usr/bin/env python3
"""Run every test group in a fresh interpreter to bound ML import/cache memory."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results/integration/grouped_tests'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    groups = [Path('tests')]
    groups += sorted(p.relative_to(ROOT) for p in (ROOT/'bazt_lab').rglob('tests') if p.is_dir())
    groups += sorted(p.relative_to(ROOT) for p in (ROOT/'scripts/tests').glob('test_*.py'))
    env = {**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
           'TORCHINDUCTOR_COMPILE_THREADS':'1','MAX_JOBS':'1','HF_HUB_OFFLINE':'1',
           'TRANSFORMERS_OFFLINE':'1','WANDB_MODE':'offline'}
    report = []
    for group in groups:
        name = str(group).replace('/','_').replace('.py','')
        command = [sys.executable,'-m','pytest','-q',str(group),'--tb=short','--junitxml='+str(OUT/(name+'.xml'))]
        print('Checking '+str(group),flush=True)
        with (OUT/(name+'.log')).open('w') as f:
            done = subprocess.run(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=False)
        report.append({'group':str(group),'exit_code':done.returncode})
        (OUT/'status.json').write_text(json.dumps(report,indent=2)+'\n')
        print((OUT/(name+'.log')).read_text()[-900:],flush=True)
        if done.returncode:
            return done.returncode
    return 0


if __name__=='__main__':
    raise SystemExit(main())
