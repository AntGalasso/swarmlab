"""Offline tests: no network, API client, real calibration or evaluation."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("antonio_run",HERE/"run_n100.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class HandoffTest(unittest.TestCase):
    def test_exact_frozen_cpu_scope(self):
        s,p=m.validate()
        self.assertEqual(p["expected_evaluation_units"],6480)
        cs=m.commands(Path('/runtime'),Path('/settings'),Path('/own-results'),8)
        self.assertEqual([c[-1] for c in cs],['mock_calibration','mock_epidemiology'])
        self.assertFalse(any('run-all' in c or 'experiments.py' in c or '--allow-live-llm' in c for c in cs))

    def test_isolated_worktree_and_dirty_source_refusal(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);repo=root/'repo';repo.mkdir()
            def git(*args):return subprocess.run(['git','-C',str(repo),*args],check=True,capture_output=True,text=True)
            git('init');git('config','user.name','Offline Test');git('config','user.email','offline-test@example.invalid')
            (repo/'source.py').write_text('PINNED = 1\n');git('add','source.py');git('commit','-m','offline fixture')
            pin=git('rev-parse','HEAD').stdout.strip();runtime=root/'runtime'
            m.ensure_runtime(repo,runtime,pin)
            self.assertTrue((runtime/'source.py').exists())
            (runtime/'source.py').write_text('PINNED = 2\n')
            with self.assertRaisesRegex(ValueError,'local changes'):m.ensure_runtime(repo,runtime,pin)
            self.assertEqual((repo/'source.py').read_text(),'PINNED = 1\n')

    def test_failed_calibration_blocks_evaluation(self):
        _,p=m.validate();cells=[]
        for a in p['evaluation_arms']:
            for t in p['topologies']:
                for n in p['agents']:cells.append({'arm':a,'topology':t,'agents':n,'diagnostics':{'rate':.02,'within_tolerance':True}})
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'thresholds.json';path.write_text(json.dumps({'cells':cells}))
            self.assertEqual(m.calibration_gate(path,p)['required_cells'],18)
            cells[0]['diagnostics']['rate']=.023;path.write_text(json.dumps({'cells':cells}))
            with self.assertRaisesRegex(ValueError,'No admissible calibration'):m.calibration_gate(path,p)


if __name__=='__main__':unittest.main()
