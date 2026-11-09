"""Offline aggregation checks against the real completed engineering preflight."""
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("merge_shards", HERE / "merge_shards.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class MergeTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="test-bazt-merge-")
        self.root = Path(self.temp.name)
        self.source = HERE / "results_preflight/cpu_mock_proposed"
        if not self.source.exists():
            self.skipTest("Local engineering preflight is not present")
        for topology in ("small_world", "star_hub"):
            dest = self.root / "artifacts" / topology / "cpu_mock_proposed"
            block = dest / "mock_frontier"
            (block / "units").mkdir(parents=True)
            settings = json.loads((self.source / "mock_frontier/settings.json").read_text())
            settings["settings"]["mock.topologies"] = [topology]
            (block / "settings.json").write_text(json.dumps(settings))
            shutil.copytree(self.source / "mock_calibration", dest / "mock_calibration")
            for file in (self.source / "mock_frontier/units").glob("*.jsonl"):
                with file.open() as stream:
                    first = json.loads(stream.readline())
                if first["topology"] == topology:
                    shutil.copy2(file, block / "units" / file.name)
                    shutil.copy2(file.with_suffix(".done"), block / "units" / file.with_suffix(".done").name)

    def tearDown(self):
        self.temp.cleanup()

    def test_disjoint_complete_grid(self):
        report = mod.merge(HERE / "preflight_only.toml", self.root / "artifacts", self.root / "merged")
        self.assertEqual(report["merged_units"], 136)
        self.assertEqual(report["pair_unit_mock"], "seed")
        self.assertEqual(report["utility_endpoint"], "accuracy_clean")

    def test_missing_shard_refused(self):
        shutil.rmtree(self.root / "artifacts/star_hub")
        with self.assertRaisesRegex(ValueError, "Incomplete grid"):
            mod.merge(HERE / "preflight_only.toml", self.root / "artifacts", self.root / "merged")

    def test_duplicate_shard_refused(self):
        shutil.copytree(self.root / "artifacts/star_hub", self.root / "artifacts/duplicate")
        with self.assertRaisesRegex(ValueError, "Duplicate unit"):
            mod.merge(HERE / "preflight_only.toml", self.root / "artifacts", self.root / "merged")


if __name__ == "__main__":
    unittest.main()
