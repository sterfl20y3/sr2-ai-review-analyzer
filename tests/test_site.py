import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class SiteTests(unittest.TestCase):
    def test_required_files(self):
        for filename in ("index.html", "app.js", "model.json"):
            self.assertTrue((ROOT / "site" / filename).is_file())

    def test_model_schema(self):
        model = json.loads((ROOT / "site/model.json").read_text(encoding="utf-8"))
        self.assertEqual(model["classes"], ["negative", "positive"])
        self.assertEqual(len(model["class_log_prior"]), 2)
        features = len(model["vocabulary"])
        self.assertGreater(features, 10)
        self.assertTrue(all(len(row) == features for row in model["feature_log_prob"]))

    def test_page_loads_model(self):
        app = (ROOT / "site/app.js").read_text(encoding="utf-8")
        self.assertIn('fetch("model.json")', app)
        self.assertIn('infer(text, model)', app)


if __name__ == "__main__":
    unittest.main()
