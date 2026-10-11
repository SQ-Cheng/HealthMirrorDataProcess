"""Guard experimental plots against PDF writers and test actual PNG exports."""

import ast
from pathlib import Path
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


class PNGOutputTests(unittest.TestCase):
    def test_experiment_sources_have_no_pdf_export(self):
        violations = []
        for directory in (ROOT / "study", ROOT / "preprocess"):
            for path in directory.rglob("*.py"):
                if set(path.relative_to(directory).parts) & {"outputs", "cache", "logs", "backbones", "__pycache__"}:
                    continue
                tree = ast.parse(path.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    is_pdf_import = isinstance(node, ast.ImportFrom) and node.module == "matplotlib.backends.backend_pdf"
                    is_pdf_name = isinstance(node, ast.Name) and node.id == "PdfPages"
                    is_pdf_formats = isinstance(node, ast.Tuple) and any(isinstance(value, ast.Constant) and value.value == "pdf" for value in node.elts)
                    is_pdf_save = (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "savefig"
                                   and any(isinstance(value, ast.Constant) and isinstance(value.value, str)
                                           and (value.value.lower().endswith(".pdf") or value.value.lower() == "pdf")
                                           for value in ast.walk(node)))
                    if is_pdf_import or is_pdf_name or is_pdf_formats or is_pdf_save:
                        violations.append(f"{path.relative_to(ROOT)}:{node.lineno}")
        self.assertEqual(violations, [])

    def test_shared_roi_and_dino_savers_are_png_only(self):
        from study.exp1_roi_color_regression.plots import save as roi_save
        from study.exp6_face_pair_dinov3_regression.plots import save as dino_save
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            for stem, save in (("roi", roi_save), ("dino", dino_save)):
                figure, axis = plt.subplots()
                axis.plot([0, 1], [0, 1])
                save(figure, root / stem)
            self.assertEqual(sorted(path.name for path in root.iterdir()), ["dino.png", "roi.png"])

    def test_surgery_pages_are_individual_pngs(self):
        from study.exp2_lab_longitudinal_statistics.surgery_analysis import _plot_phase_pngs
        summary = pd.DataFrame([{"cohort": cohort, "variable_id": "A001", "phase_order": phase,
                                 "median": 2 + phase, "q25": 1 + phase, "q75": 3 + phase,
                                 "unit": "mmol/L", "item_name_en": "Example analyte"}
                                for cohort in ("cabg", "all_surgery") for phase in (0, 5, 7)])
        with tempfile.TemporaryDirectory() as name:
            paths = _plot_phase_pngs(summary, ["A001"], name)
            self.assertEqual(paths, [str(Path(name) / "per_analyte_surgery/A001.png")])
            self.assertTrue(Path(paths[0]).is_file())
            self.assertEqual(list(Path(name).rglob("*.pdf")), [])


if __name__ == "__main__":
    unittest.main()
