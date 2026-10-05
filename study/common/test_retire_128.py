"""Deletion candidates must respect native outputs and protected dependencies."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import retire_128_artifacts as cleanup


class RetirementTests(unittest.TestCase):
    def test_only_superseded_owners_and_no_native_or_shared_clinical_records(self):
        with tempfile.TemporaryDirectory() as directory:
            study = Path(directory) / "study"
            reg, classifier, delta, spectral = [study / name for name in ("reg", "class", "delta", "spec")]
            files = [reg / "outputs/20frame/runs/model.pt", reg / "outputs/20frame/task_records/hb.csv",
                     reg / "outputs/20frame_face224/runs/model.pt", reg / "outputs/5fold/splits/hb_fold0.csv",
                     delta / "outputs/runs/hb/model.pt", delta / "outputs/runs/hb/pair_predictions.csv",
                     delta / "outputs/face224/runs/model.pt", classifier / "outputs/runs/model.pt",
                     classifier / "outputs/face224/runs/model.pt", spectral / "outputs/runs/model.pt",
                     study / "protected/outputs/runs/model.pt"]
            for path in files:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture")
            clinical = {reg / "outputs/20frame": {"task_records"},
                        reg / "outputs/5fold": {"splits"},
                        delta / "outputs": {"task_records"}}
            with patch.multiple(cleanup, STUDY=study, REG=reg, CLASS=classifier,
                                DELTA=delta, SPECTRAL=spectral, CLINICAL=clinical, STATE=study / "state"):
                selected = cleanup.candidates(True)
            deleted = [p for p in files if p in selected or any(parent in selected for parent in p.parents)]
            self.assertEqual(set(deleted), {files[0], files[4], files[7]})


if __name__ == "__main__":
    unittest.main()
