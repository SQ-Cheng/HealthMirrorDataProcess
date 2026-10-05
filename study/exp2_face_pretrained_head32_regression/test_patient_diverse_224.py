"""Check the additional monitor without starting or changing GPU training."""

import fcntl
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import run_patient_diverse_224 as runner


class MonitorTests(unittest.TestCase):
    def test_active_queue_waits_even_if_an_old_completion_marker_exists(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            (state / "COMPLETE").write_text("old marker")
            with patch.object(runner, "QUEUE_STATE", state):
                with (state / ".queue.lock").open("w") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    self.assertFalse(runner.queue_finished())
                self.assertTrue(runner.queue_finished())

    def test_stopped_incomplete_queue_does_not_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            (state / ".queue.lock").touch()
            with patch.object(runner, "QUEUE_STATE", state):
                with self.assertRaisesRegex(RuntimeError, "before completion"):
                    runner.queue_finished()


if __name__ == "__main__":
    unittest.main()
