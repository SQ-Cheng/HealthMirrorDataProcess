"""CPU-only checks for the predecessor completion/lock policy."""

import fcntl
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import run_after_current


class PredecessorTests(unittest.TestCase):
    def test_incomplete_stopped_queue_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            (directory / ".queue.lock").touch()
            with patch.object(run_after_current, "PREDECESSOR", directory):
                with self.assertRaisesRegex(RuntimeError, "stopped before completion"):
                    run_after_current.predecessor_complete()

    def test_completed_queue_is_accepted_only_after_lock_release(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            (directory / "COMPLETE").write_text("ok\n")
            with (directory / ".queue.lock").open("w") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                with patch.object(run_after_current, "PREDECESSOR", directory):
                    self.assertFalse(run_after_current.predecessor_complete())
                    fcntl.flock(lock, fcntl.LOCK_UN)
                    self.assertTrue(run_after_current.predecessor_complete())


if __name__ == "__main__":
    unittest.main()
