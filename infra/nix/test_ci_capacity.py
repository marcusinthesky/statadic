"""Runner telemetry must preserve command results and useful failure output."""

import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import unittest


SCRIPT = Path(__file__).with_name("ci-capacity.sh").resolve()


class CapacityTests(unittest.TestCase):
    def test_termination_stops_command_and_its_child(self):
        with tempfile.TemporaryDirectory() as root:
            pidfile = Path(root) / "pids"
            process = subprocess.Popen(
                [
                    "bash",
                    str(SCRIPT),
                    "run",
                    "probe",
                    "bash",
                    "-c",
                    'sleep 30 & echo "$$ $!" > "$1"; wait',
                    "bash",
                    str(pidfile),
                ],
                cwd=root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                deadline = time.monotonic() + 5
                while not pidfile.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(pidfile.exists(), "command did not start")
                pids = [int(value) for value in pidfile.read_text().split()]
                self.assertEqual(len(pids), 2)
                process.send_signal(signal.SIGTERM)
                self.assertEqual(process.wait(timeout=5), 143)
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    living = []
                    for pid in pids:
                        try:
                            state = Path(f"/proc/{pid}/stat").read_text().split()[2]
                        except FileNotFoundError:
                            continue
                        if state != "Z":
                            living.append(pid)
                    if not living:
                        break
                    time.sleep(0.01)
                self.assertEqual(living, [], "command descendants survived termination")
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                for pid in locals().get("living", []):
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    def test_command_status_and_both_streams_survive(self):
        for status in (0, 7, 143):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as root:
                result = subprocess.run(
                    [
                        "bash",
                        str(SCRIPT),
                        "run",
                        "probe",
                        "bash",
                        "-c",
                        f"echo stdout; echo stderr >&2; exit {status}",
                    ],
                    cwd=root,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                self.assertEqual(result.returncode, status, result.stderr)
                self.assertEqual(
                    (Path(root) / "build/ci/probe.log").read_text(),
                    "stdout\nstderr\n",
                )
                self.assertIn("stdout", result.stdout)
                self.assertIn("stderr", result.stdout)

    def test_memory_is_streamed_before_a_long_command_finishes(self):
        with tempfile.TemporaryDirectory() as root:
            result = subprocess.run(
                ["bash", str(SCRIPT), "run", "probe", "sleep", "1"],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Runner resources: probe", result.stdout)
            self.assertIn("Mem:", result.stdout)
            self.assertIn(
                "Mem:", (Path(root) / "build/ci/memory-probe.log").read_text()
            )


if __name__ == "__main__":
    unittest.main()
