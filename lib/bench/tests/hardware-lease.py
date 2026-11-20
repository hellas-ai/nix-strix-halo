"""Exercise the generated benchmark wrapper without hardware or host mounts."""

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def ready(path, process):
    deadline = time.monotonic() + 3
    while not path.exists():
        assert process.poll() is None, process.communicate()
        assert time.monotonic() < deadline, path
        time.sleep(0.01)


with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    lock = root / "host.lock"
    script = sys.argv[1]
    processes = []

    def start(name):
        output = root / name
        process = subprocess.Popen(
            ["bash", script],
            env={**os.environ, "out": str(output), "LEASE_FIXTURE_LOCK": str(lock)},
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        processes.append(process)
        return output, process

    try:
        output, missing = start("missing")
        missing.communicate(b"x", timeout=3)
        assert missing.returncode != 0
        assert not lock.exists() and not (output / "started").exists()

        lock.write_bytes(b"preserve this inode and content\n")
        lock.chmod(0o444)
        before = (lock.stat().st_ino, lock.read_bytes())
        marker = root / "serving-started"
        holder = subprocess.Popen(
            [
                "flock",
                "--nonblock",
                str(lock),
                sys.executable,
                "-c",
                "import pathlib,sys; pathlib.Path(sys.argv[1]).touch(); sys.stdin.buffer.read(1)",
                str(marker),
            ],
            stdin=subprocess.PIPE,
            start_new_session=True,
        )
        processes.append(holder)
        ready(marker, holder)
        output, busy = start("busy")
        began = time.monotonic()
        busy.communicate(b"x", timeout=3)
        assert busy.returncode == 75
        assert time.monotonic() - began >= 0.15
        assert not (output / "started").exists()
        holder.communicate(b"x", timeout=3)
        assert holder.returncode == 0

        output, active = start("active")
        ready(output / "started", active)
        assert (
            subprocess.run(
                ["flock", "--nonblock", str(lock), "true"], check=False
            ).returncode
            == 1
        )
        contender_output, contender = start("contender")
        contender.communicate(b"x", timeout=3)
        assert contender.returncode == 75
        assert not (contender_output / "started").exists()
        active.communicate(b"x", timeout=3)
        assert active.returncode == 0
        assert (
            subprocess.run(
                ["flock", "--nonblock", str(lock), "true"], check=False
            ).returncode
            == 0
        )
        assert (lock.stat().st_ino, lock.read_bytes()) == before
        print(
            json.dumps(
                {
                    "missing_fails_closed": True,
                    "busy_status": 75,
                    "held_through_child": True,
                    "released_after_child": True,
                    "inode_content_unchanged": True,
                }
            )
        )
    finally:
        for process in processes:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=3)
