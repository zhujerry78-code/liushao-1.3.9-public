import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import pytest

from trial_instance import HOST, STATE_FILE, state_for, status, stop

ROOT = Path(__file__).resolve().parents[1]


def test_foreign_listener_survives_missing_or_stale_state(tmp_path):
    with socket.socket() as listener:
        listener.bind((HOST, 0))
        listener.listen()
        port = listener.getsockname()[1]
        assert stop(tmp_path, port) == 1
        (tmp_path / STATE_FILE).write_text(json.dumps(
            {"root": str(tmp_path / "other"), "control": port, "port": port, "key": "x" * 64}))
        assert stop(tmp_path, port) == 1
        with socket.create_connection((HOST, port), timeout=1):
            pass


@pytest.mark.skipif(sys.platform != "win32", reason="trial starts on Windows")
def test_two_directories_handle_the_same_port_without_cross_stop(tmp_path):
    with socket.socket() as probe:
        probe.bind((HOST, 0))
        port = probe.getsockname()[1]
    (tmp_path / "RELEASE_MANIFEST.json").write_bytes((ROOT / "RELEASE_MANIFEST.json").read_bytes())
    elsewhere = tmp_path / "second"
    elsewhere.mkdir()
    env = dict(os.environ, LIUSHAO_NONINTERACTIVE="1", PYTHONUTF8="1")
    args = [sys.executable, str(ROOT / "trial_instance.py"), "start", "--root", str(tmp_path),
            "--port", str(port)]
    proc = subprocess.Popen(args, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + 45
        while status(state_for(tmp_path)) != "ready" and time.monotonic() < deadline:
            if proc.poll() is not None:
                pytest.fail(proc.stdout.read().decode("utf-8", "replace"))
            time.sleep(.2)
        state = state_for(tmp_path)
        assert status(state) == "ready"
        assert status(dict(state, key="0" * 64), "stop") is None
        assert stop(elsewhere, port) == 1
        assert subprocess.run(args, env=env, timeout=15).returncode == 0
        assert status(state) == "ready"
        assert stop(tmp_path, port) == 0
        assert proc.wait(timeout=15) == 0
        assert state_for(tmp_path) is None
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=10)
