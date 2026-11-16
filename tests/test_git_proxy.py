# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The GitHub mirror used by the build-time third-party clones."""

import ast
import http.server
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

import pytest

from build_utils.git_proxy import (
    DEFAULT_PROXY,
    GITHUB_BASE,
    PROXY_ENV,
    candidate_commands,
    proxy_config_args,
    resolve_proxy,
)

ROOT = Path(__file__).resolve().parents[1]


def test_unset_uses_the_default_mirror(monkeypatch):
    monkeypatch.delenv(PROXY_ENV, raising=False)
    assert resolve_proxy() == DEFAULT_PROXY == "https://gh-proxy.com/"


@pytest.mark.parametrize("value", ["1", "true", "ON", "yes", "default"])
def test_enabling_values_select_the_default_mirror(monkeypatch, value):
    monkeypatch.setenv(PROXY_ENV, value)
    assert resolve_proxy() == DEFAULT_PROXY


@pytest.mark.parametrize("value", ["0", "false", "OFF", "no", "none", "disabled", ""])
def test_disabling_values_clone_directly(monkeypatch, value):
    monkeypatch.setenv(PROXY_ENV, value)
    assert resolve_proxy() == ""
    assert proxy_config_args() == []
    assert candidate_commands(["git", "clone", "url"]) == [["git", "clone", "url"]]


def test_custom_mirror_is_normalized(monkeypatch):
    monkeypatch.setenv(PROXY_ENV, " https://ghproxy.net ")
    assert resolve_proxy() == "https://ghproxy.net/"


def test_rewrite_is_scoped_to_github(monkeypatch):
    monkeypatch.delenv(PROXY_ENV, raising=False)
    rewrite = f"url.{DEFAULT_PROXY}{GITHUB_BASE}.insteadOf={GITHUB_BASE}"
    assert proxy_config_args() == ["-c", rewrite]


def test_candidates_try_the_mirror_then_a_direct_connection(monkeypatch):
    monkeypatch.delenv(PROXY_ENV, raising=False)
    clone = ["git", "clone", "--depth", "1", f"{GITHUB_BASE}a/b.git", "dest"]
    mirrored, direct = candidate_commands(clone)
    # The `-c` rewrite is a git global option, so it goes after the executable.
    assert mirrored == ["git", *proxy_config_args(), *clone[1:]]
    assert direct == clone


def test_git_requests_reach_the_mirror(monkeypatch):
    """A real git invocation must resolve github.com to the mirror base."""
    monkeypatch.delenv(PROXY_ENV, raising=False)
    requests = []

    class _Mirror(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - stdlib handler name
            requests.append(self.path)
            self.send_response(404)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), _Mirror)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    mirror = f"http://127.0.0.1:{server.server_address[1]}"
    # Isolate the rewrite under test from the user's global git config.
    env = dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_SYSTEM=os.devnull)
    try:
        result = subprocess.run(
            ["git", *proxy_config_args(mirror), "ls-remote", f"{GITHUB_BASE}a/b.git"],
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
        )
    finally:
        server.shutdown()
        thread.join()

    # The fake mirror answers 404; reaching it is the assertion.
    assert result.returncode != 0, result.stderr
    assert requests, "git never contacted the mirror"
    assert requests[0].startswith("/https://github.com/a/b.git/info/refs"), requests
    assert any("git-upload-pack" in request for request in requests), requests


def test_setup_py_has_no_network_git_call_outside_the_mirror_helper():
    """Both clones must stay on the mirror-aware retry path in setup.py."""
    tree = ast.parse((ROOT / "setup.py").read_text(encoding="utf-8"))
    network_calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
        if name != "check_call":
            continue
        literal = ast.dump(node.args[0])
        if "'clone'" in literal or "'fetch'" in literal:
            network_calls.append(node)
    assert not network_calls, "use _git_check_call_with_retries for network git calls"


# The orchestration below lives in setup.py, which imports torchada and torch at
# module scope; lift just the build-extension class out of its source instead.
_FAKE_GIT = """\
#!/usr/bin/env bash
set -u
printf '%s\\n' "$*" >> "${FAKE_GIT_LOG}"
case " $* " in
  *" -c url."*) mode="${FAKE_GIT_MIRROR:-ok}" ;;
  *) mode="${FAKE_GIT_DIRECT:-ok}" ;;
esac
if [[ "${1:-}" == "clone" ]]; then
  mkdir -p "${@: -1}"  # a clone creates the destination before it can fail
fi
case "${mode}" in
  ok) exit 0 ;;
  hang) sleep 30; exit 0 ;;
  *) exit 128 ;;
esac
"""


def _build_ext_class():
    tree = ast.parse((ROOT / "setup.py").read_text(encoding="utf-8"))
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.ClassDef) and item.name == "_CustomBuildExt"
    )
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "os": os,
        "subprocess": subprocess,
        "shutil": shutil,
        "time": time,
        "Path": Path,
        "BuildExtension": object,
        "candidate_commands": candidate_commands,
        "resolve_proxy": resolve_proxy,
        "PROXY_ENV": PROXY_ENV,
    }
    exec(compile(module, "setup.py", "exec"), namespace)
    return namespace["_CustomBuildExt"]


def _fake_git(tmp_path, monkeypatch, mirror="ok", direct="ok"):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    git = bin_dir / "git"
    git.write_text(_FAKE_GIT, encoding="utf-8")
    git.chmod(0o755)
    log = tmp_path / "git.log"
    monkeypatch.setenv("PATH", os.pathsep.join([str(bin_dir), os.environ["PATH"]]))
    monkeypatch.setenv("FAKE_GIT_LOG", str(log))
    monkeypatch.setenv("FAKE_GIT_MIRROR", mirror)
    monkeypatch.setenv("FAKE_GIT_DIRECT", direct)
    monkeypatch.setenv("VLLM_MUSA_GIT_RETRY_DELAY_S", "0")
    return log


def _calls(log):
    return log.read_text(encoding="utf-8").splitlines()


def test_clone_goes_through_the_mirror_when_it_works(tmp_path, monkeypatch):
    monkeypatch.delenv(PROXY_ENV, raising=False)
    log = _fake_git(tmp_path, monkeypatch)
    _build_ext_class()._git_check_call_with_retries(
        ["git", "clone", f"{GITHUB_BASE}a/b.git", str(tmp_path / "dest")]
    )
    calls = _calls(log)
    assert len(calls) == 1, calls
    # The command-scoped `-c` rewrite is a git global option: it comes first.
    assert calls[0].startswith("-c url.https://gh-proxy.com/"), calls


def test_mirror_failure_falls_back_to_a_direct_clone(tmp_path, monkeypatch):
    monkeypatch.delenv(PROXY_ENV, raising=False)
    monkeypatch.setenv("VLLM_MUSA_GIT_RETRY_ATTEMPTS", "2")
    log = _fake_git(tmp_path, monkeypatch, mirror="fail")
    dest = tmp_path / "dest"
    _build_ext_class()._git_check_call_with_retries(
        ["git", "clone", f"{GITHUB_BASE}a/b.git", str(dest)], cleanup_path=dest
    )
    calls = _calls(log)
    assert len(calls) == 3, calls  # two mirror attempts, then one direct
    assert "-c url." not in calls[-1]
    assert dest.is_dir()


def test_disabled_mirror_clones_directly_once(tmp_path, monkeypatch):
    monkeypatch.setenv(PROXY_ENV, "0")
    log = _fake_git(tmp_path, monkeypatch)
    _build_ext_class()._git_check_call_with_retries(
        ["git", "clone", f"{GITHUB_BASE}a/b.git", str(tmp_path / "dest")]
    )
    calls = _calls(log)
    assert len(calls) == 1, calls
    assert "-c url." not in calls[0]


def test_a_hanging_mirror_is_bounded_and_falls_back(tmp_path, monkeypatch):
    monkeypatch.delenv(PROXY_ENV, raising=False)
    monkeypatch.setenv("VLLM_MUSA_GIT_PROXY_TIMEOUT_S", "1")
    log = _fake_git(tmp_path, monkeypatch, mirror="hang")
    dest = tmp_path / "dest"
    _build_ext_class()._git_check_call_with_retries(
        ["git", "clone", f"{GITHUB_BASE}a/b.git", str(dest)], cleanup_path=dest
    )
    calls = _calls(log)
    assert len(calls) == 3, calls
    assert "-c url." not in calls[-1]
    assert dest.is_dir()


def test_exhausted_attempts_raise_the_git_error(tmp_path, monkeypatch):
    monkeypatch.delenv(PROXY_ENV, raising=False)
    monkeypatch.setenv("VLLM_MUSA_GIT_RETRY_ATTEMPTS", "2")
    log = _fake_git(tmp_path, monkeypatch, mirror="fail", direct="fail")
    dest = tmp_path / "dest"
    with pytest.raises(subprocess.CalledProcessError):
        _build_ext_class()._git_check_call_with_retries(
            ["git", "clone", f"{GITHUB_BASE}a/b.git", str(dest)], cleanup_path=dest
        )
    assert len(_calls(log)) == 4, _calls(log)  # two mirror, two direct
