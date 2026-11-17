# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The GitHub mirror and pinned fetch used by the third-party source step."""

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
    DEFAULT_STALL_S,
    GITHUB_BASE,
    PROXY_ENV,
    STALL_ENV,
    candidate_commands,
    mirror_config_args,
    resolve_proxy,
    resolve_stall_s,
)

ROOT = Path(__file__).resolve().parents[1]
REWRITE = f"url.{DEFAULT_PROXY}{GITHUB_BASE}.insteadOf={GITHUB_BASE}"


def test_unset_uses_the_default_mirror(monkeypatch):
    monkeypatch.delenv(PROXY_ENV, raising=False)
    assert resolve_proxy() == DEFAULT_PROXY == "https://gh-proxy.com/"


@pytest.mark.parametrize("value", ["", "  ", "1", "true", "ON", "yes", "default"])
def test_enabling_values_select_the_default_mirror(monkeypatch, value):
    monkeypatch.setenv(PROXY_ENV, value)
    assert resolve_proxy() == DEFAULT_PROXY


@pytest.mark.parametrize("value", ["0", "false", "OFF", "no", "none", "disabled"])
def test_disabling_values_fetch_directly(monkeypatch, value):
    monkeypatch.setenv(PROXY_ENV, value)
    assert resolve_proxy() == ""
    assert mirror_config_args() == []
    assert candidate_commands(["git", "fetch", "x"]) == [["git", "fetch", "x"]]


def test_custom_mirror_is_normalized(monkeypatch):
    monkeypatch.setenv(PROXY_ENV, " https://ghproxy.net ")
    assert resolve_proxy() == "https://ghproxy.net/"


def test_mirrored_attempts_rewrite_github_and_guard_the_transfer(monkeypatch):
    monkeypatch.delenv(PROXY_ENV, raising=False)
    monkeypatch.delenv(STALL_ENV, raising=False)
    assert mirror_config_args() == [
        "-c",
        REWRITE,
        "-c",
        "http.lowSpeedLimit=1",
        "-c",
        f"http.lowSpeedTime={DEFAULT_STALL_S}",
    ]


@pytest.mark.parametrize("value", ["0", "off"])
def test_stall_guard_can_be_disabled(monkeypatch, value):
    monkeypatch.setenv(STALL_ENV, value)
    assert mirror_config_args() == ["-c", REWRITE]


def test_stall_window_is_configurable(monkeypatch):
    monkeypatch.setenv(STALL_ENV, "5")
    assert "http.lowSpeedTime=5" in mirror_config_args()


def test_empty_stall_window_keeps_the_default(monkeypatch):
    monkeypatch.setenv(STALL_ENV, "")
    assert resolve_stall_s() == DEFAULT_STALL_S
    monkeypatch.setenv(STALL_ENV, "-5")
    assert resolve_stall_s() == 0


def test_candidates_try_the_mirror_then_a_direct_fetch(monkeypatch):
    monkeypatch.delenv(PROXY_ENV, raising=False)
    fetch = ["git", "fetch", "--depth", "1", "origin", "abc123"]
    mirrored, direct = candidate_commands(fetch)
    # `-c` options are git global options: they follow the executable.
    assert mirrored == ["git", *mirror_config_args(), *fetch[1:]]
    assert direct == fetch


def _git_env():
    """Git with only the rewrite under test, not the user's global config."""
    return dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_SYSTEM=os.devnull)


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

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Mirror)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    mirror = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        result = subprocess.run(
            ["git", *mirror_config_args(mirror), "ls-remote", f"{GITHUB_BASE}a/b.git"],
            capture_output=True,
            text=True,
            timeout=60,
            env=_git_env(),
        )
    finally:
        server.shutdown()
        thread.join()

    # The fake mirror answers 404; reaching it is the assertion.
    assert result.returncode != 0, result.stderr
    assert requests, "git never contacted the mirror"
    assert requests[0].startswith("/https://github.com/a/b.git/info/refs"), requests
    assert any("git-upload-pack" in request for request in requests), requests


def test_a_stalled_mirror_fails_instead_of_hanging(monkeypatch):
    """git must abort when the mirror accepts the request and stops sending."""
    monkeypatch.delenv(PROXY_ENV, raising=False)

    class _Stalling(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):  # noqa: N802 - stdlib handler name
            self.send_response(200)
            self.send_header("Content-Length", "1000")
            self.end_headers()
            self.wfile.write(b"001e# service=git-upload-pack\n")
            self.wfile.flush()
            time.sleep(30)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Stalling)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    mirror = f"http://127.0.0.1:{server.server_address[1]}"
    started = time.monotonic()
    try:
        result = subprocess.run(
            [
                "git",
                *mirror_config_args(mirror, stall_s="2"),
                "ls-remote",
                f"{GITHUB_BASE}a/b.git",
            ],
            capture_output=True,
            text=True,
            timeout=40,
            env=_git_env(),
        )
    finally:
        server.shutdown()
        thread.join()
    elapsed = time.monotonic() - started

    assert result.returncode != 0, result.stderr
    assert "too slow" in result.stderr.lower(), result.stderr
    assert elapsed < 15, elapsed


def test_setup_py_keeps_network_git_calls_in_the_mirror_helper():
    """No clone/fetch may reach the network outside the mirror-aware helper."""
    source = (ROOT / "setup.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        receiver = node.func
        module = getattr(receiver, "value", None)
        if not (isinstance(module, ast.Name) and module.id == "subprocess"):
            continue
        if not isinstance(getattr(receiver, "attr", None), str):
            continue
        literal = ast.dump(node.args[0])
        if "'clone'" in literal or "'fetch'" in literal:
            offenders.append((node.lineno, receiver.attr))
    assert not offenders, f"use _git_check_call_with_retries: {offenders}"
    # The pins are fetched shallowly instead of cloned with their history.
    assert '"--depth"' in source
    assert '"clone"' not in source


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
case "${mode}" in
  ok) exit 0 ;;
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
    # Deterministic attempts and delays, independent of the caller's shell.
    monkeypatch.setenv("VLLM_MUSA_GIT_RETRY_ATTEMPTS", "2")
    monkeypatch.setenv("VLLM_MUSA_GIT_RETRY_DELAY_S", "0")
    monkeypatch.setenv(STALL_ENV, "0")
    return log


def _calls(log):
    return log.read_text(encoding="utf-8").splitlines()


def test_fetch_goes_through_the_mirror_when_it_works(tmp_path, monkeypatch):
    monkeypatch.delenv(PROXY_ENV, raising=False)
    log = _fake_git(tmp_path, monkeypatch)
    _build_ext_class()._git_check_call_with_retries(
        ["git", "fetch", "--depth", "1", "origin", "abc123"]
    )
    calls = _calls(log)
    assert len(calls) == 1, calls
    assert calls[0].startswith("-c url.https://gh-proxy.com/"), calls


def test_mirror_failure_falls_back_to_a_direct_fetch(tmp_path, monkeypatch):
    monkeypatch.delenv(PROXY_ENV, raising=False)
    log = _fake_git(tmp_path, monkeypatch, mirror="fail")
    _build_ext_class()._git_check_call_with_retries(
        ["git", "fetch", "--depth", "1", "origin", "abc123"]
    )
    calls = _calls(log)
    assert len(calls) == 3, calls  # two mirrored attempts, then one direct
    assert "-c url." not in calls[-1]


def test_disabled_mirror_fetches_directly_once(tmp_path, monkeypatch):
    monkeypatch.setenv(PROXY_ENV, "0")
    log = _fake_git(tmp_path, monkeypatch)
    _build_ext_class()._git_check_call_with_retries(
        ["git", "fetch", "--depth", "1", "origin", "abc123"]
    )
    calls = _calls(log)
    assert len(calls) == 1, calls
    assert "-c url." not in calls[0]


def test_exhausted_attempts_raise_the_git_error(tmp_path, monkeypatch):
    monkeypatch.delenv(PROXY_ENV, raising=False)
    log = _fake_git(tmp_path, monkeypatch, mirror="fail", direct="fail")
    with pytest.raises(subprocess.CalledProcessError):
        _build_ext_class()._git_check_call_with_retries(
            ["git", "fetch", "--depth", "1", "origin", "abc123"]
        )
    assert len(_calls(log)) == 4, _calls(log)  # two mirrored, two direct


def _source_repo(tmp_path):
    """A bare remote holding two commits; return (remote, first_sha)."""
    src = tmp_path / "src"
    subprocess.check_call(["git", "init", "-q", str(src)])
    env = dict(
        _git_env(),
        GIT_AUTHOR_NAME="t",
        GIT_AUTHOR_EMAIL="t@e",
        GIT_COMMITTER_NAME="t",
        GIT_COMMITTER_EMAIL="t@e",
    )
    (src / "pinned.txt").write_text("pinned\n", encoding="utf-8")
    subprocess.check_call(["git", "add", "pinned.txt"], cwd=src, env=env)
    subprocess.check_call(["git", "commit", "-qm", "pinned"], cwd=src, env=env)
    first = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=src, text=True
    ).strip()
    (src / "later.txt").write_text("later\n", encoding="utf-8")
    subprocess.check_call(["git", "add", "later.txt"], cwd=src, env=env)
    subprocess.check_call(["git", "commit", "-qm", "later"], cwd=src, env=env)
    remote = tmp_path / "remote.git"
    subprocess.check_call(["git", "clone", "-q", "--bare", str(src), str(remote)])
    return remote, first


def test_pinned_fetch_materializes_only_that_commit(tmp_path, monkeypatch):
    monkeypatch.setenv(PROXY_ENV, "0")
    remote, first = _source_repo(tmp_path)
    dest = tmp_path / "third_party" / "pin"
    _build_ext_class()._clone_and_checkout(dest, str(remote), first)

    assert (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=dest, text=True
        ).strip()
        == first
    )
    assert (dest / "pinned.txt").is_file()
    assert not (dest / "later.txt").exists()
    assert (
        subprocess.check_output(
            ["git", "rev-list", "--count", "HEAD"], cwd=dest, text=True
        ).strip()
        == "1"
    )
    assert subprocess.check_output(
        ["git", "remote", "get-url", "origin"], cwd=dest, text=True
    ).strip() == str(remote)


def test_an_already_fetched_pin_needs_no_network(tmp_path, monkeypatch):
    monkeypatch.setenv(PROXY_ENV, "0")
    remote, first = _source_repo(tmp_path)
    dest = tmp_path / "third_party" / "pin"
    build_ext = _build_ext_class()
    build_ext._clone_and_checkout(dest, str(remote), first)
    # The remote disappearing must not matter when the pin is already local.
    shutil.rmtree(remote)
    build_ext._clone_and_checkout(dest, str(remote), first)
    assert (dest / "pinned.txt").is_file()


def test_a_non_git_directory_fails_before_it_is_cleaned(tmp_path, monkeypatch):
    monkeypatch.setenv(PROXY_ENV, "0")
    remote, first = _source_repo(tmp_path)
    dest = tmp_path / "third_party" / "pin"
    dest.mkdir(parents=True)
    (dest / "keep.txt").write_text("keep\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="not a git checkout"):
        _build_ext_class()._clone_and_checkout(dest, str(remote), first)
    assert (dest / "keep.txt").is_file()
