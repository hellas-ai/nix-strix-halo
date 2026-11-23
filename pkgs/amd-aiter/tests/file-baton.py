"""CPU subprocess regressions; load lock/callers without importing AITER or Torch."""

import ast
import contextlib
import importlib.util
import logging
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import tempfile
from typing import Callable, Optional


JIT = Path(sys.argv[1]).resolve()
spec = importlib.util.spec_from_file_location("isolated_baton", JIT / "utils/file_baton.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
FileBaton = module.FileBaton


def core_functions(**extra):
    # Execute the actual small caller bodies, excluding GPU-only module imports.
    tree = ast.parse((JIT / "core.py").read_text())
    names = {"mp_lock", "clear_build", "get_module_custom_op"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == names
    for node in nodes:
        node.decorator_list = []
    namespace = dict(
        FileBaton=FileBaton, Callable=Callable, Optional=Optional, os=os,
        importlib=importlib, logger=logging.getLogger("test"), __mds={}, **extra,
    )
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(JIT / "core.py"), "exec"), namespace)
    return namespace


def child(mode, path):
    baton = FileBaton(path)
    if mode == "owner":
        assert baton.try_acquire()
        print("acquired", flush=True)
        sys.stdin.readline()
        baton.release()
    elif mode == "contender":
        assert not baton.try_acquire()
        print("waiting", flush=True)
        baton.wait()
        while not baton.try_acquire():
            baton.wait()
        print("acquired", flush=True)
        sys.stdin.readline()
        baton.release()
    elif mode == "missing-module":
        class ObservedBaton(FileBaton):
            def wait(self):
                print("waiting", flush=True)
                super().wait()

        functions = core_functions()
        functions["FileBaton"] = ObservedBaton

        def unexpected_build():
            raise AssertionError("A waiter must not silently retry the build")

        functions["mp_lock"](path, unexpected_build)
        os.environ["AITER_JIT_DIR"] = str(Path(path).parent)
        try:
            functions["get_module_custom_op"]("aiter_missing_killed_build")
        except ModuleNotFoundError:
            print("missing-module", flush=True)
        else:
            raise AssertionError("Owner death must not imply a successful import")
    else:
        raise AssertionError(mode)


@contextlib.contextmanager
def process(mode, path):
    proc = subprocess.Popen(
        [sys.executable, __file__, str(JIT), mode, str(path)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True,
    )
    try:
        yield proc
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.communicate(timeout=5)


def line(proc):
    assert select.select([proc.stdout], [], [], 5)[0], "child progress timeout"
    return proc.stdout.readline().strip()


def finish(proc, text=None):
    out, err = proc.communicate(input=text, timeout=5)
    assert proc.returncode == 0, (proc.returncode, out, err)
    return out


def reacquire(path):
    baton = FileBaton(path)
    assert baton.try_acquire(), "lock remained owned"
    assert not os.get_inheritable(baton.fd)
    baton.release()
    assert baton.fd is None


def tests():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        lock = root / "lock"
        lock.touch()  # The exact zero-byte leftover that previously wedged JIT.
        inode = lock.stat().st_ino
        reacquire(lock)
        reacquire(lock)
        assert lock.stat().st_ino == inode
        print("PASS stale-file reacquisition and persistent inode", flush=True)

        owner = FileBaton(lock)
        assert owner.try_acquire()
        try:
            with contextlib.ExitStack() as stack:
                peers = [stack.enter_context(process("contender", lock)) for _ in range(3)]
                for peer in peers:
                    assert line(peer) == "waiting"
                assert not select.select([p.stdout for p in peers], [], [], 0.1)[0]
                owner.release()
                while peers:
                    ready = select.select([p.stdout for p in peers], [], [], 5)[0]
                    assert len(ready) == 1, "multiple critical-section owners"
                    peer = next(p for p in peers if p.stdout is ready[0])
                    assert line(peer) == "acquired"
                    assert not FileBaton(lock).try_acquire()
                    assert lock.stat().st_ino == inode
                    others = [p.stdout for p in peers if p is not peer]
                    assert not select.select(others, [], [], 0.1)[0]
                    finish(peer, "release\n")
                    peers.remove(peer)
        finally:
            owner.release()
        reacquire(lock)
        print("PASS three-process contention and inode-stable handoff", flush=True)

        with process("owner", lock) as killed:
            assert line(killed) == "acquired"
            with process("missing-module", lock) as waiter:
                assert line(waiter) == "waiting"
                assert not select.select([waiter.stdout], [], [], 0.1)[0]
                killed.kill()
                killed.communicate(timeout=5)
                assert killed.returncode == -signal.SIGKILL
                assert finish(waiter).strip() == "missing-module"
        reacquire(lock)
        assert lock.stat().st_ino == inode
        print("PASS SIGKILL releases waiter; missing artifact still fails import", flush=True)

        functions = core_functions(bd_dir=str(root))

        def failure():
            raise ValueError("intentional caller failure")

        for main, final in ((failure, None), (lambda: None, failure)):
            try:
                functions["mp_lock"](str(lock), main, final)
            except ValueError as error:
                assert str(error) == "intentional caller failure"
            else:
                raise AssertionError("caller exception was swallowed")
            reacquire(lock)
        print("PASS MainFunc and FinalFunc failures release descriptors", flush=True)

        outer = root / "lock_module"
        build = root / "module/build"
        build.mkdir(parents=True)
        (build / "lock").touch()
        baton = FileBaton(outer)
        assert baton.try_acquire()
        try:
            outer_inode = outer.stat().st_ino
            functions["clear_build"]("module")
            assert not build.exists()
            assert outer.stat().st_ino == outer_inode
            assert not FileBaton(outer).try_acquire()
        finally:
            baton.release()
        print("PASS rebuild cleanup preserves held outer lock", flush=True)

        baton = FileBaton(lock)
        assert baton.try_acquire()
        sleeper = subprocess.Popen(
            [sys.executable, "-c", "print('ready', flush=True); input()"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, close_fds=False,
        )
        try:
            assert line(sleeper) == "ready"
            baton.release()
            reacquire(lock)  # Exec child is still alive, but cannot retain the lock.
            finish(sleeper, "exit\n")
        finally:
            baton.release()
            if sleeper.poll() is None:
                sleeper.kill()
            sleeper.communicate(timeout=5)
        print("PASS exec child does not inherit lock ownership", flush=True)
    assert "torch" not in sys.modules and "aiter" not in sys.modules


if __name__ == "__main__":
    if len(sys.argv) == 4:
        child(sys.argv[2], sys.argv[3])
    else:
        tests()
