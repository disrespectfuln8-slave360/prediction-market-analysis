"""Single-instance guard for the bot launcher scripts.

Two copies of the same launcher running at once double API traffic
(rate-limit storms) and race each other's writes to the shared ledger,
learning, and pending-fill files — this actually happened: run_all.py,
run_poly.py, and run_polyus.py were each found running twice (once from
.venv, once from a stray uv-managed interpreter), corrupting a whole
session's results.

Uses an OS-level file lock (msvcrt on Windows, fcntl elsewhere) rather
than a pidfile: the kernel releases the lock the instant the process dies,
so there is no stale-lock state to detect or clean up after a crash.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Anchored to the repo root, NOT the process CWD — the logon-triggered
# Task Scheduler jobs and manual launches must always collide on the same
# lock file no matter where they were started from.
LOCK_DIR = Path(__file__).resolve().parents[2] / "data" / "locks"


def acquire_single_instance_lock(name: str):
    """Take an exclusive lock for `name`, or exit(1) with a clear message if
    another process already holds it.

    Returns the open lock-file handle. The CALLER MUST KEEP A REFERENCE to it
    for the life of the process — the lock is released when the handle is
    garbage-collected or the process exits.
    """
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    path = LOCK_DIR / f"{name}.lock"
    fh = open(path, "a+")
    try:
        fh.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        sys.exit(
            f"ERROR: another '{name}' instance is already running (lock file: {path}).\n"
            "Two copies double API traffic and corrupt the shared ledger/learning "
            "files. Stop the other process first (check Task Manager for python "
            "processes running the same script)."
        )
    try:
        fh.seek(0)
        fh.truncate()
        fh.write(str(os.getpid()))
        fh.flush()
    except OSError:
        pass  # informational only — the lock itself is what matters
    return fh
