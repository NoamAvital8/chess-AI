"""Allow one training/play writer per shared checkpoint directory."""

from __future__ import annotations

import os
from contextlib import contextmanager
from functools import wraps
from pathlib import Path


@contextmanager
def checkpoint_lock(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    # OS locks are released on close or process exit; the file can remain in place.
    with (directory / ".model.lock").open("a+b") as handle:
        if os.fstat(handle.fileno()).st_size == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(
                f"Cannot lock checkpoint directory {directory.resolve()}. "
                "Another training or play session may be using it; finish that session first."
            ) from error
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def single_writer(function):
    @wraps(function)
    def guarded(args, *positional, **keywords):
        with checkpoint_lock(Path(args.checkpoint_dir)):
            return function(args, *positional, **keywords)
    return guarded
