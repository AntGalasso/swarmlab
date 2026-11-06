"""Reading a file the experiment is still writing, holding it open for as short a time as possible.

On Windows nothing the reader can do lets the experiment `os.replace` over a file that is open: MoveFileEx fails with
"Access is denied" while ANY handle is open, whatever sharing flags it has. (Deleting and renaming a file does work
when the reader opened it with FILE_SHARE_DELETE, which Python's open() does not pass.) So the sync child never holds
an experiment file open while it talks to the network: `copy_version` opens it with
FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, copies one consistent version to a staging file and closes it
at once. The upload then reads the staged copy. The remaining window is the local copy of one file.

The version copied is the first `fstat` size bytes of the handle that was opened: a file replaced after listing gives
its full new version, a file appended to gives the prefix that was there; never a torn mix.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import BinaryIO, Tuple

GENERIC_READ = 0x80000000
SHARE_ALL = 0x1 | 0x2 | 0x4                  # FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE
OPEN_EXISTING = 3
FILE_ATTRIBUTE_NORMAL = 0x80
LONG_PATH_AT = 240
CHUNK = 1 << 20


def open_shared(path) -> BinaryIO:
    """The file opened for reading in binary mode, without blocking the writer's delete or rename. Raises OSError
    (FileNotFoundError for a file that is gone)."""
    if os.name != "nt":
        return Path(path).open("rb")             # POSIX: unlink and rename never fail because a reader exists
    import ctypes
    import msvcrt
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD,
                                   wintypes.DWORD, wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    name = os.path.abspath(path)
    if len(name) >= LONG_PATH_AT and not name.startswith("\\\\?\\"):
        name = "\\\\?\\" + name
    handle = kernel.CreateFileW(name, GENERIC_READ, SHARE_ALL, None, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, None)
    if handle is None or handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except OSError:
        kernel.CloseHandle(handle)
        raise
    return os.fdopen(descriptor, "rb")


def copy_version(source, destination) -> Tuple[int, int]:
    """Copies one consistent version of `source` to `destination`; returns its (mtime_ns, size). The source handle is
    closed before this returns. Raises OSError if the source is gone or unreadable."""
    with open_shared(source) as handle:
        stat = os.fstat(handle.fileno())
        with open(destination, "wb") as out:
            shutil.copyfileobj(_Capped(handle, stat.st_size), out, CHUNK)
    return stat.st_mtime_ns, stat.st_size


class _Capped:
    """read() of at most `limit` bytes in total, so a file appended to during the copy is not copied past what the
    handle had when it was opened."""

    def __init__(self, handle: BinaryIO, limit: int) -> None:
        self._handle, self._left = handle, limit

    def read(self, size: int = -1) -> bytes:
        size = self._left if size is None or size < 0 else min(size, self._left)
        data = self._handle.read(size)
        self._left -= len(data)
        return data
