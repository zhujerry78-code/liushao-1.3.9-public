"""Build new outputs privately, then publish without replacing existing data."""
from contextlib import contextmanager
import errno
import os
from pathlib import Path
import shutil
import sys
import tempfile


def _publish_no_replace(source, destination):
    if os.name == 'nt':
        # Windows rename fails if any destination entry already exists.
        os.rename(source, destination)
        return
    if sys.platform.startswith('linux'):
        # POSIX rename alone can replace an empty directory created by another
        # process. Linux RENAME_NOREPLACE preserves that competing output too.
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        rename = getattr(libc, 'renameat2', None)
        if rename is None:
            raise OSError(errno.ENOSYS, '当前系统不支持不覆盖的目录发布。', str(destination))
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        if rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1):
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code), str(destination))
        return
    # Do not fall back to a check-then-replace rename on other platforms.
    raise OSError(errno.ENOTSUP, '当前系统不支持不覆盖的目录发布。', str(destination))


def _check_owned_stage(stage, identity, parent):
    stat = stage.lstat()
    if (stage.is_symlink() or stage.is_junction() or not stage.is_dir()
            or (stat.st_dev, stat.st_ino) != identity
            or stage.resolve().parent != parent or stage.resolve() != stage):
        raise OSError('暂存路径身份已变化，未清理该路径：' + str(stage))


def _remove_owned_stage(stage, identity, parent):
    if not os.path.lexists(stage):
        return
    _check_owned_stage(stage, identity, parent)
    # The absolute directory and its creation identity have been checked. rmtree
    # removes links inside this private tree without traversing their targets.
    shutil.rmtree(stage)


@contextmanager
def staged_output_directory(destination):
    """Yield a private sibling directory and publish only on successful exit.

    Existing destinations (including broken links) are refused. Cleanup never
    touches the final destination, its parents, or a replacement staging entry.
    """
    requested = Path(destination).absolute()
    if os.path.lexists(requested):
        raise FileExistsError(errno.EEXIST, '输出路径已存在，请使用新的目录。', str(requested))
    parent = requested.parent.resolve()
    parent.mkdir(parents=True, exist_ok=True)
    parent = parent.resolve()
    destination = parent / requested.name
    if os.path.lexists(destination):
        raise FileExistsError(errno.EEXIST, '输出路径已存在，请使用新的目录。', str(destination))
    stage = Path(tempfile.mkdtemp(prefix='.' + requested.name + '.tmp-', dir=parent)).resolve()
    stat = stage.lstat()
    identity = stat.st_dev, stat.st_ino
    try:
        yield stage
        _check_owned_stage(stage, identity, parent)
        _publish_no_replace(stage, destination)
    except BaseException as error:
        try:
            _remove_owned_stage(stage, identity, parent)
        except OSError as cleanup_error:
            error.add_note('结果未发布；暂存清理未完成：' + str(cleanup_error))
        raise
