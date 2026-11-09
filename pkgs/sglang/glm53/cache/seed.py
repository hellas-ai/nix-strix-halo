#!/usr/bin/env python3
"""Export, verify and relocate a pinned Triton warm cache; does not compile kernels."""
import argparse
import ctypes
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

KEY = re.compile(r"^[A-Z2-7]{52}$")


def fail(message):
    raise ValueError(message)


def canonical(obj):
    return (json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=True) + '\n').encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def regular(path):
    if path.is_symlink() or not path.is_file():
        fail(f'not a regular file: {path}')
    return path.read_bytes()


def identity(path):
    obj = json.loads(regular(path))
    if not isinstance(obj, dict) or set(obj) != {'runtime', 'target', 'configuration'}:
        fail('identity must contain exactly runtime, target, configuration')
    if not isinstance(obj['runtime'], str) or not obj['runtime'].startswith('/nix/store/'):
        fail('runtime must be an absolute Nix store path')
    if not isinstance(obj['target'], str) or not obj['target']:
        fail('target must be a nonempty string')
    if not isinstance(obj['configuration'], dict) or not obj['configuration']:
        fail('configuration must be a nonempty object')
    return obj


def group(data, directory, *, source):
    obj = json.loads(data)
    if not isinstance(obj, dict) or set(obj) != {'child_paths'} or not isinstance(obj['child_paths'], dict) or not obj['child_paths']:
        fail('invalid Triton group')
    children = {}
    for name, value in obj['child_paths'].items():
        if not isinstance(name, str) or not name or Path(name).name != name or not isinstance(value, str):
            fail('invalid group child name')
        if source:
            p = Path(value)
            if not p.is_absolute() or p.parent != directory or p.name != name:
                fail(f'group child outside its source directory: {value}')
            regular(p)
        elif value != name:
            fail('seed group paths must be relative child names')
        children[name] = name
    return {'child_paths': children}


def export(source, dest, ident):
    source = source.resolve(strict=True)
    records = {}
    with staging(dest) as stage:
        payload = stage / 'payload'
        payload.mkdir()
        for d in sorted(source.iterdir()):
            if d.name.startswith('tmp.'):
                continue
            if not KEY.fullmatch(d.name) or d.is_symlink() or not d.is_dir():
                fail(f'unexpected cache entry: {d}')
            entries = sorted(d.iterdir())
            groups = [p for p in entries if p.name.startswith('__grp__') and p.name.endswith('.json')]
            if not groups:
                continue  # Incomplete compilation has no reusable group.
            out = payload / d.name
            out.mkdir()
            for p in entries:
                if p.name == 'lock' or p.name.startswith('tmp.'):
                    continue
                data = regular(p)
                if p in groups:
                    data = canonical(group(data, d, source=True))
                (out / p.name).write_bytes(data)
                records[f'{d.name}/{p.name}'] = digest(data)
        if not records:
            fail('no complete cache entries')
        manifest = {'schema': 1, 'identity': ident, 'files': records}
        (stage / 'manifest.json').write_bytes(canonical(manifest))
        verify(stage, ident)
        publish(stage, dest)


class staging:
    def __init__(self, dest):
        self.dest = Path(dest).absolute()
    def __enter__(self):
        if self.dest.exists() or self.dest.is_symlink():
            fail('destination already exists')
        self.path = Path(tempfile.mkdtemp(prefix='.triton-seed-', dir=self.dest.parent))
        return self.path
    def __exit__(self, *args):
        if self.path.exists():
            shutil.rmtree(self.path)


def _rename_noreplace(stage, dest):
    # Linux renameat2 prevents replacing even an empty directory in a race.
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        rename = libc.renameat2
    except AttributeError:
        raise OSError(errno.ENOSYS, 'renameat2 unavailable')
    if rename(-100, os.fsencode(stage), -100, os.fsencode(dest), 1):
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(dest))


def publish(stage, dest):
    try:
        _rename_noreplace(stage, dest)
        return
    except OSError as exc:
        if exc.errno not in {errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP}:
            raise
    # Some NFS servers reject RENAME_NOREPLACE. All cooperating publishers
    # lock the same persistent sibling before testing and renaming. Do not
    # unlink the lock: waiters could otherwise lock different inodes. This
    # is advisory and cannot prevent replacement by non-cooperating writers.
    dest = Path(dest).absolute()
    lock_name = '.triton-seed-publish-' + hashlib.sha256(os.fsencode(dest.name)).hexdigest() + '.lock'
    flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW
    fd = os.open(dest.parent / lock_name, flags, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        if os.path.lexists(dest):
            raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), str(dest))
        os.rename(stage, dest)
    finally:
        os.close(fd)


def verify(seed, ident):
    seed = Path(seed)
    if seed.is_symlink() or not seed.is_dir():
        fail('seed must be a directory')
    manifest = json.loads(regular(seed / 'manifest.json'))
    if set(manifest) != {'schema', 'identity', 'files'} or manifest['schema'] != 1 or manifest['identity'] != ident:
        fail('manifest schema or runtime identity mismatch')
    files = manifest['files']
    if not isinstance(files, dict) or not files:
        fail('empty manifest')
    if set(p.name for p in seed.iterdir()) != {'manifest.json', 'payload'}:
        fail('unexpected seed contents')
    payload = seed / 'payload'
    if payload.is_symlink() or not payload.is_dir():
        fail('invalid payload')
    seen = set()
    for d in payload.iterdir():
        if not KEY.fullmatch(d.name) or d.is_symlink() or not d.is_dir():
            fail('invalid cache directory')
        groups = 0
        for p in d.iterdir():
            rel = f'{d.name}/{p.name}'
            data = regular(p)
            if files.get(rel) != digest(data):
                fail(f'hash mismatch or unlisted file: {rel}')
            seen.add(rel)
            if p.name.startswith('__grp__') and p.name.endswith('.json'):
                groups += 1
                children = group(data, d, source=False)['child_paths']
                for child in children:
                    if f'{d.name}/{child}' not in files:
                        fail(f'missing group child: {child}')
        if not groups:
            fail('cache directory has no group')
    if seen != set(files):
        fail('manifest file set mismatch')
    return manifest


def install(seed, dest, ident):
    manifest = verify(seed, ident)
    payload = Path(seed) / 'payload'
    with staging(dest) as stage:
        for d in payload.iterdir():
            out = stage / d.name
            out.mkdir()
            for p in d.iterdir():
                data = regular(p)
                if manifest['files'].get(f'{d.name}/{p.name}') != digest(data):
                    fail('seed changed during installation')
                if p.name.startswith('__grp__') and p.name.endswith('.json'):
                    obj = group(data, d, source=False)
                    obj['child_paths'] = {k: str(Path(dest).absolute() / d.name / k) for k in obj['child_paths']}
                    data = canonical(obj)
                (out / p.name).write_bytes(data)
        # Recheck source to detect corruption during the copy.
        verify(seed, ident)
        publish(stage, dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('export', 'verify', 'install'):
        p = sub.add_parser(name)
        p.add_argument('--identity', required=True, type=Path)
        p.add_argument('--seed', required=True)
        if name == 'export': p.add_argument('--cache', required=True)
        if name == 'install': p.add_argument('--dest', required=True)
    args = parser.parse_args()
    try:
        ident = identity(args.identity)
        if args.command == 'export': export(Path(args.cache), Path(args.seed), ident)
        elif args.command == 'verify': verify(Path(args.seed), ident)
        else: install(Path(args.seed), Path(args.dest), ident)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f'triton-cache-seed: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
