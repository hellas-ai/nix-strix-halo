#!/usr/bin/env python3
"""Stage a verified AOT seed once into a fresh, versioned writable cache."""
import argparse
import errno
import importlib.util
from pathlib import Path
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--seed-tool', type=Path, required=True)
p.add_argument('--seed', type=Path, required=True)
p.add_argument('--identity', type=Path, required=True)
p.add_argument('--runtime', required=True)
p.add_argument('--dest', type=Path, required=True)
a = p.parse_args()
try:
    spec = importlib.util.spec_from_file_location('glm53_cache_seed', a.seed_tool)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ident = module.identity(a.identity)
    if ident['runtime'] != a.runtime:
        raise ValueError('seed runtime does not match launcher runtime')
    module.verify(a.seed, ident)
    a.dest.parent.mkdir(parents=True, exist_ok=True)
    if not a.dest.exists() and not a.dest.is_symlink():
        try:
            module.install(a.seed, a.dest, ident)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
        except ValueError as exc:
            if str(exc) != 'destination already exists':
                raise
        # A concurrent starter may have published its complete directory.
    if a.dest.is_symlink() or not a.dest.is_dir():
        raise ValueError('destination must be a real directory')
except (OSError, ValueError, TypeError, KeyError) as exc:
    print(f'glm53-aot-stage: {exc}', file=sys.stderr)
    raise SystemExit(1)
