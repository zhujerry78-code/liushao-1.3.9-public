"""Strict portable archive names and JSON objects shared by delivery tools."""
import json
from pathlib import PurePosixPath
import re


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result: raise ValueError('Duplicate JSON key: ' + key)
        result[key] = value
    return result


def read_json(content):
    return json.loads(content, object_pairs_hook=unique_object)


def safe_name(name):
    path = PurePosixPath(name)
    if (any(ord(c) < 32 or c in '*?<>|\"' for c in name) or not name or path.is_absolute() or '\\' in name or ':' in name
            or any(p in ('', '.', '..') or p[-1:] in (' ', '.') or
                   re.fullmatch(r'(?i)(con|prn|aux|nul|com[1-9¹²³]|lpt[1-9¹²³]) *(?:\..*)?', p)
                   for p in name.split('/'))):
        raise ValueError('Unsafe manifest path: ' + name)
    return path


def regular_file(root, name):
    safe_name(name)
    path = root / name
    for parent in [path, *path.parents]:
        if parent == root.parent: break
        if parent.is_symlink() or parent.is_junction():
            raise ValueError('Unsafe file link: ' + name)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Unsafe file: ' + name)
    return path
