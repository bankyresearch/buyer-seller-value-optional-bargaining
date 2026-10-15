"""Check an extracted package with only Python's standard library."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json


def verify(root):
    root = root.resolve()
    manifest = json.loads((root/'SHA256SUMS.json').read_text(encoding='utf-8'))
    for entry in manifest['files']:
        rel = PurePosixPath(entry['path'])
        assert not rel.is_absolute() and '..' not in rel.parts, rel
        path = root.joinpath(*rel.parts).resolve()
        assert path.is_relative_to(root) and path.is_file(), path
        assert path.stat().st_size == entry['bytes'], f'Size mismatch: {rel}'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256'], f'Hash mismatch: {rel}'
    print(f"Verified {len(manifest['files'])} explicitly listed files; manifest excludes itself.")
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    verify(parser.parse_args().root)
