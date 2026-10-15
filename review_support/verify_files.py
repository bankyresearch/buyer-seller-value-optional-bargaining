"""Verify all review material parts after extraction; standard library only."""
from pathlib import Path
import hashlib, json

root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'review_support/FILE_MANIFEST.json').read_text(encoding='utf8'))
failures=[]
for e in manifest['files']:
    p=root/e['path']
    if not p.is_file(): failures.append([e['path'],'missing'])
    elif hashlib.sha256(p.read_bytes()).hexdigest()!=e['sha256']:
        failures.append([e['path'],'hash mismatch'])
print(json.dumps({'verified_files':len(manifest['files'])-len(failures),'failures':failures},indent=2))
raise SystemExit(bool(failures))
