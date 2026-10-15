"""Optional pinned source download. No downloaded code is executed."""
from pathlib import Path
import json,urllib.request,hashlib
ROOT=Path(__file__).resolve().parent
manifest=json.loads((ROOT/'data/raw/acquisition.json').read_text())
for group in ['amazon','casino']:
    for e in manifest[group]:
        p=ROOT/Path(e['path'].replace('\\','/'));p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists():data=p.read_bytes()
        else:
            with urllib.request.urlopen(e['url'],timeout=60) as r:data=r.read(50_000_001)
        assert len(data)<=50_000_000 and hashlib.sha256(data).hexdigest()==e['sha256'],e['url']
        if not p.exists():p.write_bytes(data)
        print('VERIFIED',p.name)
