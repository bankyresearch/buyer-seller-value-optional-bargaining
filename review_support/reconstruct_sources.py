"""Rebuild numerical source tables in a new directory; no network or model load."""
from pathlib import Path
import argparse, importlib.util, json, hashlib, shutil

root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--craigslist-dir',type=Path)
args=parser.parse_args()
out=args.output.resolve()
assert not out.exists(), 'Use a NEW output directory.'
spec=importlib.util.spec_from_file_location('source_processing',root/'multidataset_bargaining/prepare_data.py')
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
out.mkdir(parents=True)
mod.OUT=out
mod.amazon();mod.casino()
if args.craigslist_dir:
    audit=json.loads((root/'multidataset_bargaining/data/processed/craigslist_audit.json').read_text())
    for e in audit['source']:
        p=args.craigslist_dir/e['file']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==e['sha256'], 'Wrong source bytes: '+str(p)
    # The original function resolves ROOT-relative source paths. Give it a new
    # scratch tree containing only the verified inputs; never edit source code.
    scratch=out/'source_input'
    target=scratch/'research_paper_bm3il/data/raw/craigslist'
    target.mkdir(parents=True)
    for e in audit['source']: shutil.copy2(args.craigslist_dir/e['file'],target/e['file'])
    mod.WORK=scratch
    mod.craigslist()
report=[]
for p in sorted(out.glob('*.csv')):
    saved=root/'multidataset_bargaining/data/processed'/p.name
    record={'file':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    if saved.exists(): record['matches_packaged_bytes']=p.read_bytes()==saved.read_bytes()
    report.append(record)
(out/'reconstruction_report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps(report,indent=2))
