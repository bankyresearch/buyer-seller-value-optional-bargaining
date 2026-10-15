"""Regenerate the 15 Craigslist-derived files listed in SOURCE_OMISSIONS.json from
checksum-verified original JSON inputs, and compare them with the study checksums.

Runs the unchanged study code inside a NEW workspace directory; the packaged code
and canonical results are never edited. Requires the pinned environment in
focused_bargaining_20261002/packaging/requirements.txt. No network access.

Usage (after review_support/fetch_craigslist_sources.py):
    python review_support/reconstruct_craigslist_derivatives.py --craigslist-dir craigslist_inputs --output reviewer_craigslist
"""
from pathlib import Path
import argparse, gzip, hashlib, json, shutil, subprocess, sys

ROOT = Path(__file__).resolve().parents[1]
CODE_DIRS = ['multidataset_bargaining', 'bargaining_framework_revision', 'bargaining_strengthening_20260926']
CODE_SUFFIXES = {'.py', '.md'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args, cwd):
    print('RUN', ' '.join(str(a) for a in args), flush=True)
    subprocess.run([sys.executable, *args], cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--craigslist-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    assert not out.exists(), 'Use a NEW output directory.'

    audit = json.loads((ROOT/'multidataset_bargaining/data/processed/craigslist_audit.json').read_text(encoding='utf8'))
    for e in audit['source']:
        assert sha(args.craigslist_dir/e['file']) == e['sha256'], 'Wrong source bytes: ' + e['file']

    # Workspace: study code only, the non-Craigslist processed tables, and the verified raw inputs.
    work = out/'workspace'
    for name in CODE_DIRS:
        for p in (ROOT/name).rglob('*'):
            if p.is_file() and p.suffix in CODE_SUFFIXES and 'results' not in p.relative_to(ROOT/name).parts:
                target = work/name/p.relative_to(ROOT/name)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, target)
    processed = work/'multidataset_bargaining/data/processed'
    processed.mkdir(parents=True, exist_ok=True)
    for p in (ROOT/'multidataset_bargaining/data/processed').iterdir():
        if p.is_file() and not p.name.startswith('craigslist_'):
            shutil.copy2(p, processed/p.name)
    raw = work/'research_paper_bm3il/data/raw/craigslist'
    raw.mkdir(parents=True)
    for e in audit['source']:
        shutil.copy2(args.craigslist_dir/e['file'], raw/e['file'])

    # 1. Source tables (prepare_data.craigslist), 2. structured offer events, 3. history benchmark.
    run(['-c', 'import sys; sys.path.insert(0, "multidataset_bargaining"); import prepare_data; prepare_data.craigslist()'], work)
    run(['-c', 'import sys; sys.path.insert(0, "bargaining_framework_revision"); import data_validation; data_validation.structured_events()'], work)
    run(['bargaining_strengthening_20260926/history_controls.py', '--output',
         str(work/'bargaining_strengthening_20260926/history_results')], work)

    expected = json.loads((ROOT/'review_support/CRAIGSLIST_DERIVATIVE_SHA256.json').read_text(encoding='utf8'))['files']
    predictions_ok = False
    report = []
    for e in expected:
        p = work/e['path']
        if not p.is_file():
            status = 'missing'
        elif sha(p) == e['sha256']:
            status = 'byte_identical'
        elif 'content_sha256' in e and hashlib.sha256(gzip.decompress(p.read_bytes())).hexdigest() == e['content_sha256']:
            status = 'content_identical_gzip_timestamp_differs'
        elif p.suffix == '.joblib':
            status = 'model_bytes_differ_check_predictions'
        else:
            status = 'different'
        if p.name == 'predictions.csv.gz':
            predictions_ok = status in ('byte_identical', 'content_identical_gzip_timestamp_differs')
        report.append(dict(path=e['path'], status=status))
    for r in report:
        if r['status'] == 'model_bytes_differ_check_predictions':
            r['status'] = 'model_predictions_identical' if predictions_ok else 'model_predictions_unverified'
    verified = {'byte_identical', 'content_identical_gzip_timestamp_differs', 'model_predictions_identical'}

    # Compare the rebuilt history metrics with the packaged summary (tolerance 1e-12).
    def numbers(x, prefix=''):
        if isinstance(x, dict):
            for k, v in x.items():
                yield from numbers(v, prefix + '/' + k)
        elif isinstance(x, (int, float)) and not isinstance(x, bool):
            yield prefix, float(x)
    packaged = json.loads((ROOT/'bargaining_strengthening_20260926/history_results/summary.json').read_text(encoding='utf8'))
    rebuilt = json.loads((work/'bargaining_strengthening_20260926/history_results/summary.json').read_text(encoding='utf8'))
    keys = ['metrics', 'paired_contrasts', 'model_selection']
    a = dict(numbers({k: packaged[k] for k in keys}))
    b = dict(numbers({k: rebuilt[k] for k in keys}))
    worst = max(abs(a[k] - b[k]) for k in a) if a.keys() == b.keys() else None
    result = dict(files=report, expected=len(report),
                  byte_identical=sum(r['status'] == 'byte_identical' for r in report),
                  verified=sum(r['status'] in verified for r in report),
                  history_summary_same_keys=a.keys() == b.keys(), history_summary_max_abs_difference=worst,
                  history_summary_within_tolerance=worst is not None and worst <= 1e-12,
                  source_sha256_match=sorted(rebuilt['source_sha256'].values()) == sorted(packaged['source_sha256'].values()),
                  environment=rebuilt.get('environment'))
    (out/'craigslist_reconstruction_report.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    print(json.dumps(result, indent=2))
    ok = result['verified'] == result['expected'] and result['history_summary_within_tolerance'] and result['source_sha256_match']
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
