"""Retrieve the exact CraigslistBargain JSON inputs from their original CodaLab
endpoints and accept them only if every SHA-256 checksum matches the study record.

Standard library only. Nothing is redistributed by this script: each reviewer
downloads the files directly from the original maintainers' server.

Integrity rests on the SHA-256 comparison, not on the transport. If the server's
TLS certificate cannot be verified (it expired on 7 October 2026), the script
stops unless --allow-unverified-tls is given. With that flag the bytes are still
rejected unless they match the recorded checksums exactly.

Usage:
    python review_support/fetch_craigslist_sources.py --output craigslist_inputs
    python review_support/fetch_craigslist_sources.py --output craigslist_inputs --allow-unverified-tls
Then:
    python review_support/reconstruct_sources.py --output reviewer_all_sources --craigslist-dir craigslist_inputs
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, ssl, sys, urllib.error, urllib.request

SOURCES = [
    {'file': 'train.json',
     'url': 'https://worksheets.codalab.org/rest/bundles/0xd34bbbc5fb3b4fccbd19e10756ca8dd7/contents/blob/parsed.json',
     'sha256': '34033ff87565b9fc9eb0efe867e9d3e32456dbe1528cd1683f94a84b09f66ace', 'bytes': 20148723},
    {'file': 'dev.json',
     'url': 'https://worksheets.codalab.org/rest/bundles/0x15c4160b43d44ee3a8386cca98da138c/contents/blob/parsed.json',
     'sha256': '03b35dc18bd90d87dac46893ac4db8ab3eed51786d192975be68d3bab38e306e', 'bytes': 2287054},
    {'file': 'test.json',
     'url': 'https://worksheets.codalab.org/rest/bundles/0x54d325bbcfb2463583995725ed8ca42b/contents/blob/',
     'sha256': 'c802f15f80ea3066d429375393319d7234daacbd6a26a6ad5afd0ad78a2f7736', 'bytes': 2937841},
]


def check_against_register():
    """Confirm the embedded checksums equal those in SOURCE_REGISTER.json, if present."""
    register = Path(__file__).resolve().parent / 'SOURCE_REGISTER.json'
    if not register.is_file():
        return
    recorded = {e['filename']: e['sha256'] for e in json.loads(register.read_text(encoding='utf8'))
                if e.get('dataset') == 'craigslist'}
    for s in SOURCES:
        assert recorded.get(s['file']) == s['sha256'], 'Embedded checksum differs from SOURCE_REGISTER.json: ' + s['file']


def is_certificate_error(exc):
    reason = getattr(exc, 'reason', exc)
    return isinstance(reason, ssl.SSLCertVerificationError)


def download(url, target, context):
    digest = hashlib.sha256()
    size = 0
    request = urllib.request.Request(url, headers={'User-Agent': 'hssc-review-fetch/1.0'})
    with urllib.request.urlopen(request, timeout=300, context=context) as response, open(target, 'wb') as out:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
            out.write(chunk)
    return digest.hexdigest(), size


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--output', type=Path, required=True, help='New or empty directory for the three JSON files')
    parser.add_argument('--allow-unverified-tls', action='store_true',
                        help='Retry without TLS certificate verification; files are still checksum-verified')
    args = parser.parse_args()

    check_against_register()
    out = args.output.resolve()
    assert not out.exists() or not any(out.iterdir()), 'Use a NEW or empty output directory.'
    out.mkdir(parents=True, exist_ok=True)

    log = {'retrieved_utc': datetime.now(timezone.utc).isoformat(), 'python': sys.version.split()[0], 'files': []}
    failures = []
    for s in SOURCES:
        target = out / s['file']
        part = target.with_suffix('.part')
        record = {'file': s['file'], 'url': s['url'], 'expected_sha256': s['sha256']}
        try:
            try:
                sha, size = download(s['url'], part, ssl.create_default_context())
                record['tls_certificate_verified'] = True
            except (urllib.error.URLError, ssl.SSLError) as exc:
                if not is_certificate_error(exc):
                    raise
                record['tls_certificate_error'] = str(getattr(exc, 'reason', exc))
                if not args.allow_unverified_tls:
                    part.unlink(missing_ok=True)
                    print('TLS certificate verification failed for', s['url'])
                    print('  ', record['tls_certificate_error'])
                    print('Re-run with --allow-unverified-tls to download and accept the file only if its SHA-256 matches.')
                    raise SystemExit(2)
                sha, size = download(s['url'], part, ssl._create_unverified_context())
                record['tls_certificate_verified'] = False
        except SystemExit:
            raise
        except Exception as exc:
            part.unlink(missing_ok=True)
            record['error'] = repr(exc)
            failures.append(s['file'])
            log['files'].append(record)
            continue
        record.update({'downloaded_sha256': sha, 'downloaded_bytes': size, 'matches_study_record': sha == s['sha256']})
        if sha == s['sha256'] and size == s['bytes']:
            part.replace(target)
        else:
            part.unlink(missing_ok=True)
            failures.append(s['file'])
        log['files'].append(record)

    log['all_files_match'] = not failures
    (out / 'retrieval_log.json').write_text(json.dumps(log, indent=2), encoding='utf8')
    print(json.dumps(log, indent=2))
    raise SystemExit(1 if failures else 0)


if __name__ == '__main__':
    main()
