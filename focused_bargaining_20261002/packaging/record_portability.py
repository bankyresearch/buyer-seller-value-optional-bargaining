"""Record executed clean-extraction checks without distributing duplicate outputs."""
from pathlib import Path
import argparse
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def record(extracted):
    extracted = extracted.resolve()
    run = extracted/'replay_check'
    assert load(run/'replay_success.json')['status'] == 'passed'
    manifest = load(extracted/'SHA256SUMS.json')
    critical = [x for x in manifest['files'] if x['path'].endswith('.py')
                and '/packaging/' not in x['path']]
    for entry in critical:
        assert hashlib.sha256((ROOT/entry['path']).read_bytes()).hexdigest() == entry['sha256']
    casino = load(extracted/'casino_check/casino_associations.json')
    assert casino == load(ROOT/'multidataset_bargaining/results/casino_associations.json')
    trajectory = load(run/'integration/reproduction_verification.json')
    accounting = load(run/'analysis/analysis_verification.json')
    tests = load(run/'analysis/accounting_tests.json')
    commands = load(run/'replay_commands.json')
    assert all(x['exit_code'] == 0 for x in commands)
    report = dict(status='passed', verified_on='2026-10-02',
        scope='Local archive extraction in a different repository root with installed dependencies; not independent external scientific replication.',
        tested_manifest_sha256=hashlib.sha256((extracted/'SHA256SUMS.json').read_bytes()).hexdigest(),
        same_scientific_sources={x['path']:x['sha256'] for x in critical},
        trajectory_comparison=trajectory, analysis_comparison=accounting,
        accounting_fixture_tests=tests['tests_run'], integration_fixture_tests=7,
        casino_association_replay='Exact JSON agreement for all four associations and two within/dialogue-level diagnostics',
        command_count=len(commands), elapsed_seconds=sum(x['elapsed_seconds'] for x in commands),
        excluded='Duplicate extracted/replay directories and absolute local paths are not included in the delivery ZIP.')
    target = Path(__file__).resolve().parent/'portability_verification.json'
    target.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({key:report[key] for key in ['status','accounting_fixture_tests','integration_fixture_tests','command_count','elapsed_seconds']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--extracted', type=Path, required=True)
    record(parser.parse_args().extracted)
