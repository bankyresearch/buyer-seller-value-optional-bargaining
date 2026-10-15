"""Create a local ZIP from an explicit allowlist; never upload or delete files."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[2]
PREFIX = 'focused_bargaining_20261002'
SEEDS = [271, 811, 1601, 2027, 4093]
REGIMES = ['reference', 'stubborn_offers', 'low_valuations']
COEFFICIENTS = ['0', '0.25', '0.5', '1']


def allowlist():
    files = [
        f'{PREFIX}/packaging/{name}' for name in [
            'package_reproduction.py', 'verify_manifest.py', 'replay.py', 'record_portability.py',
            'requirements.txt', 'README.txt', 'DATA_AND_RIGHTS.txt', 'technical_audit.txt',
            'portability_verification.json']]
    files += [f'{PREFIX}/integration/{name}' for name in [
        'generate_trajectories.py', 'verify_grid.py', 'compare_reproduction.py',
        'test_integration.py', 'protocol.json', 'schema.json']]
    files += [f'{PREFIX}/analysis/{name}' for name in [
        'channel_analysis.py', 'test_channel_analysis.py', 'verify_analysis.py', 'protocol.json', 'schema.json']]
    files += [f'{PREFIX}/figures/{name}' for name in [
        'build_figures.py', 'numeric_plot_data.csv', 'figure_verification.json', 'visual_inspection.json']]
    for number in range(1, 5):
        for extension in ['png', 'eps', 'svg']:
            files.append(f'{PREFIX}/figures/Fig{number}.{extension}')
    files += [f'{PREFIX}/integration/results/{name}' for name in [
        'trajectory_manifest.csv', 'product_negotiation_outcomes.csv',
        'seed_negotiation_outcomes.csv', 'integration_summary.json',
        'grid_verification.json', 'reproduction_verification.json', 'test_verification.json']]
    for regime in REGIMES:
        for name in [f'scenarios_{regime}.csv.gz', f'randoms_{regime}.npz']:
            files += [f'{PREFIX}/integration/results/{name}',
                      f'bargaining_strengthening_20260926/platform_results/{name}']
        for coefficient in COEFFICIENTS:
            for policy in ['time', 'responsive', 'fqi_full', 'fqi_clock']:
                files.append(f'{PREFIX}/integration/results/base_{regime}_a{coefficient}_{policy}.csv.gz')
            for policy in ['time_concession', 'buyer_responsive']:
                files.append(f'bargaining_strengthening_20260926/platform_results/base_{regime}_{coefficient}_{policy}.csv.gz')
    files += [f'{PREFIX}/analysis/results/{name}' for name in [
        'thresholds.csv', 'summary.json', 'ranking_mechanisms.csv',
        'product_coefficients.csv.gz', 'primary_intervals.csv',
        'policy_cost_crossings.csv', 'paired_policy_contrasts.csv',
        'grid_point_estimates.csv', 'finite_condition_summary.csv',
        'descriptive_rankings.csv', 'descriptive_hindsight_loss.csv',
        'composition_thresholds.csv', 'analysis_verification.json', 'accounting_tests.json']]
    files += ['bargaining_framework_revision/experiment.py',
              'bargaining_framework_revision/data_validation.py',
              'bargaining_framework_revision/results/selected_rules.json',
              'bargaining_framework_revision/results/structured_offer_events.csv']
    files += [f'bargaining_strengthening_20260926/{name}' for name in [
        'learning_controls.py', 'platform_controls.py', 'history_controls.py',
        'history_verify.py', 'LEARNING_PROTOCOL.md', 'PLATFORM_PROTOCOL.md', 'HISTORY_PROTOCOL.md']]
    files += [f'bargaining_strengthening_20260926/learning_results/{name}' for name in [
        'frozen_settings.json', 'candidate_scores.csv', 'training_coverage.csv']]
    for specification in ['mixed_enriched_128', 'clock_mixed_enriched_128']:
        for seed in SEEDS:
            files.append(f'bargaining_strengthening_20260926/learning_results/fqi_{specification}_40_{seed}.joblib')
    files.append('bargaining_strengthening_20260926/platform_results/summary.csv')
    files += [f'bargaining_strengthening_20260926/history_results/{name}' for name in [
        'development_selection.csv', 'predictions.csv.gz', 'scenario_errors.csv',
        'verification.json', 'summary.json', 'INTERPRETATION.md']]
    for view in ['Mixed_history', 'Native-only_history', 'No-price-history_context']:
        files.append(f'bargaining_strengthening_20260926/history_results/{view}_features.csv.gz')
        for family in ['Ridge', 'Gradient_boosting']:
            files.append(f'bargaining_strengthening_20260926/history_results/{view}__{family}.joblib')
    files += [f'multidataset_bargaining/{name}' for name in [
        'analyze.py', 'simulate.py', 'prepare_data.py', 'download_pinned.py']]
    files += [f'multidataset_bargaining/data/processed/{name}' for name in [
        'amazon_products.csv', 'craigslist_dialogues.csv', 'craigslist_price_events.csv',
        'casino_dialogues.csv', 'casino_participants.csv',
        'amazon_audit.json', 'craigslist_audit.json', 'casino_audit.json']]
    files += [f'multidataset_bargaining/results/{name}' for name in [
        'casino_associations.json', 'analysis_environment.json', 'descriptive.json',
        'craigslist_category_summary.csv', 'amazon_category_summary.csv']]
    files += ['multidataset_bargaining/data/raw/acquisition.json']
    for source in ['amazon', 'casino']:
        for name in ['LICENSE', 'README.md']:
            files.append(f'multidataset_bargaining/data/raw/{source}/{name}')
    assert len(files) == len(set(files)), 'Duplicate allowlist member'
    return sorted(files)


def package(output):
    output = output.resolve()
    assert output.suffix.lower() == '.zip'
    assert not output.exists(), 'Choose a new ZIP path; archives are not overwritten.'
    entries = []
    for name in allowlist():
        rel = PurePosixPath(name)
        assert not rel.is_absolute() and '..' not in rel.parts
        path = ROOT.joinpath(*rel.parts)
        assert path.resolve().is_relative_to(ROOT) and path.is_file() and not path.is_symlink(), path
        data = path.read_bytes()
        entries.append(dict(path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
    manifest = dict(format_version=1, scope='Explicit local reviewer-preparation package; no publication or upload.',
                    source_date='2026-10-02', files=entries,
                    exclusions=['Raw dialogue utterances', 'Private author documents', 'Duplicate rerun directories',
                                'Vendored runtimes/dependencies', 'Unrelated older policy evaluations'],
                    manifest_excludes_itself=True)
    payload = json.dumps(manifest, indent=2, ensure_ascii=False).encode('utf-8')
    output.parent.mkdir(parents=True, exist_ok=True)
    # Fixed metadata makes packaging deterministic for unchanged allowlisted bytes.
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for entry in entries:
            item = zipfile.ZipInfo(entry['path'], date_time=(2026, 10, 2, 0, 0, 0))
            item.compress_type = zipfile.ZIP_DEFLATED
            data = (ROOT/entry['path']).read_bytes()
            assert hashlib.sha256(data).hexdigest() == entry['sha256'], 'Source changed while packaging'
            archive.writestr(item, data)
        item = zipfile.ZipInfo('SHA256SUMS.json', date_time=(2026, 10, 2, 0, 0, 0))
        item.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(item, payload)
    manifest_path = output.with_suffix('.manifest.json')
    assert not manifest_path.exists()
    manifest_path.write_bytes(payload)
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        assert sorted(archive.namelist()) == sorted([x['path'] for x in entries]+['SHA256SUMS.json'])
        for entry in entries:
            assert hashlib.sha256(archive.read(entry['path'])).hexdigest() == entry['sha256']
    report = dict(zip=output.name, archive_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                  listed_files=len(entries), uncompressed_bytes=sum(x['bytes'] for x in entries),
                  compressed_bytes=output.stat().st_size, sha256_verified=True)
    output.with_suffix('.verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent/'deliverables/Online_Resource_1.zip')
    package(parser.parse_args().output)
