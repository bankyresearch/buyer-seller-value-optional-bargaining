"""Offline numerical replay into a fresh directory; never select/refit policies."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
INTEGRATION = Path('focused_bargaining_20261002/integration')
ANALYSIS = Path('focused_bargaining_20261002/analysis')


def run(output, casino_only=False):
    output = output.resolve()
    assert not output.exists(), 'Use a new output directory; existing artifacts are not overwritten.'
    output.mkdir(parents=True)
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = '2'
    env['OPENBLAS_NUM_THREADS'] = '2'
    env['MKL_NUM_THREADS'] = '2'
    logs = []

    def command(args, cwd=ROOT):
        number = len(logs)+1
        target = output/f'{number:02d}_command.log'
        begin = time.perf_counter()
        print('RUN', ' '.join(map(str, args)), flush=True)
        with target.open('w', encoding='utf-8') as stream:
            result = subprocess.run([sys.executable, *map(str, args)], cwd=cwd,
                                    env=env, stdout=stream, stderr=subprocess.STDOUT)
        logs.append(dict(arguments=list(map(str, args)), exit_code=result.returncode,
                         elapsed_seconds=time.perf_counter()-begin, log=target.name))
        (output/'replay_commands.json').write_text(json.dumps(logs, indent=2), encoding='utf-8')
        if result.returncode:
            raise RuntimeError(f'Command failed; inspect {target}')

    if casino_only:
        sys.path.insert(0, str(ROOT/'multidataset_bargaining'))
        import analyze
        analyze.OUT = output
        analyze.casino_relations()
        return
    if (ROOT/'SHA256SUMS.json').exists():
        command(['focused_bargaining_20261002/packaging/verify_manifest.py'])
    # unittest imports the module without invoking its canonical-report writer.
    command(['-m', 'unittest', 'test_integration', '-v'], ROOT/INTEGRATION)
    generated = output/'integration'
    command([INTEGRATION/'generate_trajectories.py', '--output', generated])
    command([INTEGRATION/'verify_grid.py', '--output', generated])
    command([INTEGRATION/'compare_reproduction.py', '--original', generated,
             '--repeat', ROOT/INTEGRATION/'results'])
    analysis = output/'analysis'
    command([ANALYSIS/'test_channel_analysis.py', '--output', analysis])
    # Canonical input gzip hashes are provenance; regenerated numerical values
    # have separately been compared, so retain the same input hashes here.
    command([ANALYSIS/'channel_analysis.py', '--inputs', ROOT/INTEGRATION/'results',
             '--output', analysis])
    command([ANALYSIS/'verify_analysis.py', '--current', analysis,
             '--replay', ROOT/ANALYSIS/'results'])
    (output/'replay_success.json').write_text(json.dumps(dict(status='passed',
        trajectory_directory=str(generated), analysis_directory=str(analysis),
        scope='Offline frozen-policy replay and conditional accounting, not external scientific validation.'),
        indent=2), encoding='utf-8')
    print('REPLAY PASSED', output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--casino-only', action='store_true')
    args = parser.parse_args()
    run(args.output, args.casino_only)
