"""Sequential full-task benchmark: baseline4, fast4, fast8, fast16. No cameras."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


GRID_IDS = ['r08_c08', 'r00_c00', 'r00_c16', 'r16_c16']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir', type=Path, default=Path(__file__).parent / 'validation' / 'throughput')
    parser.add_argument('--num_envs', type=int, nargs='+', default=[4, 8, 16])
    parser.add_argument('--include_baseline', action='store_true')
    args = parser.parse_args()
    if any(count < 4 or count % 4 for count in args.num_envs):
        parser.error('Environment counts must be positive multiples of four')
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / 'benchmark.json'
    if report_path.exists():
        parser.error('Choose a fresh output directory; previous benchmark will not be overwritten')
    cases = ([('baseline', 4)] if args.include_baseline else []) + [('fast', n) for n in args.num_envs]
    report = dict(state='running', results=[], case_family=GRID_IDS,
        note='4-env baseline/fast have identical cases; 8/16 repeat the same four-case distribution, not the same total episode count.')
    def save():
        temporary = report_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2), encoding='utf-8')
        temporary.replace(report_path)
    save()
    for profile, count in cases:
        case_id = f'{profile}_{count}'
        folder = root / case_id
        report['active_case'] = case_id
        save()
        command = [sys.executable, '-u', str(Path(__file__).with_name('test_parallel_expert.py')),
            '--num_envs', str(count), '--expert_profile', profile,
            '--grid_ids', *(GRID_IDS * (count // 4)), '--output_dir', str(folder)]
        print(f'[BENCHMARK] {case_id}', flush=True)
        start = time.perf_counter()
        with (root / f'{case_id}.log').open('w', encoding='utf-8') as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            try:
                while process.poll() is None:
                    if (root / 'STOP_REQUESTED').exists():
                        raise InterruptedError('Benchmark stop requested')
                    if time.perf_counter() - start > 900:
                        raise TimeoutError('Case exceeded 900 wall seconds')
                    time.sleep(1)
            except (KeyboardInterrupt, InterruptedError, TimeoutError) as error:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                report.update(state='stopped', reason=str(error))
                save()
                return
        summary_path = folder / 'summary.json'
        if not summary_path.exists():
            report.update(state='failed', reason=f'{case_id}: missing summary; inspect its log')
            save()
            return
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        passed = sum(bool(row['task_success'] and row['contact_gate_pass']) for row in summary['results'])
        elapsed = summary['wall_seconds']
        row = dict(case_id=case_id, num_envs=count, passed=passed, attempted=count,
            all_passed=summary['all_passed'], loop_wall_seconds=elapsed,
            total_process_wall_seconds=time.perf_counter() - start,
            successful_rollouts_per_second=passed / elapsed,
            mean_wall_seconds_per_successful_rollout=elapsed / passed if passed else None,
            summary_file=str(summary_path.relative_to(root)))
        report['results'].append(row)
        save()
        print(json.dumps(row), flush=True)
        if not row['all_passed']:
            report.update(state='failed', reason=f'{case_id}: task/contact regression; larger batches not started')
            save()
            return
    report.update(state='completed', active_case=None)
    save()


if __name__ == '__main__':
    main()
