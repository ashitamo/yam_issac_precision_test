"""0.10 mm small-sample convergence check. Never schedules the full grid."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from test_physics import PROFILES, source_signature, write_json
from env_task import contact_gate_pass, geometry_diagnostic_thresholds


def reassess(source, destination):
    """Apply today's acceptance tolerance to existing measurements; no simulation."""
    source = source.resolve()
    source_path = source / 'subset_summary.json'
    original = json.loads(source_path.read_text(encoding='utf-8'))
    target = destination.resolve() / 'reassessment.json'
    if target.exists():
        raise FileExistsError('Choose a fresh reassessment output directory')
    thresholds = geometry_diagnostic_thresholds()
    rows = []
    for profile, summary in original['profiles'].items():
        for case in summary['results']:
            path = source / profile / case['result_file'].replace('\\', '/')
            measured = json.loads(path.read_text(encoding='utf-8'))
            passed = contact_gate_pass(measured['max_insertion_radial_violation_m'],
                measured['max_insertion_bottom_penetration_m'], thresholds)
            rows.append(dict(profile=profile, env_id=case['env_id'], grid_id=case['grid_id'],
                task_success=case['task_success'], original_contact_gate_pass=case['contact_gate_pass'],
                reassessed_contact_gate_pass=passed,
                radial_violation_m=measured['max_insertion_radial_violation_m'],
                bottom_penetration_m=measured['max_insertion_bottom_penetration_m'],
                source_file=str(path), source_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    report = dict(assessment_only=True, simulation_rerun=False,
        original_summary=str(source_path), original_summary_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        original_source_signature=original['source_signature'], original_all_passed=original['all_passed'],
        geometry_diagnostic_thresholds=thresholds, cases=rows,
        unchanged_convergence_comparisons=original['comparisons'],
        all_passed=bool(rows) and all(row['task_success'] and row['reassessed_contact_gate_pass'] for row in rows)
            and all(row['passed'] for row in original['comparisons']),
        note='Acceptance criteria changed by user. Original measurements/reports are unchanged; not a new dynamics validation or full-grid certification.')
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json(target, report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def signature():
    digest = hashlib.sha256(source_signature().encode())
    for name in ('env_yam.py', 'env_expert.py', 'test_parallel_expert.py', 'test_precision_subset.py'):
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grid_ids', nargs='+', default=['r08_c08', 'r16_c16'])
    parser.add_argument('--reassess_from', type=Path, help='Reassess existing measurements only; never launches Isaac.')
    parser.add_argument('--output_dir', type=Path, default=Path(__file__).parent / 'validation' / 'precision_c0p10_subset')
    args = parser.parse_args()
    if args.reassess_from is not None:
        reassess(args.reassess_from, args.output_dir)
        return
    if not 1 <= len(args.grid_ids) <= 4 or len(set(args.grid_ids)) != len(args.grid_ids):
        parser.error('Choose 1..4 unique grid IDs; full-grid execution is deliberately unavailable')
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'subset_summary.json').exists():
        parser.error('Choose a fresh output directory')
    initial_signature = signature()
    report = dict(state='running', grid_ids=args.grid_ids, clearance_mm=.1,
        expert_profile='fast', source_signature=initial_signature, profiles={},
        all_passed=False, note='Small-sample numerical check only, not 289-point certification')
    write_json(root / 'subset_summary.json', report)
    for profile in PROFILES:
        if signature() != initial_signature:
            raise RuntimeError('Validation source changed; refusing to mix versions')
        command = [sys.executable, '-u', str(Path(__file__).with_name('test_parallel_expert.py')),
            '--num_envs', str(len(args.grid_ids)), '--grid_ids', *args.grid_ids,
            '--clearance_mm', '.1', '--expert_profile', 'fast', '--physics_profile', profile,
            '--output_dir', str(root / profile)]
        print('[PRECISION]', profile, args.grid_ids, flush=True)
        with (root / (profile + '.log')).open('w', encoding='utf-8') as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            start = time.perf_counter()
            try:
                while process.poll() is None:
                    if (root / 'STOP_REQUESTED').exists() or time.perf_counter() - start > 900:
                        raise InterruptedError('Stop requested or 900-second case timeout')
                    time.sleep(1)
            except (KeyboardInterrupt, InterruptedError):
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                report['state'] = 'stopped'
                write_json(root / 'subset_summary.json', report)
                return
        if signature() != initial_signature:
            raise RuntimeError('Validation source changed during execution')
        summary_path = root / profile / 'summary.json'
        if not summary_path.exists():
            report.update(state='error', reason=f'{profile}: missing summary; inspect log')
            write_json(root / 'subset_summary.json', report)
            return
        report['profiles'][profile] = json.loads(summary_path.read_text(encoding='utf-8'))
        write_json(root / 'subset_summary.json', report)
    comparisons = []
    for env_id, grid_id in enumerate(args.grid_ids):
        results = [json.loads((root / profile / f'env_{env_id:03d}' / 'pick_insert_result.json').read_text(encoding='utf-8'))
                   for profile in PROFILES]
        depths = [result['final_insertion_depth_m'] for result in results]
        errors = [result['preinsertion_xy_error_m'] for result in results]
        valid = all(value is not None for value in errors)
        depth_spread = max(depths) - min(depths)
        xy_spread = max(errors) - min(errors) if valid else None
        comparisons.append(dict(grid_id=grid_id, depth_spread_m=depth_spread, preinsertion_xy_spread_m=xy_spread,
            passed=valid and depth_spread <= .0001 and xy_spread <= .00005))
    report.update(state='completed', comparisons=comparisons,
        all_passed=all(item['all_passed'] for item in report['profiles'].values())
            and all(item['passed'] for item in comparisons))
    write_json(root / 'subset_summary.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
