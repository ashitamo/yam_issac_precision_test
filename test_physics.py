"""Sequential isolated PhysX convergence tests. No training data are generated."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import os
from collections import Counter
from html import escape
import env_config as cfg
from env_task import contact_gate_pass, geometry_diagnostic_thresholds

PROFILES = {
    'baseline': dict(hz=240, peg_position=64, peg_velocity=16, scene_position=64, scene_velocity=16),
    'half_dt': dict(hz=480, peg_position=64, peg_velocity=16, scene_position=64, scene_velocity=16),
    'strong_solver': dict(hz=240, peg_position=128, peg_velocity=32, scene_position=128, scene_velocity=32),
}


def source_signature():
    """Fingerprint exactly the local code/assets used by a case."""
    names = ['env_config.py', 'env_expert.py', 'env_task.py', 'robot_yam.py', 'run_environment.py',
             'env_workspace_grid.py', 'env_yam.py', 'env_dataset.py', 'test_physics.py',
             'assets/workpiece.stl', 'assets/yam_robot_with_gripper/config/cumotion/yam.urdf']
    assets = [p for p in (cfg.PROJECT_ROOT / 'assets').rglob('*')
              if p.suffix.lower() in {'.usd', '.usda', '.usdc', '.stl', '.xrdf', '.yaml', '.urdf'}]
    digest = hashlib.sha256()
    for path in sorted({cfg.PROJECT_ROOT / name for name in names} | set(assets)):
        digest.update(str(path.relative_to(cfg.PROJECT_ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def write_json(path, data):
    """Readers see either the previous complete report or the next one."""
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    for attempt in range(10):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.05)


def read_json(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def failure_category(reason, state, task_success, contact_ok):
    if state not in {'completed', 'error', 'interrupted', 'invalid_source'}:
        return None
    if state != 'completed':
        return state
    text = (reason or '').lower()
    if task_success:
        return None if contact_ok else 'contact_diagnostic'
    if 'ik failed' in text:
        return 'ik_failure'
    if 'slip' in text or 'dropped' in text or 'not lifted' in text:
        return 'grasp_or_slip'
    if 'timeout' in text:
        return 'tracking_timeout'
    if 'bottom' in text or 'depth' in text or 'geometry' in text:
        return 'insertion_or_release_validation'
    return 'task_failure'


def run_case(path):
    case = json.loads(Path(path).read_text(encoding='utf-8'))
    profile = case.get('physics_profile', PROFILES[case['profile']])
    if 'hole_position_m' in case:
        cfg.WORKPIECE_POSITION_M = tuple(case['hole_position_m'])
    cfg.PHYSICS_DT_S = 1.0 / profile['hz']
    cfg.PEG_POSITION_ITERATIONS = profile['peg_position']
    cfg.PEG_VELOCITY_ITERATIONS = profile['peg_velocity']
    cfg.SCENE_MIN_POSITION_ITERATIONS = profile['scene_position']
    cfg.SCENE_MIN_VELOCITY_ITERATIONS = profile['scene_velocity']
    # Control rate, speed, gains, material and command depth are unchanged.
    # This is a physics validation run, not an RGB collection run.
    cfg.RENDER_EVERY_N_STEPS = 100000
    cfg.CAMERA_PERIOD_S = 100000.0
    sys.argv = ['run_environment.py', '--viz', 'none', '--expert',
                '--clearance_mm', str(case['clearance_mm']),
                '--output_dir', str(Path(path).parent), '--max_steps', str(profile['hz'] * 180)]
    if 'hole_position_m' not in case:
        sys.argv += ['--grid_id', case['grid_id']]
    import run_environment
    try:
        run_environment.main()
    finally:
        run_environment.app.close()


def summarize(root, cases):
    records = []
    for case in cases:
        result_path = root / case['case_id'] / 'pick_insert_result.json'
        case_path = result_path.parent / 'case.json'
        matching_case = read_json(case_path) == case
        result = read_json(result_path) if matching_case else {}
        execution = read_json(result_path.parent / 'execution.json') if matching_case else {}
        completed = result.get('status') == 'completed'
        state = 'completed' if completed else execution.get('state', result.get('status', 'not_started'))
        if state == 'starting':
            state = 'running'
        # Preserve historical results that predate explicit threshold metadata.
        radial = result.get('max_insertion_radial_violation_m')
        bottom = result.get('max_insertion_bottom_penetration_m')
        thresholds = result.get('geometry_diagnostic_thresholds', dict(radial_m=.00002, bottom_m=.00002))
        contact_ok = contact_gate_pass(radial, bottom, thresholds)
        reason = result.get('reason') or execution.get('reason')
        success = result.get('task_success') if completed else None
        records.append(dict(**case, state=state, completed=completed, task_success=success,
                            contact_diagnostic_pass=contact_ok,
                            geometry_diagnostic_thresholds=thresholds,
                            final_depth_m=result.get('final_insertion_depth_m'),
                            preinsertion_xy_error_m=result.get('preinsertion_xy_error_m'),
                            max_radial_violation_m=radial, max_bottom_penetration_m=bottom,
                            failure_stage=result.get('failure_stage'), reason=reason,
                            failure_category=failure_category(reason, state, success, contact_ok)))
    comparisons = []
    for clearance in sorted({c['clearance_mm'] for c in cases}):
        for grid_id in sorted({c['grid_id'] for c in cases}):
            group = [r for r in records if r['clearance_mm'] == clearance and r['grid_id'] == grid_id]
            complete_profiles = {r['profile'] for r in group} == set(PROFILES)
            depths = [r['final_depth_m'] for r in group if r['final_depth_m'] is not None]
            xy = [r['preinsertion_xy_error_m'] for r in group if r['preinsertion_xy_error_m'] is not None]
            depth_spread = max(depths) - min(depths) if depths else None
            xy_spread = max(xy) - min(xy) if xy else None
            passed = (complete_profiles and all(r['completed'] and r['task_success'] and r['contact_diagnostic_pass'] for r in group)
                      and depth_spread is not None and depth_spread <= 0.0001
                      and xy_spread is not None and xy_spread <= 0.00005)
            comparisons.append(dict(clearance_mm=clearance, grid_id=grid_id, profiles_complete=complete_profiles and all(r['completed'] for r in group),
                                    comparison_status='requested' if complete_profiles else 'not_requested',
                                    depth_spread_m=depth_spread, preinsertion_xy_spread_m=xy_spread,
                                    convergence_pass=bool(passed) if complete_profiles else None))
    finished = sum(r['state'] in {'completed', 'error', 'interrupted', 'invalid_source'} for r in records)
    failures = [r for r in records if r['failure_category']]
    requested_comparisons = [c for c in comparisons if c['comparison_status'] == 'requested']
    report = dict(scope='Deterministic expert physics diagnostics at listed positions only; not a statistical success-rate estimate',
                  cases=records, comparisons=comparisons,
                  progress=dict(total=len(cases), finished=finished, completed=sum(r['completed'] for r in records),
                                task_passed=sum(r['task_success'] is True for r in records),
                                fully_passed=sum(r['completed'] and r['task_success'] and r['contact_diagnostic_pass'] for r in records),
                                remaining=len(cases)-finished),
                  failure_counts=dict(Counter(r['failure_category'] for r in failures)),
                  all_requested_cases_pass=bool(records) and all(r['completed'] and r['task_success'] and r['contact_diagnostic_pass'] for r in records),
                  all_profile_comparisons_pass=all(c['convergence_pass'] for c in requested_comparisons) if requested_comparisons else None,
                  thresholds=dict(current_defaults=geometry_diagnostic_thresholds(),
                    legacy_unrecorded_radial_and_bottom_m=.00002,
                    note='Each case uses its recorded diagnostic thresholds, not the current default.',
                    depth_spread_m=0.0001, preinsertion_xy_spread_m=0.00005))
    write_json(root / 'physics_summary.json', report)
    lines = ['# Physics validation progress', '', f'Finished {finished}/{len(cases)}. Task passes {report["progress"]["task_passed"]}.',
             '', 'Categories describe observed failure checks, not proven contact root causes.', '',
             '| Grid | Clearance (mm) | Profile | Category | Stage | Reported reason |', '| --- | --- | --- | --- | --- | --- |']
    for r in failures:
        reason_text = str(r['reason'] or '').replace('|', '/').replace('\n', ' ')
        lines.append(f'| {r["grid_id"]} | {r["clearance_mm"]} | {r["profile"]} | {r["failure_category"]} | {r["failure_stage"]} | {reason_text} |')
    (root / 'physics_summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    baseline = [r for r in records if r['profile'] == 'baseline']
    if baseline:
        dots = []
        for r in baseline:
            row, col = int(r['grid_id'][1:3]), int(r['grid_id'][5:7])
            color = '#aaa' if r['state'] == 'not_started' else '#2979b9'
            if r['failure_category']:
                color = '#d33'
            elif r['completed'] and r['task_success'] and r['contact_diagnostic_pass']:
                color = '#229653'
            dots.append(f'<circle cx="{30+col*25}" cy="{430-row*25}" r="7" fill="{color}"><title>{escape(r["grid_id"]+": "+r["state"]+" "+str(r["reason"] or ""))}</title></circle>')
        (root / 'task_grid.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg" width="480" height="470"><rect width="100%" height="100%" fill="white"/><text x="15" y="18">Green pass / red fail / blue running / gray pending</text>'+''.join(dots)+'</svg>', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_case', type=Path)
    parser.add_argument('--output_dir', type=Path, default=cfg.PROJECT_ROOT / 'validation/physics')
    parser.add_argument('--grid_ids', nargs='+', default=['r08_c08'])
    parser.add_argument('--all_grid', action='store_true', help='Run all 289 task positions, potentially many hours.')
    parser.add_argument('--clearances', nargs='+', type=float, choices=[1., .5, .25, .1], default=[.5, .1])
    parser.add_argument('--profiles', nargs='+', choices=list(PROFILES), default=list(PROFILES))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop_file', type=Path, help='Stop cooperatively when this file exists.')
    parser.add_argument('--expected_signature', help='Reject a changed source/assets version.')
    args = parser.parse_args()
    if args.run_case:
        run_case(args.run_case)
        return
    if args.all_grid:
        from env_workspace_grid import make_manifest
        args.grid_ids = [p['grid_id'] for p in make_manifest()['points']]
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    signature = source_signature()
    if args.expected_signature and signature != args.expected_signature:
        raise RuntimeError('Source/assets changed since campaign launch')
    from env_workspace_grid import get_grid_point
    cases = [dict(case_id=f'c{c:.2f}_{g}_{p}', clearance_mm=c, grid_id=g, profile=p, source_signature=signature,
                  hole_position_m=get_grid_point(g)['hole_position_m'], physics_profile=PROFILES[p])
             for c in args.clearances for g in args.grid_ids for p in args.profiles]
    write_json(root / 'suite_manifest.json', cases)
    summarize(root, cases)
    infrastructure_failures = 0
    for case in cases:
        if args.stop_file and args.stop_file.exists():
            raise SystemExit(130)
        if source_signature() != signature:
            raise RuntimeError('Source/assets changed; stopping instead of mixing versions')
        folder = root / case['case_id']
        folder.mkdir(exist_ok=True)
        case_path = folder / 'case.json'
        result_path = folder / 'pick_insert_result.json'
        if args.resume and case_path.exists() and result_path.exists():
            old = json.loads(case_path.read_text())
            result = json.loads(result_path.read_text())
            if old == case and result.get('status') == 'completed':
                print('[RESUME]', case['case_id'], flush=True)
                continue
        write_json(case_path, case)
        # Invalidate old results BEFORE spawning, including startup failures.
        write_json(result_path, dict(status='starting', task_success=None))
        execution = dict(state='running', started_unix_s=time.time())
        write_json(folder / 'execution.json', execution)
        summarize(root, cases)
        print('[PHYSICS START]', case['case_id'], flush=True)
        start = time.perf_counter()
        with (folder / 'simulation.log').open('w', encoding='utf-8') as log:
            try:
                process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), '--run_case', str(case_path)],
                                           cwd=cfg.PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT)
                completed_at = None
                cleanup_performed = False
                stopped = False
                while process.poll() is None:
                    if args.stop_file and args.stop_file.exists():
                        stopped = True
                        process.terminate()
                        process.wait(timeout=10)
                        break
                    completed = False
                    if result_path.exists():
                        try:
                            completed = json.loads(result_path.read_text()).get('status') == 'completed'
                        except json.JSONDecodeError:
                            pass
                    if completed:
                        completed_at = completed_at or time.perf_counter()
                    else:
                        completed_at = None
                    if completed_at is not None and time.perf_counter() - completed_at > 10:
                        # Some Kit builds hang during shutdown after [EXIT].
                        # Only terminate our own child, after its result is complete.
                        process.terminate()
                        process.wait(timeout=10)
                        cleanup_performed = True
                        break
                    if time.perf_counter() - start > 420:
                        process.kill()
                        process.wait(timeout=10)
                        break
                    time.sleep(0.5)
                code = 'completed_cleanup' if cleanup_performed else process.returncode
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                code = 'timeout'
        result = read_json(result_path)
        execution.update(exit_code=code, wall_s=time.perf_counter()-start, finished_unix_s=time.time())
        if source_signature() != signature:
            execution.update(state='invalid_source', reason='Source/assets changed during this case')
            write_json(result_path, dict(status='invalid_source', task_success=None, reason=execution['reason']))
        elif stopped:
            execution.update(state='interrupted', reason='Campaign stop requested')
        elif result.get('status') == 'completed':
            execution.update(state='completed')
        else:
            execution.update(state='error', reason=f'No completed simulation result, process exit {code}; see simulation.log')
        write_json(folder / 'execution.json', execution)
        print('[PHYSICS END]', case['case_id'], 'code', code, 'wall_s', round(time.perf_counter() - start, 1), flush=True)
        summarize(root, cases)
        if execution['state'] == 'invalid_source':
            raise RuntimeError(execution['reason'])
        if stopped:
            raise SystemExit(130)
        infrastructure_failures = infrastructure_failures + 1 if execution['state'] == 'error' else 0
        if infrastructure_failures >= 3:
            raise RuntimeError('Three consecutive infrastructure failures; inspect logs before resuming')
    report = summarize(root, cases)
    print('[PHYSICS SUMMARY]', json.dumps(report['comparisons']), flush=True)


if __name__ == '__main__':
    main()
