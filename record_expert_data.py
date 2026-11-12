"""Parallel Cartesian expert RGB-D collection. No implicit train/test split."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import traceback
from contextlib import ExitStack


def jsonable(value):
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if hasattr(value, 'detach'):
        return value.detach().cpu().tolist()
    if hasattr(value, 'tolist'):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(jsonable(value), indent=2), encoding='utf-8')
    temporary.replace(path)


def audit_episode(path):
    """Audit every timestamp and save small previews; never infer success from filename."""
    import h5py
    import numpy as np
    from PIL import Image
    with h5py.File(path, 'r') as file:
        t = file['time/observation_s'][:]
        count = len(file['actions/cartesian'])
        if not file.attrs['complete'] or len(t) != count + 1 or not np.all(np.diff(t) > 0):
            raise ValueError('Incomplete episode or invalid T/T+1 chronology')
        np.testing.assert_array_equal(t, file['time/camera_s'][:])
        if not np.all(np.diff(file['time/camera_frame'][:]) > 0):
            raise ValueError('Stale camera frames')
        for key in ('joint_position', 'joint_velocity', 'gripper_opening', 'ee_pose_base'):
            values = file['obs/' + key][:]
            if len(values) != count + 1 or not np.isfinite(values).all():
                raise ValueError('Invalid observation: ' + key)
        if not bool(file['terminated'][-1] or file['truncated'][-1]):
            raise ValueError('Missing terminal transition')
        preview = path.parent / path.stem
        preview.mkdir()
        for name, index in (('initial', 0), ('middle', count // 2), ('terminal', count)):
            Image.fromarray(file['obs/images/rgb'][index]).save(preview / ('rgb_' + name + '.png'))
        errors = np.linalg.norm(file['commands/ee_target_base'][:, :3] - file['obs/ee_pose_base'][1:, :3], axis=1)
        return dict(transitions=count, observations=len(t),
            duration_s=float(t[-1] - t[0]), structure_valid=True,
            tcp_target_next_position_rmse_m=float(np.sqrt(np.mean(errors**2))),
            tcp_target_next_position_p95_m=float(np.percentile(errors, 95)),
            episode_success=bool(file.attrs['episode_success']))


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_dir', type=Path, default=Path(__file__).parent / 'data' / 'expert_data')
    parser.add_argument('--num_envs', type=int, default=4)
    parser.add_argument('--grid_ids', nargs='+', default=['r08_c08'])
    parser.add_argument('--episodes_per_env', type=int, default=1)
    parser.add_argument('--clearance_mm', type=float, choices=[1., .5, .25, .1], default=.5)
    parser.add_argument('--servo_hz', type=int, choices=[60, 120, 240], default=240)
    parser.add_argument('--solver_profile', choices=['baseline', 'low_iter_candidate'], default='baseline')
    parser.add_argument('--replay', action='store_true', help='Extra saved-action replay for validation; not required for every collection.')
    parser.add_argument('--reference_reads', action='store_true', help='Use legacy per-row pose/CPU reads for paired performance comparison.')
    parser.add_argument('--profile', action='store_true', help='Time inclusive SimulationContext.step calls; no separate simulate/fetch attribution.')
    parser.add_argument('--max_wall_seconds', type=float, default=3600.)
    parser.add_argument('--compression', choices=['gzip', 'lzf'], default='lzf')
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs < 1 or args.episodes_per_env < 1 or len(args.grid_ids) not in (1, args.num_envs):
        parser.error('Provide positive counts and one grid ID or exactly num_envs IDs')
    if args.max_wall_seconds <= 0:
        parser.error('max_wall_seconds must be positive')
    args.data_dir = args.data_dir.resolve()
    if args.data_dir.exists() and any(args.data_dir.iterdir()):
        parser.error('Choose a fresh, empty data directory; no overwrite/resume is performed')
    import env_config as cfg
    for key, value in cfg.EXPERT_FAST_OVERRIDES.items():
        setattr(cfg, key, value)
    if args.solver_profile != 'baseline' and args.clearance_mm != .5:
        parser.error('The low-iteration candidate is restricted to 0.50 mm; use baseline for other clearances')
    for key, value in cfg.COLLECTION_SOLVER_PROFILES[args.solver_profile].items():
        setattr(cfg, key, value)
    if abs(cfg.PHYSICS_DT_S - 1/240) > 1e-12:
        parser.error('Collector currently validates 240 Hz physics only')
    from env_workspace_grid import get_grid_point
    grids = args.grid_ids * args.num_envs if len(args.grid_ids) == 1 else args.grid_ids
    positions = [get_grid_point(grid)['hole_position_m'] for grid in grids]
    args.data_dir.mkdir(parents=True, exist_ok=True)
    report = dict(state='starting', split='unassigned', num_envs=args.num_envs,
        grid_ids=grids, requested_episodes=args.num_envs * args.episodes_per_env,
        physics_hz=240, servo_hz=args.servo_hz, policy_hz=30, solver_profile=args.solver_profile,
        replay_requested=args.replay, reference_reads=args.reference_reads, device=args.device,
        reset_mode='batch_barrier_fixed_holes', episodes=[], timing_seconds={})
    def save():
        write_json(args.data_dir / 'collection_summary.json', report)
        write_json(args.data_dir / 'manifest.json', dict(schema='yam_cartesian_rgbd_pilot_v1',
            split='unassigned', episodes=report['episodes']))
    save()
    args.enable_cameras = True
    launcher = AppLauncher(args)
    writers = []
    stack = ExitStack()
    try:
        import numpy as np
        import torch
        import h5py
        np.random.seed(0)
        torch.manual_seed(0)
        import isaaclab.sim as sim_utils
        from env_yam import create_simulation
        from env_yam import ParallelYamScene, YamPickInsertEnv, CartesianYamEnv
        from env_expert import ParallelExpert
        from env_dataset import WristCameraBatch, EpisodeRecorder
        sim = create_simulation(args.device)
        sim.set_camera_view(cfg.OVERVIEW_EYE_M, cfg.OVERVIEW_TARGET_M)
        light = sim_utils.DomeLightCfg(intensity=cfg.LIGHT_INTENSITY)
        light.func('/World/Light', light)
        scene = ParallelYamScene(sim, args.num_envs, args.clearance_mm, positions)
        origin = scene.scene.env_origins[0].cpu().numpy()
        sim.set_camera_view(np.asarray(cfg.OVERVIEW_EYE_M) + origin, np.asarray(cfg.OVERVIEW_TARGET_M) + origin)
        cameras = WristCameraBatch([path + '/YAM/Geometry/world/base' for path in scene.scene.env_prim_paths])
        sim.reset()
        from env_profiling import RolloutProfiler
        profiler = RolloutProfiler()
        if args.profile:
            stack.enter_context(profiler.engine_calls(sim))
        checks = scene.validate_workpiece_colliders()
        if not all(check['passed'] for check in checks):
            raise RuntimeError('Workpiece collider verification failed')
        report['collider_checks'] = checks
        low = YamPickInsertEnv(scene, args.clearance_mm, control_period_s=1/args.servo_hz, max_episode_seconds=60.)
        low.legacy_pose_reads = args.reference_reads
        low.profile_seconds = {}
        env = CartesianYamEnv(low)
        obs, _ = env.reset()
        reference = [scene.robot.get_ee_pose_base(row) for row in range(args.num_envs)]
        position_difference = max(float(np.linalg.norm(obs['ee_pose_base'][row, :3].cpu().numpy() - p))
            for row, (p, _) in enumerate(reference))
        if position_difference > 2e-6:
            raise RuntimeError('Batched native TCP does not match reference getter')
        report['native_pose_max_position_difference_m'] = position_difference
        timings = report['timing_seconds']
        def charge(name, stamp):
            timings[name] = timings.get(name, 0.) + time.perf_counter() - stamp
        def capture(force=False):
            before = low.global_steps
            stamp = time.perf_counter()
            frames = cameras.capture(sim, low.global_steps * low.physics_dt, force=force)
            charge('render_download', stamp)
            if low.global_steps != before:
                raise RuntimeError('Rendering advanced physics')
            return frames
        for _ in range(8):
            frames = capture(force=True)
        if any(not frame['rgb'].any() for frame in frames):
            raise RuntimeError('Black camera after warmup')
        sources = ['env_config.py', 'robot_yam.py', 'env_yam.py', 'env_expert.py',
            'env_task.py', 'env_dataset.py', 'env_workspace_grid.py', 'env_profiling.py', 'record_expert_data.py']
        hashes = {name: hashlib.sha256((cfg.PROJECT_ROOT / name).read_bytes()).hexdigest() for name in sources}
        metadata = dict(split='unassigned', seed=0, action_schema=env.ACTION_SCHEMA,
            backend='isaaclab-2.1.0_physx_sim-4.5', public_quaternion_order='xyzw', native_quaternion_order='wxyz',
            device=str(sim.device), solver_profile=args.solver_profile,
            robot_config=jsonable(vars(scene.robot.config)),
            physics_dt_s=low.physics_dt, servo_period_s=low.control_period, policy_period_s=env.period,
            clearance_mm=args.clearance_mm, peg_diameter_mm=cfg.HOLE_DIAMETER_MM - 2*args.clearance_mm,
            policy_observation_keys=['obs/images/rgb', 'obs/joint_position', 'obs/gripper_opening'],
            privileged_not_policy_input=True, compression=args.compression, source_sha256=hashes,
            environment_config={key: jsonable(getattr(cfg, key)) for key in dir(cfg) if key.isupper()},
            versions={key: importlib.metadata.version(key) for key in ('isaaclab','isaacsim','torch','h5py')})
        write_json(args.data_dir / 'environment_config.json', metadata)
        # Timed collection excludes setup and shader warmup; replay is timed separately.
        timings.clear()
        total_start = time.perf_counter()
        collection_profile = {}
        collection_engine = {}
        report['state'] = 'recording'
        for batch in range(args.episodes_per_env):
            report['state'] = 'recording'
            low.profile_seconds = {}
            profiler.sections.clear()
            if batch:
                obs, _ = env.reset()
                frames = capture(force=True)
            expert = ParallelExpert(scene, args.data_dir / 'expert_diagnostics' / f'batch_{batch:03d}',
                args.clearance_mm, observation=obs)
            expert.batch_cpu_snapshots = not args.reference_reads
            writers = []
            batch_entries = [None] * args.num_envs
            for row in range(args.num_envs):
                number = batch * args.num_envs + row
                path = args.data_dir / 'episodes' / f'episode_{number:06d}.hdf5'
                detail = dict(metadata, grid_id=grids[row], env_id=row, batch_id=batch,
                    episode_id=int(obs['episode_id'][row]), camera=cameras.metadata(row),
                    initial_state=jsonable(low._observation_row(obs, row)))
                writer = EpisodeRecorder(path, detail, compression=args.compression)
                writer.begin(low._observation_row(obs, row), frames[row])
                writers.append(writer)
            ticks = 0
            while any(writer is not None for writer in writers):
                if not launcher.app.is_running() or (args.data_dir / 'STOP_REQUESTED').exists():
                    raise InterruptedError('Stopped: partial episodes retained')
                if time.perf_counter() - total_start > args.max_wall_seconds:
                    raise TimeoutError('Collection wall-time limit exceeded')
                stamp = time.perf_counter()
                action = expert.get_cartesian_actions(env.period, observation=obs)
                charge('expert_action', stamp)
                terminal_frames = {}
                def observer(dt, current):
                    expert.observe(dt, current)
                    for row, controller in enumerate(expert.experts):
                        if writers[row] is not None and controller.finished and not controller.insertion_success:
                            low.report_failure([row], 'expert_controller_failed')
                    ended = [row for row in range(args.num_envs) if writers[row] is not None
                        and bool((low.terminated | low.truncated)[row]) and row not in terminal_frames]
                    if ended:
                        images = capture()
                        for row in ended:
                            terminal_frames[row] = images[row]
                stamp = time.perf_counter()
                obs, reward, _, _, info = env.step(action, physics_observer=observer)
                charge('env_step_total_including_observer_render', stamp)
                active_rows = [row for row, writer in enumerate(writers) if writer is not None]
                continuing = [row for row in active_rows if not bool((low.terminated | low.truncated)[row])]
                images = capture() if continuing else None
                stamp = time.perf_counter()
                for row in active_rows:
                    done = bool((low.terminated | low.truncated)[row])
                    state = low.terminal_observations[row] if done else low._observation_row(obs, row)
                    frame = terminal_frames[row] if done else images[row]
                    end = float(state['physics_timestamp_s'])
                    keep = info['servo_command_timestamp_s'][:, row] < end - 1e-12
                    servo_times = info['servo_command_timestamp_s'][keep, row]
                    measured = info['servo_measured_joint_position'][keep, row].clone()
                    measured_times = (servo_times + low.control_period).clamp(max=end)
                    if done and len(measured):
                        # At 60/120 Hz a terminal physics substep can precede
                        # the end of its servo tick. Do not label future state.
                        measured[-1] = state['joint_position']
                    writer = writers[row]
                    writer.append(action[row], info['commanded_pose_base'][row], state, frame,
                        servo_actions=info['servo_joint_actions'][keep, row],
                        servo_timestamps=servo_times, servo_joint_positions=measured,
                        servo_measurement_timestamps=measured_times,
                        terminated=bool(low.terminated[row]), truncated=bool(low.truncated[row]), reward=float(reward[row]))
                    if done:
                        controller = expert.experts[row]
                        if not controller.finished:
                            controller.insertion_success = bool(low.success[row])
                            controller.released_to_bottom = bool(low.success[row])
                            controller.finish(bool(low.success[row]) and controller.grasp_success, low.reasons[row])
                        writer.finish(success=bool(low.success[row]) and controller.grasp_success, reason=low.reasons[row])
                        entry = dict(file=writer.path.relative_to(args.data_dir).as_posix(), grid_id=grids[row],
                            env_id=row, batch_id=batch, **audit_episode(writer.path))
                        batch_entries[row] = entry
                        report['episodes'].append(entry)
                        writers[row] = None
                charge('writer_finalize_audit', stamp)
                ticks += 1
                if ticks % 30 == 0:
                    report.update(completed_episodes=len(report['episodes']), batch_id=batch, policy_ticks=ticks)
                    save()
                    print(f'[RECORD] batch={batch} ticks={ticks} complete={len(report["episodes"])}', flush=True)
            for key, value in low.profile_seconds.items():
                collection_profile[key] = collection_profile.get(key, 0.) + value
            for key, value in profiler.sections.items():
                entry = collection_engine.setdefault(key, dict(seconds=0., count=0))
                entry['seconds'] += value['seconds']
                entry['count'] += value['count']
            report['completed_episodes'] = len(report['episodes'])
            if args.replay:
                report['state'] = 'replaying'
                save()
                print(f'[REPLAY] batch={batch}', flush=True)
                stamp = time.perf_counter()
                replay = []
                for entry in batch_entries:
                    with h5py.File(args.data_dir / entry['file'], 'r') as file:
                        replay.append((file['actions/cartesian'][:], file['obs/ee_pose_base'][:]))
                obs, _ = env.reset()
                maxima = np.zeros(args.num_envs)
                for tick in range(max(len(item[0]) for item in replay)):
                    if not launcher.app.is_running() or (args.data_dir / 'STOP_REQUESTED').exists():
                        raise InterruptedError('Replay stopped; completed recordings retained')
                    if time.perf_counter() - total_start > args.max_wall_seconds:
                        raise TimeoutError('Collection/replay wall-time limit exceeded')
                    issued = np.zeros((args.num_envs, 7), dtype=np.float32)
                    for row, (actions, _) in enumerate(replay):
                        if tick < len(actions):
                            issued[row] = actions[tick]
                    obs, _, _, _, _ = env.step(torch.as_tensor(issued, device=sim.device))
                    measured = obs['ee_pose_base'][:, :3].cpu().numpy()
                    for row, (actions, poses) in enumerate(replay):
                        if tick < len(actions):
                            maxima[row] = max(maxima[row], float(np.linalg.norm(measured[row] - poses[tick+1, :3])))
                for row, entry in enumerate(batch_entries):
                    entry.update(replay_success=bool(low.success[row]), replay_max_position_difference_m=float(maxima[row]),
                        replay_passed=bool(low.success[row]) and maxima[row] <= .003)
                charge('optional_replay', stamp)
            save()
        report.update(state='completed', wall_seconds=time.perf_counter() - total_start,
            completed_episodes=len(report['episodes']), low_level_timing_seconds=collection_profile,
            engine_collection_timing=collection_engine,
            success_count=sum(entry['episode_success'] for entry in report['episodes']))
        report['collection_wall_seconds'] = report['wall_seconds'] - timings.get('optional_replay', 0.)
        report['episodes_per_hour'] = len(report['episodes']) * 3600 / report['collection_wall_seconds']
        report['passed'] = all(entry['episode_success'] and entry['structure_valid']
            and (not args.replay or entry['replay_passed']) for entry in report['episodes'])
        report['timing_note'] = 'Wall timings include synchronization; low-level fields are nested in env_step, render may be nested in observer. Do not sum overlapping categories.'
        save()
        print(json.dumps(jsonable(report), indent=2), flush=True)
        if not report['passed']:
            raise RuntimeError('Some episodes failed; all outcomes retained in manifest')
    except BaseException as error:
        traceback.print_exc()
        report.update(state='error', error=str(error))
        save()
        raise
    finally:
        try:
            for writer in writers:
                if writer is not None:
                    writer.close_partial()
        finally:
            try:
                stack.close()
            finally:
                from env_yam import close_simulation
                close_simulation(launcher.app)


def campaign_plan(points, clearances=(1., .5, .25, .1), num_envs=4):
    """Interleave clearances, including the short last batch without padding."""
    if num_envs < 1 or len({p['grid_id'] for p in points}) != len(points):
        raise ValueError('Invalid grid or parallel count')
    jobs = []
    for offset in range(0, len(points), num_envs):
        ids = [p['grid_id'] for p in points[offset:offset + num_envs]]
        for clearance in clearances:
            tag = f'c{clearance:.2f}'.replace('.', 'p')
            jobs.append(dict(job_id=f'{tag}_batch_{offset//num_envs:03d}',
                clearance_mm=clearance, dataset='dataset_' + tag, grid_ids=ids))
    return jobs


def campaign_main():
    """Durable process supervisor: new scene per batch, never move static holes."""
    import os
    import shutil
    import subprocess
    import sys
    import env_config as cfg
    from env_workspace_grid import make_manifest
    parser = argparse.ArgumentParser(description='Collect all 17x17 positions for four clearances. No automatic train/test split.')
    parser.add_argument('--all_grid', action='store_true', required=True)
    parser.add_argument('--data_dir', type=Path, required=True)
    parser.add_argument('--num_envs', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.num_envs < 1:
        parser.error('num_envs must be positive')
    root = args.data_dir.resolve()
    checkpoint = root / 'campaign.json'
    if checkpoint.exists() and not args.resume:
        parser.error('Campaign exists; use --resume, never overwrite')
    if args.resume and not checkpoint.exists():
        parser.error('No campaign checkpoint to resume')
    if not checkpoint.exists() and root.exists() and any(root.iterdir()):
        parser.error('New campaign directory must be empty')
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / 'RUNNING.lock'
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        parser.error('RUNNING.lock exists: confirm prior supervisor exited before removing a stale lock')
    os.write(lock_fd, str(os.getpid()).encode())
    os.close(lock_fd)
    child = None
    state = None
    source_names = ['record_expert_data.py','env_config.py','env_yam.py','env_expert.py',
        'env_dataset.py','env_task.py','env_workspace_grid.py','env_profiling.py','robot_yam.py']
    def sources():
        return {name: hashlib.sha256((cfg.PROJECT_ROOT/name).read_bytes()).hexdigest() for name in source_names}
    def assets():
        return {p.relative_to(cfg.PROJECT_ROOT).as_posix(): [p.stat().st_size,p.stat().st_mtime_ns]
            for p in sorted((cfg.PROJECT_ROOT/'assets').rglob('*')) if p.is_file()}
    def save():
        write_json(checkpoint, state)
        rows = list(state['cases'].values())
        finished = [row for row in rows if row['status'] in ('success','task_failure')]
        elapsed = time.time() - state['started_unix_s']
        status = dict(state=state['state'], pid=os.getpid(), active=state.get('active'),
            expected=len(rows), completed=len(finished), succeeded=sum(r['status']=='success' for r in finished),
            failed=sum(r['status']=='task_failure' for r in finished), pending=len(rows)-len(finished),
            elapsed_seconds=elapsed, estimated_remaining_seconds=(elapsed/len(finished)*(len(rows)-len(finished)) if finished else None),
            free_disk_gb=shutil.disk_usage(root).free/1e9, updated_unix_s=time.time(),
            error=state.get('error'), worker_progress=state.get('worker_progress'))
        write_json(root/'status.json', status)
        for clearance in (1., .5, .25, .1):
            selected = [row for row in rows if row['clearance_mm']==clearance]
            folder = root/('dataset_c'+f'{clearance:.2f}'.replace('.','p'))
            folder.mkdir(exist_ok=True)
            write_json(folder/'manifest.json', dict(split='unassigned', clearance_mm=clearance,
                expected=289, cases=selected))
    try:
        grid = make_manifest()
        if grid['side_count'] != 17 or len(grid['points']) != 289:
            raise ValueError('Requested campaign requires the complete 17x17 grid')
        if checkpoint.exists():
            state = json.loads(checkpoint.read_text(encoding='utf-8'))
            if state['num_envs'] != args.num_envs or state['sources'] != sources() or state['assets'] != assets():
                raise RuntimeError('Source/assets/settings changed: refusing mixed-version resume')
            if state['grid'] != grid:
                raise RuntimeError('Grid changed: refusing resume')
            state.update(state='running', active=None, error=None)
        else:
            jobs = campaign_plan(grid['points'], num_envs=args.num_envs)
            cases = {f"{job['dataset']}/{grid_id}": dict(case_id=f"{job['dataset']}/{grid_id}",
                grid_id=grid_id, clearance_mm=job['clearance_mm'], status='pending')
                for job in jobs for grid_id in job['grid_ids']}
            state = dict(state='running', started_unix_s=time.time(), num_envs=args.num_envs,
                sources=sources(), assets=assets(), grid=grid, jobs=jobs, cases=cases,
                attempts=[], settings=dict(servo_hz=240, physics_hz=240, solver_profile='baseline',
                    compression='gzip', replay=False, episodes_per_position=1, split='unassigned'))
            snapshot = root/'source_snapshot'
            snapshot.mkdir()
            for name in source_names:
                shutil.copy2(cfg.PROJECT_ROOT/name, snapshot/name)
            write_json(root/'grid_manifest.json', grid)
        save()
        consecutive_infra = 0
        for job in state['jobs']:
            while True:
                remaining = [g for g in job['grid_ids'] if state['cases'][f"{job['dataset']}/{g}"]['status']=='pending']
                if not remaining:
                    break
                if (root/'STOP_REQUESTED').exists():
                    state['state']='stopped_by_request'; save(); return
                if sources()!=state['sources'] or assets()!=state['assets']:
                    state.update(state='stopped_source_changed', error='Source or assets changed during campaign'); save(); return
                # Reserve 20 GB plus worst-case uncompressed 60 s RGB-D for this batch.
                required = 20e9 + len(remaining)*1801*640*360*8
                if shutil.disk_usage(root).free < required:
                    state.update(state='stopped_low_disk', error='Insufficient free space for next batch plus 20 GB reserve'); save(); return
                attempt = sum(row['job_id']==job['job_id'] for row in state['attempts'])
                relative = Path(job['dataset'])/f"batch_{job['job_id']}_attempt_{attempt:02d}"
                destination = root/relative
                log_path = destination.with_suffix('.log')
                command = [sys.executable,'-u',str(Path(__file__).resolve()),'--headless',
                    '--device','cuda:0','--num_envs',str(len(remaining)),'--grid_ids',*remaining,
                    '--clearance_mm',str(job['clearance_mm']),'--servo_hz','240',
                    '--solver_profile','baseline','--compression','gzip','--max_wall_seconds','3600',
                    '--data_dir',str(destination)]
                record = dict(job_id=job['job_id'], grid_ids=remaining, directory=relative.as_posix(),
                    command=command, state='running', started_unix_s=time.time())
                state['attempts'].append(record)
                state['active']=record
                save()
                print('[CAMPAIGN START]',job['job_id'],remaining,flush=True)
                with log_path.open('x',encoding='utf-8') as log:
                    child=subprocess.Popen(command,cwd=cfg.PROJECT_ROOT,stdout=log,stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                    while child.poll() is None:
                        summary_path=destination/'collection_summary.json'
                        try:
                            progress=json.loads(summary_path.read_text(encoding='utf-8'))
                            state['worker_progress']={k:progress.get(k) for k in ('state','policy_ticks','completed_episodes','error')}
                        except (FileNotFoundError,json.JSONDecodeError):
                            pass
                        stop = ((root/'STOP_REQUESTED').exists() or shutil.disk_usage(root).free<20e9
                            or time.time()-record['started_unix_s']>3780)
                        if stop:
                            if destination.exists():
                                (destination/'STOP_REQUESTED').touch(exist_ok=True)
                            try:
                                child.wait(timeout=90)
                            except subprocess.TimeoutExpired:
                                child.terminate()
                                try: child.wait(timeout=30)
                                except subprocess.TimeoutExpired: child.kill(); child.wait()
                            record.update(state='interrupted',returncode=child.returncode)
                            state.update(state='stopped',error='Stop request, disk reserve or worker deadline reached')
                            save(); return
                        save()
                        time.sleep(5)
                returncode=child.returncode
                child=None
                try:
                    summary=json.loads((destination/'collection_summary.json').read_text(encoding='utf-8'))
                except (FileNotFoundError,json.JSONDecodeError):
                    summary={}
                accepted=0
                import h5py
                for episode in summary.get('episodes',[]):
                    grid_id=episode.get('grid_id')
                    if grid_id not in remaining or not episode.get('structure_valid'):
                        continue
                    path=(destination/episode['file']).resolve()
                    if not path.is_relative_to(destination.resolve()):
                        raise RuntimeError('Episode path escapes batch directory')
                    with h5py.File(path,'r') as file:
                        meta=json.loads(file.attrs['metadata_json'])
                        if not file.attrs['complete'] or meta['grid_id']!=grid_id or meta['clearance_mm']!=job['clearance_mm']:
                            raise RuntimeError('Episode metadata disagrees with campaign plan')
                        success=bool(file.attrs['episode_success'])
                        reason=str(file.attrs['terminal_reason'])
                    case=state['cases'][f"{job['dataset']}/{grid_id}"]
                    case.update(status='success' if success else 'task_failure', episode_success=success,
                        file=path.relative_to(root/job['dataset']).as_posix(), reason=reason,
                        transitions=episode['transitions'], structure_valid=True)
                    accepted+=1
                record.update(state='finished' if accepted==len(remaining) else 'infrastructure_failure',
                    returncode=returncode, accepted=accepted, finished_unix_s=time.time())
                state['active']=None
                consecutive_infra=0 if accepted else consecutive_infra+1
                save()
                print('[CAMPAIGN END]',job['job_id'],'accepted',accepted,'/',len(remaining),'returncode',returncode,flush=True)
                if consecutive_infra>=3:
                    state.update(state='stopped_infrastructure',error='Three worker failures with no completed episodes'); save(); return
        state.update(state='completed',active=None,worker_progress=None)
        save()
    except BaseException as error:
        traceback.print_exc()
        if state is not None:
            state.update(state='error',error=str(error))
            save()
        raise
    finally:
        if child is not None and child.poll() is None:
            if state and state.get('active'):
                directory=root/state['active']['directory']
                if directory.exists(): (directory/'STOP_REQUESTED').touch(exist_ok=True)
            try: child.wait(timeout=90)
            except subprocess.TimeoutExpired: child.terminate(); child.wait()
        lock_path.unlink(missing_ok=True)


if __name__ == '__main__':
    import sys
    if '--all_grid' in sys.argv:
        campaign_main()
    else:
        main()
