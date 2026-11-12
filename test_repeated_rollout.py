"""Shared-observation expert: independent complete episodes and partial resets.

This saves validation diagnostics, not a policy training dataset.
"""
import argparse
import json
from pathlib import Path
import time
import traceback


def json_value(value):
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if hasattr(value, 'detach'):
        return value.detach().cpu().tolist()
    if hasattr(value, 'tolist'):
        return value.tolist()
    return value


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes_per_env', type=int, default=3)
    parser.add_argument('--output_dir', type=Path, default=Path(__file__).parent / 'outputs' / 'repeated_rollout')
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.episodes_per_env < 1:
        parser.error('episodes_per_env must be positive')
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Choose an empty output directory; previous episodes are never overwritten')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    import env_config as cfg
    for name, value in cfg.EXPERT_FAST_OVERRIDES.items():
        setattr(cfg, name, value)
    launcher = AppLauncher(args)
    report = dict(state='starting', num_envs=4, episodes_per_env=args.episodes_per_env,
        episodes=[], resets=[], training_data_collected=False, observation_source='shared_env')
    def save():
        temporary = args.output_dir / 'repeat_summary.tmp'
        temporary.write_text(json.dumps(report, indent=2), encoding='utf-8')
        temporary.replace(args.output_dir / 'repeat_summary.json')
    save()
    try:
        import torch
        import isaaclab.sim as sim_utils
        from isaaclab_physx.physics import PhysxCfg
        from env_yam import ParallelYamScene
        from env_yam import YamPickInsertEnv
        from env_expert import ParallelExpert
        from env_workspace_grid import get_grid_point
        from env_task import geometry_diagnostic_thresholds
        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=cfg.PHYSICS_DT_S, device=args.device,
            enable_scene_query_support=True, physics=PhysxCfg(
                min_position_iteration_count=cfg.SCENE_MIN_POSITION_ITERATIONS,
                min_velocity_iteration_count=cfg.SCENE_MIN_VELOCITY_ITERATIONS,
                enable_external_forces_every_iteration=cfg.TGS_EXTERNAL_FORCES_EVERY_ITERATION,
                solve_articulation_contact_last=cfg.SOLVE_ARTICULATION_CONTACT_LAST)))
        grids = ['r08_c08', 'r00_c00', 'r00_c16', 'r16_c16']
        scene = ParallelYamScene(sim, 4, .5, [get_grid_point(grid)['hole_position_m'] for grid in grids])
        sim.reset()
        checks = scene.validate_workpiece_colliders()
        if not all(item['passed'] for item in checks):
            raise RuntimeError('Workpiece collider pose check failed')
        env = YamPickInsertEnv(scene, .5, max_episode_seconds=60.)
        obs, _ = env.reset()
        expert = ParallelExpert(scene, args.output_dir, .5, observation=obs)
        completed = [0] * 4
        last_timestamp = [-1.] * 4
        saw_async_reset = False
        report.update(state='running', grids=grids, collider_checks=checks,
            geometry_diagnostic_thresholds=geometry_diagnostic_thresholds())
        save()
        start = time.perf_counter()
        # Bound wall time independently of simulation timeout, and stop our run
        # without affecting any other user process.
        while launcher.app.is_running() and min(completed) < args.episodes_per_env:
            if (args.output_dir / 'STOP_REQUESTED').exists() or time.perf_counter() - start > 900:
                raise InterruptedError('Stop requested or 900-second rollout limit')
            actions = expert.get_actions(env.control_period, observation=obs)
            obs, reward, terminated, truncated, info = env.step(actions,
                render=not args.headless and env.global_steps % cfg.RENDER_EVERY_N_STEPS == 0,
                physics_observer=expert.observe)
            for index, controller in enumerate(expert.experts):
                if completed[index] < args.episodes_per_env and controller.finished and not controller.insertion_success:
                    env.report_failure([index], 'expert_controller_failed')
            reset_ids = []
            for index in range(4):
                if completed[index] >= args.episodes_per_env:
                    continue
                timestamp = float(obs['physics_timestamp_s'][index].item())
                if timestamp <= last_timestamp[index]:
                    raise AssertionError('Observation physics time did not advance')
                last_timestamp[index] = timestamp
                if not bool(env.terminated[index] | env.truncated[index]):
                    continue
                controller = expert.experts[index]
                success = bool(env.success[index])
                if not controller.finished:
                    controller.insertion_success = success
                    controller.released_to_bottom = success
                    controller.finish(success and controller.grasp_success,
                        'Shared environment: ' + env.reasons[index])
                terminal = env.terminal_observations[index]
                episode_id = int(terminal['episode_id'].item())
                assert episode_id == completed[index] + 1
                assert float(terminal['episode_time_s'].item()) > 0
                assert float(terminal['physics_timestamp_s'].item()) <= timestamp
                terminal_path = controller.output_dir / 'environment_terminal.json'
                if terminal_path.exists():
                    raise AssertionError('Terminal observation would overwrite a previous episode')
                terminal_path.write_text(json.dumps(json_value(terminal), indent=2), encoding='utf-8')
                record = dict(env_id=index, grid_id=grids[index], episode_id=episode_id,
                    success=success, reason=env.reasons[index],
                    controller_success=controller.grasp_success and controller.insertion_success,
                    episode_seconds=float(terminal['episode_time_s'].item()),
                    terminal_physics_timestamp_s=float(terminal['physics_timestamp_s'].item()),
                    max_radial_violation_m=float(env.max_radial_violation[index]),
                    max_bottom_penetration_m=float(env.max_bottom_penetration[index]),
                    terminal_file=str(terminal_path.relative_to(args.output_dir)))
                report['episodes'].append(record)
                completed[index] += 1
                print('[EPISODE]', json.dumps(record), flush=True)
                if completed[index] < args.episodes_per_env:
                    reset_ids.append(index)
            if reset_ids:
                before = obs
                before_steps = env.global_steps
                other_ids = [index for index in range(4) if index not in reset_ids]
                other_controllers = [expert.experts[index] for index in other_ids]
                other_done = (env.terminated | env.truncated)[other_ids].clone()
                saw_async_reset |= any(not bool(value) for value in other_done)
                obs, _ = env.reset(reset_ids)
                for name in ('joint_position', 'joint_velocity', 'ee_pose_base', 'gripper_opening',
                             'previous_action', 'episode_id', 'episode_time_s', 'physics_timestamp_s'):
                    torch.testing.assert_close(obs[name][other_ids], before[name][other_ids], rtol=0, atol=0)
                for name in ('peg_pose_base', 'peg_linear_velocity', 'hole_position_base'):
                    torch.testing.assert_close(obs['privileged'][name][other_ids], before['privileged'][name][other_ids], rtol=0, atol=0)
                assert env.global_steps == before_steps
                assert not bool((env.terminated | env.truncated)[reset_ids].any())
                expert.reset(reset_ids, reset_scene=False, observation=obs)
                assert [expert.experts[index] for index in other_ids] == other_controllers
                for index in reset_ids:
                    assert obs['episode_time_s'][index] == 0
                    assert int(obs['episode_id'][index]) == completed[index] + 1
                report['resets'].append(dict(env_ids=reset_ids, physics_step=env.global_steps,
                    other_rows_unchanged=True, no_physics_advance=True))
            report['completed_per_env'] = completed.copy()
            if reset_ids or report['episodes'] and sum(completed) != report.get('saved_episode_count', 0):
                report['saved_episode_count'] = sum(completed)
                save()
        report.update(state='completed' if min(completed) == args.episodes_per_env else 'interrupted',
            all_passed=len(report['episodes']) == 4 * args.episodes_per_env
                and all(item['success'] and item['controller_success'] for item in report['episodes']),
            asynchronous_reset_observed=saw_async_reset, wall_seconds=time.perf_counter() - start,
            physics_steps=env.global_steps)
        save()
        print(json.dumps(report, indent=2), flush=True)
        if not report['all_passed']:
            raise RuntimeError('Repeated rollout regression failed; inspect retained episodes')
    except Exception as error:
        traceback.print_exc()
        report.update(state='error', error=str(error))
        save()
        raise
    finally:
        launcher.app.close()


if __name__ == '__main__':
    main()
