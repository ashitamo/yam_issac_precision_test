"""Native shared-env contract smoke test, optionally driving it with the expert."""
import argparse
import json
from pathlib import Path
import time
import traceback


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expert', action='store_true')
    parser.add_argument('--output_dir', type=Path, default=Path(__file__).parent / 'outputs' / 'env_interface')
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Choose an empty output directory')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    import env_config as cfg
    if args.expert:
        for name, value in cfg.EXPERT_FAST_OVERRIDES.items():
            setattr(cfg, name, value)
    launcher = AppLauncher(args)
    try:
        import torch
        import isaaclab.sim as sim_utils
        from env_yam import create_simulation
        from env_yam import ParallelYamScene
        from env_yam import YamPickInsertEnv
        from env_task import geometry_diagnostic_thresholds
        from env_workspace_grid import get_grid_point
        sim = create_simulation(args.device)
        positions = [get_grid_point(grid)['hole_position_m'] for grid in ('r08_c08', 'r16_c16')]
        scene = ParallelYamScene(sim, 2, .5, positions)
        sim.reset()
        checks = scene.validate_workpiece_colliders()
        if not all(check['passed'] for check in checks):
            raise RuntimeError('Physical workpiece pose verification failed')
        env = YamPickInsertEnv(scene, .5, max_episode_seconds=60.)
        obs, reset_info = env.reset()
        report = dict(mode='expert' if args.expert else 'contract_smoke', collider_checks=checks,
            geometry_diagnostic_thresholds=geometry_diagnostic_thresholds(),
            action_schema=env.ACTION_SCHEMA, observation_fields=list(obs), cameras_enabled=False)
        start = time.perf_counter()
        if args.expert:
            from env_expert import ParallelExpert
            expert = ParallelExpert(scene, args.output_dir, .5, observation=obs)
            # The expert only generates actions; the environment owns gravity FF
            # and all simulator stepping. observe remains a diagnostic controller
            # phase tracker; shared env_task defines physical geometry for both.
            while launcher.app.is_running() and not bool((env.terminated | env.truncated).all()):
                actions = expert.get_actions(env.control_period, observation=obs)
                def observer(dt, current_obs):
                    expert.observe(dt, current_obs)
                obs, reward, terminated, truncated, info = env.step(actions,
                    render=not args.headless, physics_observer=observer)
                for index, controller in enumerate(expert.experts):
                    if controller.finished and not controller.insertion_success:
                        env.report_failure([index], 'expert_controller_failed')
            # Preserve original expert diagnostics; complete phase bookkeeping
            # if the shared physical terminal condition fired one tick earlier.
            for index, controller in enumerate(expert.experts):
                if not controller.finished:
                    if env.success[index]:
                        controller.insertion_success = True
                        controller.released_to_bottom = True
                        controller.finish(True, 'Shared environment observed released peg settled at bottom')
                    else:
                        controller.finish(False, env.reasons[index] or 'Run interrupted')
            report.update(success=env.success.cpu().tolist(), reasons=env.reasons,
                max_radial_violation_m=env.max_radial_violation.tolist(),
                max_bottom_penetration_m=env.max_bottom_penetration.tolist(),
                passed=bool(env.success.all()))
        else:
            actions = env.last_action.clone()
            actions[:, 6] = .02
            for _ in range(8):
                obs, *_ = env.step(actions)
            other_q = obs['joint_position'][1].clone()
            other_time = obs['episode_time_s'][1].clone()
            before_steps = env.global_steps
            obs, _ = env.reset([0])
            torch.testing.assert_close(obs['joint_position'][1], other_q, rtol=0, atol=0)
            torch.testing.assert_close(obs['episode_time_s'][1], other_time, rtol=0, atol=0)
            assert env.global_steps == before_steps
            assert obs['episode_time_s'][0] == 0 and obs['episode_id'].tolist() == [2, 1]
            report.update(passed=True, partial_reset_preserves_other_row=True,
                reset_does_not_advance_physics=True, episode_ids=obs['episode_id'].cpu().tolist())
            # Deliberate lifecycle probes, not failed manipulation episodes.
            env.max_episode_seconds = float(other_time.item()) + env.physics_dt * .5
            def inject_failure(dt, current_obs):
                env.report_failure([0], 'injected_controller_failure')
            obs, reward, terminated, truncated, info = env.step(env.last_action.clone(), physics_observer=inject_failure)
            assert terminated.tolist() == [True, False]
            assert truncated.tolist() == [False, True]
            assert info['reason'] == ['injected_controller_failure', 'timeout']
            assert info['newly_done'].tolist() == [True, True]
            assert set(info['final_observation']) == {0, 1}
            assert not bool(info['success'].any()) and not bool(reward.any())
            ids_before = obs['episode_id'].clone()
            _, reward, _, _, info = env.step(env.last_action.clone())
            assert not bool(info['newly_done'].any()) and not bool(reward.any())
            torch.testing.assert_close(scene.episode_count, ids_before)
            report.update(injected_failure_and_timeout_verified=True,
                terminal_snapshots_preserved=True, no_automatic_reset=True)
        report.update(wall_seconds=time.perf_counter() - start, physics_steps=env.global_steps)
        (args.output_dir / 'interface_result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2), flush=True)
        if not report['passed']:
            raise RuntimeError('Shared environment regression failed')
    except Exception:
        traceback.print_exc()
        raise
    finally:
        from env_yam import close_simulation
        close_simulation(launcher.app)


if __name__ == '__main__':
    main()
