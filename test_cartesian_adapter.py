"""Native TCP-Jacobian, 30 Hz action contract, and optional full expert regression."""
import argparse
import json
from pathlib import Path
import time
import traceback


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expert', action='store_true')
    parser.add_argument('--output_dir', type=Path, default=Path(__file__).parent / 'outputs' / 'cartesian_adapter')
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Use an empty output directory')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    import env_config as cfg
    if args.expert:
        for name, value in cfg.EXPERT_FAST_OVERRIDES.items():
            setattr(cfg, name, value)
    launcher = AppLauncher(args)
    report = dict(state='starting', training_data_collected=False, cameras_enabled=False)
    def save():
        (args.output_dir / 'cartesian_result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    save()
    try:
        import numpy as np
        import torch
        import isaaclab.sim as sim_utils
        from isaaclab_physx.physics import PhysxCfg
        from env_yam import ParallelYamScene, tensor
        from env_yam import YamPickInsertEnv
        from env_yam import CartesianYamEnv, compose_target, encode_reference
        from env_workspace_grid import get_grid_point
        from scipy.spatial.transform import Rotation
        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=cfg.PHYSICS_DT_S, device=args.device,
            enable_scene_query_support=True, physics=PhysxCfg(
                min_position_iteration_count=cfg.SCENE_MIN_POSITION_ITERATIONS,
                min_velocity_iteration_count=cfg.SCENE_MIN_VELOCITY_ITERATIONS,
                enable_external_forces_every_iteration=cfg.TGS_EXTERNAL_FORCES_EVERY_ITERATION,
                solve_articulation_contact_last=cfg.SOLVE_ARTICULATION_CONTACT_LAST)))
        grids = ['r08_c08', 'r16_c16'] if args.expert else ['r08_c08']
        scene = ParallelYamScene(sim, len(grids), .5, [get_grid_point(grid)['hole_position_m'] for grid in grids])
        sim.reset()
        checks = scene.validate_workpiece_colliders()
        if not all(check['passed'] for check in checks):
            raise RuntimeError('Workpiece pose check failed')
        low = YamPickInsertEnv(scene, .5, max_episode_seconds=60.)
        low.reset()
        art = scene.robot.articulation
        # Kinematics-only finite differences before rollout; no physics time
        # advances and no such fixture states are labelled as demonstrations.
        original = tensor(art.data.joint_pos).clone()
        native = scene.robot.get_ee_jacobian_base()[0].detach().cpu().numpy()
        finite = []
        ids = torch.tensor([0], device=sim.device)
        epsilon = .001
        for joint in scene.robot.arm_joint_indices:
            poses = []
            for sign in (1, -1):
                q = original[:1].clone()
                q[0, joint] += sign * epsilon
                art.write_joint_state_to_sim_index(position=q, velocity=q * 0., env_ids=ids)
                sim.forward()
                poses.append(scene.robot.get_ee_pose_base(0))
            finite.append(np.r_[(poses[0][0] - poses[1][0]) / (2 * epsilon),
                (poses[0][1] * poses[1][1].inv()).as_rotvec() / (2 * epsilon)])
        finite = np.array(finite).T
        position_difference = float(np.max(abs(native[:3] - finite[:3])))
        rotation_difference = float(np.max(abs(native[3:] - finite[3:])))
        report.update(jacobian_position_difference=position_difference, jacobian_rotation_difference=rotation_difference,
            jacobian_passed=position_difference < .0002 and rotation_difference < .002,
            jacobian_probe_is_kinematic_only=True, collider_checks=checks)
        save()
        if not report['jacobian_passed']:
            raise RuntimeError('TCP Jacobian disagrees with native FK finite differences')
        wrapped = CartesianYamEnv(low)
        obs, _ = wrapped.reset()
        report.update(action_schema=wrapped.ACTION_SCHEMA, policy_period_s=wrapped.period,
            servo_period_s=low.control_period, servo_steps_per_policy=wrapped.servo_steps,
            controller='Isaac Lab DifferentialIKController DLS', state='running')
        start = time.perf_counter()
        if args.expert:
            from env_expert import ParallelExpert
            expert = ParallelExpert(scene, args.output_dir, .5, observation=obs)
            policy_steps = 0
            while launcher.app.is_running() and not bool((low.terminated | low.truncated).all()):
                actions = expert.get_cartesian_actions(wrapped.period, observation=obs)
                done = low.terminated | low.truncated
                actions[done, :6] = 0.
                actions[done, 6] = obs['gripper_opening'][done, 0]
                report.update(last_policy_step=policy_steps, last_action=actions.detach().cpu().tolist(),
                    last_measured_pose=obs['ee_pose_base'].detach().cpu().tolist(),
                    last_commanded_reference=expert.last_cartesian_reference.detach().cpu().tolist(),
                    last_measured_joint=obs['joint_position'].detach().cpu().tolist())
                obs, reward, terminated, truncated, info = wrapped.step(actions,
                    render=not args.headless, physics_observer=expert.observe)
                policy_steps += 1
                for index, controller in enumerate(expert.experts):
                    if controller.finished and not controller.insertion_success:
                        low.report_failure([index], 'expert_controller_failed')
            for index, controller in enumerate(expert.experts):
                if not controller.finished:
                    controller.insertion_success = bool(low.success[index])
                    controller.released_to_bottom = bool(low.success[index])
                    controller.finish(bool(low.success[index]) and controller.grasp_success, 'Cartesian shared environment: ' + low.reasons[index])
            report.update(passed=bool(low.success.all()), success=low.success.cpu().tolist(),
                reasons=low.reasons, policy_steps=policy_steps,
                max_radial_violation_m=low.max_radial_violation.tolist(),
                max_bottom_penetration_m=low.max_bottom_penetration.tolist(),
                reference_source='FK of expert issued joint reference, not observed future motion')
        else:
            pose = obs['ee_pose_base'].cpu().numpy()
            dq = np.array([0., .01, 0., 0., 0., .02])
            delta = (scene.robot.get_ee_jacobian_base()[0].cpu().numpy() @ dq)[None, :]
            action = np.c_[delta, [.02]]
            target = compose_target(pose, action)
            for index in range(20):
                actions = torch.tensor(encode_reference(obs['ee_pose_base'].cpu().numpy(), target, [.02]),
                    device=sim.device, dtype=torch.float32)
                before_steps = low.global_steps
                obs, reward, terminated, truncated, info = wrapped.step(actions, render=not args.headless)
                assert low.global_steps - before_steps == 8
                np.testing.assert_allclose(info['commanded_pose_base'].cpu().numpy()[:, :3], target[:, :3], atol=1e-7)
            actual = obs['ee_pose_base'].cpu().numpy()
            position_error = float(np.linalg.norm(actual[0, :3] - target[0, :3]))
            rotation_error = float(np.degrees((Rotation.from_quat(actual[0, 3:]).inv() * Rotation.from_quat(target[0, 3:])).magnitude()))
            report.update(passed=position_error < .001 and rotation_error < .5,
                position_error_m=position_error, orientation_error_deg=rotation_error, policy_steps=20,
                fixed_target_not_reintegrated=True, raw_reference_source='native TCP Jacobian tangent')
        report.update(state='completed', wall_seconds=time.perf_counter() - start, physics_steps=low.global_steps)
        save()
        print(json.dumps(report, indent=2), flush=True)
        if not report['passed']:
            raise RuntimeError('Cartesian adapter regression did not pass')
    except Exception as error:
        traceback.print_exc()
        report.update(state='error', error=str(error))
        save()
        raise
    finally:
        launcher.app.close()


if __name__ == '__main__':
    main()
