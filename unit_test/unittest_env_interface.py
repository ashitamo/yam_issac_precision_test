"""CPU contract tests for the shared environment; loaded with unittest discovery."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from scipy.spatial.transform import Rotation

import env_config as cfg
from env_task import peg_geometry, release_ready, contact_gate_pass, geometry_diagnostic_thresholds


def load_environment_class():
    tree = ast.parse((cfg.PROJECT_ROOT / 'env_yam.py').read_text(encoding='utf-8'))
    node = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == 'YamPickInsertEnv')
    namespace = dict(np=np, torch=torch, Rotation=Rotation, cfg=cfg,
        tensor=lambda value: value.torch if hasattr(value, 'torch') else value,
        peg_geometry=peg_geometry, release_ready=release_ready,
        contact_gate_pass=contact_gate_pass, geometry_diagnostic_thresholds=geometry_diagnostic_thresholds)
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<environment contract>', 'exec'), namespace)
    return namespace['YamPickInsertEnv']


class FakeScene:
    def __init__(self, dt=.01):
        self.num_envs = 2
        self.sim = SimpleNamespace(device='cpu', get_physics_dt=lambda: dt, forward=lambda: None)
        self.scene = SimpleNamespace(env_origins=torch.tensor([[0., 0., 0.], [2., 0., 0.]]))
        self.episode_count = torch.zeros(2, dtype=torch.long)
        self.workpiece_positions = np.array([[.265, .065, 0.], [.265, .065, 0.]])
        self.hole_radius_m = .004992666880464469
        data = SimpleNamespace(joint_pos=torch.zeros(2, 8), joint_vel=torch.zeros(2, 8),
            joint_pos_target=torch.zeros(2, 8), joint_pos_limits=torch.tensor([[[-3., 3.]] * 8] * 2),
            gravity_compensation_forces=torch.zeros(2, 8), applied_torque=torch.zeros(2, 8),
            computed_torque=torch.zeros(2, 8))
        art = SimpleNamespace(data=data, num_base_dofs=0)
        art.set_joint_position_target_index = lambda target: setattr(data, 'joint_pos_target', target.clone())
        self.robot = SimpleNamespace(articulation=art, arm_joint_indices=list(range(6)), gripper_joint_indices=[6, 7],
            max_gripper_opening_m=.093961,
            config=SimpleNamespace(gripper_contact_overlap_m=.001039,
                gripper_joint_open_m=-.0475, gripper_joint_closed_m=0.),
            get_ee_pose_base=lambda index: (np.array([.265, .065, .139]), Rotation.identity()),
            set_feedforward_effort_target=lambda value: None)
        self.peg = SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.zeros(2, 3),
            root_quat_w=torch.tensor([[0., 0., 0., 1.]] * 2), root_lin_vel_w=torch.zeros(2, 3)))
        self.calls = 0

    def reset(self, ids):
        self.robot.articulation.data.joint_pos[ids] = 0.
        self.robot.articulation.data.joint_pos_target[ids] = 0.
        self.peg.data.root_pos_w[ids] = self.scene.env_origins[ids] + torch.tensor([.3, -.08, .025])
        self.peg.data.root_lin_vel_w[ids] = 0.
        self.episode_count[ids] += 1

    def step(self, render=False):
        self.calls += 1
        self.robot.articulation.data.joint_pos = self.robot.articulation.data.joint_pos_target.clone()


class EnvironmentInterfaceTests(unittest.TestCase):
    def make_env(self, dt=.01, period=.01, timeout=1.):
        scene = FakeScene(dt)
        env = load_environment_class()(scene, control_period_s=period, max_episode_seconds=timeout)
        return env, scene

    def test_first_reset_required_and_snapshot_is_not_live(self):
        env, scene = self.make_env()
        with self.assertRaises(RuntimeError):
            env.step(torch.zeros(2, 7))
        obs, _ = env.reset()
        obs['joint_position'][0, 0] = 2.
        self.assertEqual(scene.robot.articulation.data.joint_pos[0, 0], 0.)
        torch.testing.assert_close(obs['privileged']['peg_pose_base'][:, :3], torch.tensor([[.3, -.08, .025]] * 2))

    def test_action_units_shape_bounds_and_decimation(self):
        env, scene = self.make_env(dt=.005, period=.01)
        env.reset()
        action = torch.zeros(2, 7)
        action[:, 6] = .02
        obs, reward, terminated, truncated, info = env.step(action)
        self.assertEqual(scene.calls, 2)
        torch.testing.assert_close(obs['gripper_opening'], torch.full((2, 1), .02))
        torch.testing.assert_close(obs['episode_time_s'], torch.full((2,), .01, dtype=torch.float64))
        self.assertFalse(bool((terminated | truncated).any()))
        for bad in (torch.zeros(2, 8), torch.full((2, 7), float('nan')), torch.ones(2, 7, dtype=torch.long)):
            with self.assertRaises(ValueError):
                env.step(bad)
        action[0, 0] = 4.
        with self.assertRaises(ValueError):
            env.step(action)
        self.assertEqual(scene.calls, 2)

    def test_physics_observer_gets_each_post_step_snapshot(self):
        env, scene = self.make_env(dt=.005, period=.01)
        env.reset()
        action = torch.zeros(2, 7)
        action[:, 6] = .02
        received = []
        def observer(dt, obs):
            received.append((dt, obs))
        final, *_ = env.step(action, physics_observer=observer)
        self.assertEqual([value[0] for value in received], [.005, .005])
        self.assertEqual(received[0][1]['physics_timestamp_s'][0].item(), .005)
        self.assertEqual(received[1][1]['physics_timestamp_s'][0].item(), .01)
        self.assertIs(final, received[1][1])
        for _, obs in received:
            torch.testing.assert_close(obs['diagnostics']['joint_pos'][:, :6], obs['joint_position'])
            torch.testing.assert_close(obs['gripper_opening'], torch.full((2, 1), .02))

    def test_partial_reset_preserves_other_row_and_episode_clock(self):
        env, scene = self.make_env()
        env.reset()
        action = torch.zeros(2, 7)
        action[:, 6] = .02
        env.step(action)
        before = env.get_observations()
        after, _ = env.reset([0])
        self.assertEqual(after['episode_time_s'][0], 0.)
        self.assertEqual(after['episode_id'].tolist(), [2, 1])
        torch.testing.assert_close(after['joint_position'][1], before['joint_position'][1])
        self.assertEqual(after['episode_time_s'][1], before['episode_time_s'][1])
        with self.assertRaises(ValueError):
            env.reset([1, 1])
        with self.assertRaises(ValueError):
            env.reset([.5])

    def test_timeout_latches_terminal_snapshot_without_auto_reset(self):
        env, scene = self.make_env(dt=.005, period=.01, timeout=.004)
        env.reset()
        action = torch.zeros(2, 7)
        obs, reward, terminated, truncated, info = env.step(action)
        self.assertTrue(bool(truncated.all()))
        self.assertFalse(bool(terminated.any()))
        self.assertEqual(info['final_observation'][0]['physics_timestamp_s'].item(), .005)
        self.assertEqual(obs['physics_timestamp_s'][0].item(), .01)
        self.assertEqual(obs['episode_time_s'][0].item(), .005)
        self.assertEqual(scene.episode_count.tolist(), [1, 1])
        _, reward, _, _, info = env.step(action)
        self.assertEqual(float(reward.sum()), 0.)
        self.assertFalse(bool(info['newly_done'].any()))

    def test_lift_then_open_and_settle_succeeds_once(self):
        env, scene = self.make_env()
        env.reset()
        action = torch.zeros(2, 7)
        action[:, 6] = cfg.GRASP_OPENING_M
        scene.peg.data.root_pos_w[0] = torch.tensor([.265, .065, .1])
        env.step(action)
        scene.peg.data.root_pos_w[0, 2] = .025
        for _ in range(12):
            _, reward, terminated, _, info = env.step(action)
            if terminated[0]:
                break
        self.assertTrue(bool(info['success'][0]))
        self.assertEqual(reward.tolist(), [1., 0.])
        _, reward, _, _, _ = env.step(action)
        self.assertEqual(reward.tolist(), [0., 0.])

    def test_controller_failure_latches_only_requested_row(self):
        env, scene = self.make_env()
        env.reset()
        env.report_failure([0], 'ik_failed')
        self.assertEqual(env.terminated.tolist(), [True, False])
        action = torch.zeros(2, 7)
        obs, reward, _, _, info = env.step(action)
        self.assertEqual(info['reason'], ['ik_failed', ''])
        self.assertFalse(bool(info['success'].any()))
        self.assertEqual(obs['episode_time_s'][0], 0.)
        self.assertGreater(obs['episode_time_s'][1], 0.)
        self.assertEqual(float(reward.sum()), 0.)

    def test_contact_violation_is_not_hidden_by_later_settling(self):
        env, scene = self.make_env()
        env.reset()
        action = torch.zeros(2, 7)
        action[:, 6] = cfg.GRASP_OPENING_M
        scene.peg.data.root_pos_w[0] = torch.tensor([.265, .065, .1])
        env.step(action)
        scene.peg.data.root_pos_w[0] = torch.tensor([.266, .065, .025])
        env.step(action)
        scene.peg.data.root_pos_w[0] = torch.tensor([.265, .065, .025])
        for _ in range(12):
            _, reward, terminated, _, info = env.step(action)
            if terminated[0]:
                break
        self.assertEqual(info['reason'][0], 'contact_violation')
        self.assertFalse(bool(info['success'][0]))
        self.assertEqual(float(reward.sum()), 0.)


if __name__ == '__main__':
    unittest.main()
