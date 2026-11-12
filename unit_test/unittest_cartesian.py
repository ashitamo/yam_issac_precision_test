"""Cartesian action semantics and multi-rate contract without simulator startup."""
import unittest
import numpy as np
import torch
from scipy.spatial.transform import Rotation

import env_config as cfg
from env_yam import CartesianYamEnv, compose_target, encode_reference, bound_expert_reference
from unittest_env_interface import FakeScene, load_environment_class


class HoldingIK:
    def __init__(self):
        self.commands = []
        self.computes = 0
    def reset(self, env_ids):
        pass
    def set_command(self, target):
        self.commands.append(target.clone())
    def compute(self, position, rotation, jacobian, joints):
        self.computes += 1
        return joints.clone()


class CartesianContractTests(unittest.TestCase):
    def test_lower_servo_rates_preserve_policy_and_physics_periods(self):
        for hz, controls in ((60, 2), (120, 4), (240, 8)):
            scene = FakeScene(1/240)
            low = load_environment_class()(scene, control_period_s=1/hz)
            controller = HoldingIK()
            wrapped = CartesianYamEnv(low, controller=controller,
                jacobian_provider=lambda: torch.eye(6).expand(2, -1, -1))
            wrapped.reset()
            obs, _, _, _, info = wrapped.step(torch.zeros(2, 7))
            self.assertEqual(controller.computes, controls)
            self.assertEqual(scene.calls, 8)
            self.assertEqual(info['servo_steps'], controls)
            torch.testing.assert_close(obs['physics_timestamp_s'], torch.full((2,), 1/30, dtype=torch.float64))

    def test_teacher_governor_changes_issued_reference_not_stored_nominal(self):
        raw = np.array([[.1, 0, 0, 0, 0, 1., .02]])
        issued, limited = bound_expert_reference(raw)
        self.assertTrue(limited[0])
        self.assertEqual(raw[0, 0], .1)
        self.assertEqual(raw[0, 5], 1.)
        self.assertAlmostEqual(issued[0, 0], cfg.CARTESIAN_MAX_TRANSLATION_M)
        self.assertAlmostEqual(issued[0, 5], cfg.CARTESIAN_MAX_ROTATION_RAD)
        self.assertEqual(issued[0, 6], .02)

    def make_env(self, dt=1/240, timeout=1.):
        scene = FakeScene(dt)
        low = load_environment_class()(scene, control_period_s=1/240, max_episode_seconds=timeout)
        controller = HoldingIK()
        wrapped = CartesianYamEnv(low, controller=controller, jacobian_provider=lambda: torch.eye(6).expand(2, -1, -1))
        return wrapped, low, scene, controller

    def test_rotation_is_base_left_composed_and_reference_round_trips(self):
        pose = np.array([[.1, .2, .3, *Rotation.from_euler('x', 90, degrees=True).as_quat()]])
        action = np.array([[.01, 0., 0., 0., 0., .1, .02]])
        target = compose_target(pose, action)
        expected = Rotation.from_rotvec([0., 0., .1]) * Rotation.from_quat(pose[0, 3:])
        self.assertLess((expected.inv() * Rotation.from_quat(target[0, 3:])).magnitude(), 1e-12)
        np.testing.assert_allclose(target[0, :3], [.11, .2, .3])
        np.testing.assert_allclose(encode_reference(pose, target, [.02]), action, atol=1e-12)
        right_composed = Rotation.from_quat(pose[0, 3:]) * Rotation.from_rotvec([0., 0., .1])
        self.assertGreater((right_composed.inv() * expected).magnitude(), .1)

    def test_one_action_target_once_eight_servo_steps_and_si_opening(self):
        wrapped, low, scene, controller = self.make_env()
        before, _ = wrapped.reset()
        action = torch.zeros(2, 7)
        action[:, 0] = .01
        action[:, 6] = .02
        obs, _, _, _, info = wrapped.step(action)
        self.assertEqual(len(controller.commands), 1)
        self.assertEqual(controller.computes, 8)
        self.assertEqual(scene.calls, 8)
        torch.testing.assert_close(controller.commands[0][:, 0], before['ee_pose_base'][:, 0] + .01)
        torch.testing.assert_close(obs['physics_timestamp_s'], torch.full((2,), 1/30, dtype=torch.float64))
        torch.testing.assert_close(obs['gripper_opening'], torch.full((2, 1), .02))
        self.assertEqual(tuple(info['servo_joint_actions'].shape), (8, 2, 7))
        self.assertEqual(info['action_schema'], wrapped.ACTION_SCHEMA)

    def test_480_hz_physics_still_eight_240hz_controls(self):
        wrapped, low, scene, controller = self.make_env(dt=1/480)
        wrapped.reset()
        action = torch.zeros(2, 7)
        wrapped.step(action)
        self.assertEqual(scene.calls, 16)
        self.assertEqual(controller.computes, 8)

    def test_invalid_actions_do_not_advance_or_clip(self):
        wrapped, low, scene, controller = self.make_env()
        wrapped.reset()
        for index, value in ((0, cfg.CARTESIAN_MAX_TRANSLATION_M + .001),
                             (3, cfg.CARTESIAN_MAX_ROTATION_RAD + .001), (6, -.001), (1, float('nan'))):
            action = torch.zeros(2, 7)
            action[0, index] = value
            with self.assertRaises(ValueError):
                wrapped.step(action)
        self.assertEqual(scene.calls, 0)
        self.assertEqual(len(controller.commands), 0)

    def test_early_timeout_terminal_frame_not_replaced_by_later_servo_frames(self):
        wrapped, low, scene, controller = self.make_env(timeout=.001)
        wrapped.reset()
        obs, reward, term, trunc, info = wrapped.step(torch.zeros(2, 7))
        self.assertTrue(bool(trunc.all()))
        self.assertEqual(scene.calls, 8)
        self.assertAlmostEqual(info['final_observation'][0]['physics_timestamp_s'].item(), 1/240)
        self.assertAlmostEqual(obs['physics_timestamp_s'][0].item(), 1/30)
        self.assertTrue(bool(info['newly_done'].all()))
        self.assertFalse(bool(reward.any()))
        _, _, _, _, info = wrapped.step(torch.zeros(2, 7))
        self.assertFalse(bool(info['newly_done'].any()))
        self.assertEqual(info['final_observation'], {})

    def test_partial_reset_and_external_advance_detection(self):
        wrapped, low, scene, controller = self.make_env()
        wrapped.reset()
        wrapped.step(torch.zeros(2, 7))
        before = low.get_observations()
        after, _ = wrapped.reset([0])
        self.assertEqual(after['episode_id'].tolist(), [2, 1])
        self.assertEqual(after['episode_time_s'][1], before['episode_time_s'][1])
        low.step(low.last_action)
        with self.assertRaises(RuntimeError):
            wrapped.step(torch.zeros(2, 7))


if __name__ == '__main__':
    unittest.main()
