"""Expert snapshot contract: simulator-state access deliberately raises (CPU test)."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from scipy.spatial.transform import Rotation

import env_config as cfg
from env_expert import GraspValidation


class ForbiddenState:
    @property
    def data(self):
        raise AssertionError('Expert bypassed the shared observation')


def observation():
    q = torch.zeros(2, 8)
    q[:, 1:3] = .5
    rotation = (Rotation.from_euler('z', cfg.GRASP_YAW_DEG, degrees=True)
                * Rotation.from_euler('x', 180, degrees=True)).as_quat()
    ee = torch.tensor(np.array([np.r_[[.3, -.08, .14], rotation], np.r_[[.31, -.08, .14], rotation]]), dtype=torch.float32)
    return dict(joint_position=q[:, :6].clone(), gripper_opening=torch.tensor([[.06], [.06]]),
        ee_pose_base=ee, episode_id=torch.ones(2, dtype=torch.long),
        physics_timestamp_s=torch.zeros(2, dtype=torch.float64),
        diagnostics=dict(joint_pos=q.clone(), joint_vel=q * 0, joint_pos_target=q.clone(),
            applied_torque=q * 0 + .1, computed_torque=q * 0 + .1),
        privileged=dict(peg_pose_base=torch.tensor([[.3, -.08, .025, 0, 0, 0, 1], [.31, -.08, .025, 0, 0, 0, 1]]),
            peg_linear_velocity=torch.zeros(2, 3), hole_position_base=torch.tensor([[.265, .065, 0], [.275, .065, 0]])))


class ExpertObservationTests(unittest.TestCase):
    def test_batched_host_snapshot_preserves_dtypes_shapes_and_independence(self):
        from env_expert import ParallelExpert
        expert = ParallelExpert.__new__(ParallelExpert)
        expert.shared_observation_mode = True
        expert.experts = [SimpleNamespace(finished=False, set_observation=lambda obs: None)]
        source = dict(q=torch.arange(12, dtype=torch.float32).reshape(2, 6),
            nested=dict(time=torch.tensor([.1234567890123, .2], dtype=torch.float64),
                ids=torch.tensor([1, 2], dtype=torch.int64)))
        expert._accept_observation(source)
        for key in ('time', 'ids'):
            torch.testing.assert_close(expert.observation['nested'][key], source['nested'][key])
        source['q'].zero_()
        self.assertEqual(expert.observation['q'][1, 5], 11.)
        previous = expert.observation
        expert._accept_observation(source)
        self.assertIs(previous, expert.observation)

    def robot(self):
        def forbidden(*args, **kwargs):
            raise AssertionError('Expert used simulator read/write instead of snapshot/actions')
        return SimpleNamespace(articulation=ForbiddenState(),
            config=SimpleNamespace(cumotion_dir=cfg.PROJECT_ROOT / 'assets/yam_robot_with_gripper/config/cumotion',
                urdf_filename='yam.urdf', initial_joint_positions={f'joint{i+1}': value for i, value in enumerate([0, .5, .5, 0, 0, 0])},
                ee_position_in_gripper_m=np.array([0., 0., .14472]), arm_effort_limit_sim=10.),
            arm_joint_indices=list(range(6)), gripper_joint_indices=[6, 7],
            get_ee_pose_base=forbidden, get_gripper_opening=forbidden,
            set_joint_position_target=forbidden, set_feedforward_effort_target=forbidden,
            opening_to_joint_position=lambda opening: -.5 * (opening + .001039))

    def test_constructor_command_and_observe_use_only_selected_snapshot(self):
        obs = observation()
        with tempfile.TemporaryDirectory() as directory, patch('builtins.print'):
            expert = GraspValidation(self.robot(), ForbiddenState(), env_id=1, observation=obs,
                output_dir=Path(directory), workpiece_position=[.275, .065, 0], defer_commands=True)
            expert.command(1 / 240)
            expert.observe(1 / 240)
            self.assertEqual(expert.observation_source, 'shared_env')
            self.assertEqual(len(expert.pending_joint_command), 8)
            np.testing.assert_allclose(expert.stages[0][1], [.31, -.08, .14], atol=1e-7)
            np.testing.assert_allclose(expert._peg_position(), [.31, -.08, .025], atol=1e-7)
            self.assertAlmostEqual(expert._opening(), .06, places=7)
            newer = copy.deepcopy(obs)
            newer['physics_timestamp_s'] += 1 / 240
            expert.set_observation(newer)
            with self.assertRaises(ValueError):
                expert.set_observation(obs)
            wrong_episode = copy.deepcopy(newer)
            wrong_episode['episode_id'][1] = 2
            with self.assertRaises(ValueError):
                expert.set_observation(wrong_episode)

    def test_shared_snapshot_cannot_enable_direct_simulator_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                GraspValidation(self.robot(), ForbiddenState(), observation=observation(),
                    output_dir=Path(directory), defer_commands=False)


if __name__ == '__main__':
    unittest.main()
