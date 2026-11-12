"""CPU tests for HDF5 schema, temporal alignment and crash-safe completion."""
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from env_dataset import EpisodeRecorder


def state(timestamp):
    return dict(joint_position=np.zeros(6), joint_velocity=np.zeros(6), gripper_opening=np.array([.02]),
        ee_pose_base=np.array([.3, 0., .1, 0., 0., 0., 1.]), physics_timestamp_s=timestamp,
        episode_time_s=timestamp, privileged=dict(peg_pose_base=np.array([.3, 0., .025, 0., 0., 0., 1.]),
            peg_linear_velocity=np.zeros(3), hole_position_base=np.array([.265, .065, 0.])))


def frame(timestamp, index):
    return dict(rgb=np.zeros((8, 12, 3), dtype=np.uint8), depth=np.ones((8, 12), dtype=np.float32),
        depth_valid=np.ones((8, 12), dtype=np.uint8), timestamp_s=timestamp, frame_id=index)


class RecorderTests(unittest.TestCase):
    def test_full_campaign_has_all_1156_cases_once_and_short_final_batch(self):
        from record_expert_data import campaign_plan
        from env_workspace_grid import make_manifest
        points = make_manifest()['points']
        jobs = campaign_plan(points, num_envs=4)
        cases = [(job['clearance_mm'], grid) for job in jobs for grid in job['grid_ids']]
        self.assertEqual(len(jobs), 292)
        self.assertEqual(len(cases), 1156)
        self.assertEqual(len(set(cases)), 1156)
        self.assertEqual([job['clearance_mm'] for job in jobs[:4]], [1., .5, .25, .1])
        self.assertTrue(all(len(job['grid_ids']) == 1 for job in jobs[-4:]))
        for clearance in (1., .5, .25, .1):
            self.assertEqual({grid for c, grid in cases if c == clearance}, {p['grid_id'] for p in points})

    def test_campaign_rejects_duplicate_grid_ids(self):
        from record_expert_data import campaign_plan
        with self.assertRaises(ValueError):
            campaign_plan([dict(grid_id='r00_c00'), dict(grid_id='r00_c00')])

    def test_camera_order_is_explicit_not_lexicographic_env_number(self):
        from env_dataset import WristCamera
        self.assertEqual(WristCamera.match_camera_order(['env_2', 'env_10'], ['env_10', 'env_2']), [1, 0])
        with self.assertRaises(RuntimeError):
            WristCamera.match_camera_order(['env_0', 'env_1'], ['env_0', 'env_0'])

    def test_episode_switch_and_nonfinite_servo_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = EpisodeRecorder(Path(directory) / 'invalid.hdf5', {'episode_id': 1})
            first = dict(state(0.), episode_id=1)
            writer.begin(first, frame(0., 1))
            kwargs = dict(servo_actions=np.zeros((8, 7)), servo_timestamps=np.arange(8)/240,
                servo_joint_positions=np.zeros((8, 6)), terminated=True, truncated=False, reward=1.)
            next_obs = dict(state(1/30), episode_id=2)
            with self.assertRaises(ValueError):
                writer.append(np.zeros(7), next_obs['ee_pose_base'], next_obs, frame(1/30, 2), **kwargs)
            next_obs['episode_id'] = 1
            kwargs['servo_actions'][1, 0] = np.nan
            with self.assertRaises(ValueError):
                writer.append(np.zeros(7), next_obs['ee_pose_base'], next_obs, frame(1/30, 2), **kwargs)
            self.assertEqual(writer.steps, 0)
            writer.close_partial()

    def test_partial_servo_tick_records_terminal_not_future_measurement(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'terminal.hdf5'
            writer = EpisodeRecorder(path, {'servo_period_s': 1/60})
            writer.begin(state(0.), frame(0., 1))
            kwargs = dict(servo_actions=np.zeros((1, 7)), servo_timestamps=np.array([0.]),
                servo_joint_positions=np.zeros((1, 6)), terminated=True, truncated=False, reward=1.)
            with self.assertRaises(ValueError):
                writer.append(np.zeros(7), state(1/240)['ee_pose_base'], state(1/240), frame(1/240, 2), **kwargs)
            writer.append(np.zeros(7), state(1/240)['ee_pose_base'], state(1/240), frame(1/240, 2),
                servo_measurement_timestamps=np.array([1/240]), **kwargs)
            writer.finish(success=True, reason='success')
            with h5py.File(path, 'r') as file:
                self.assertEqual(file['servo/measured_timestamp_s'][0], 1/240)

    def test_camera_batch_reuses_one_render_timestamp_but_forces_reset_refresh(self):
        from unittest.mock import patch
        from env_dataset import WristCameraBatch
        class FakeCamera:
            def __init__(self, paths):
                self.count = len(paths)
                self.renders = 0
            def capture_all(self, sim, timestamp):
                self.renders += 1
                return [frame(timestamp, self.renders) for _ in range(self.count)]
        with patch('env_dataset.WristCamera', FakeCamera):
            cameras = WristCameraBatch(['env0', 'env1'])
            first = cameras.capture(None, 0.)
            self.assertIs(cameras.capture(None, 0.), first)
            self.assertEqual(cameras.camera.renders, 1)
            self.assertEqual(len(cameras.capture(None, 0., force=True)), 2)
            self.assertEqual(cameras.camera.renders, 2)

    def test_buffered_lzf_flush_preserves_all_rows_and_servo_offsets(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'buffered.hdf5'
            writer = EpisodeRecorder(path, {}, compression='lzf', buffer_rows=4)
            writer.begin(state(0.), frame(0., 1))
            for tick in range(9):
                end = (tick + 1) / 30
                writer.append(np.full(7, tick), state(end)['ee_pose_base'], state(end), frame(end, tick+2),
                    servo_actions=np.full((8, 7), tick), servo_timestamps=tick/30 + np.arange(8)/240,
                    servo_joint_positions=np.zeros((8, 6)), terminated=tick == 8, truncated=False, reward=0.)
            writer.finish(success=True, reason='success')
            with h5py.File(path, 'r') as file:
                np.testing.assert_array_equal(file['actions/cartesian'][:, 0], np.arange(9))
                np.testing.assert_array_equal(file['servo/start_index'][:], np.arange(9)*8)
                self.assertEqual(len(file['obs/images/rgb']), 10)
                self.assertEqual(len(file['servo/joint_actions']), 72)
                self.assertEqual(file['obs/images/rgb'].compression, 'lzf')

    def test_t_plus_one_partial_final_interval_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'episode.hdf5'
            writer = EpisodeRecorder(path, {'pilot_only': True})
            writer.begin(state(0.), frame(0., 1))
            for start, end, count, index in ((0., 1/30, 8, 2), (1/30, .05, 4, 3)):
                writer.append(np.zeros(7), state(end)['ee_pose_base'], state(end), frame(end, index),
                    servo_actions=np.zeros((count, 7)), servo_timestamps=start + np.arange(count)/240,
                    servo_joint_positions=np.zeros((count, 6)), terminated=index == 3, truncated=False, reward=float(index == 3))
            writer.finish(success=True, reason='success')
            self.assertFalse(writer.partial.exists())
            with h5py.File(path, 'r') as file:
                self.assertTrue(file.attrs['complete'])
                self.assertEqual(file['actions/cartesian'].shape, (2, 7))
                self.assertEqual(file['obs/images/rgb'].shape, (3, 8, 12, 3))
                self.assertEqual(file['servo/joint_actions'].shape, (12, 7))
                np.testing.assert_array_equal(file['servo/count'][:], [8, 4])
                self.assertEqual(file['time/observation_s'].dtype, np.dtype('float64'))
                self.assertEqual(file['obs/images/depth'].dtype, np.dtype('float32'))
            with self.assertRaises(FileExistsError):
                EpisodeRecorder(path, {})

    def test_stale_or_misaligned_image_is_rejected_before_transition_write(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = EpisodeRecorder(Path(directory) / 'episode.hdf5', {})
            writer.begin(state(0.), frame(0., 1))
            for bad in (frame(.02, 2), frame(1/30, 1)):
                with self.assertRaises(ValueError):
                    writer.append(np.zeros(7), state(1/30)['ee_pose_base'], state(1/30), bad,
                        servo_actions=np.zeros((8, 7)), servo_timestamps=np.arange(8)/240,
                        servo_joint_positions=np.zeros((8, 6)), terminated=True, truncated=False, reward=1.)
                self.assertEqual(writer.steps, 0)
                self.assertNotIn('actions', writer.file)
            writer.close_partial()
            self.assertTrue(writer.partial.exists())
            self.assertFalse(writer.path.exists())
            with h5py.File(writer.partial, 'r') as file:
                self.assertFalse(file.attrs['complete'])

    def test_unterminated_episode_cannot_be_finalized(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = EpisodeRecorder(Path(directory) / 'episode.hdf5', {})
            writer.begin(state(0.), frame(0., 1))
            with self.assertRaises(ValueError):
                writer.finish(success=True, reason='not actually done')
            writer.close_partial()


if __name__ == '__main__':
    unittest.main()
