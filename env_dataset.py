"""Wrist RGB-D acquisition and versioned episode recording.

Importable without launching Kit; simulator imports are local to WristCamera.
"""

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
import env_config as cfg
import json
from pathlib import Path
import h5py


class WristCamera:
    def __init__(self, articulation_path):
        from pxr import Gf, UsdGeom
        import isaaclab.sim as sim_utils
        from isaaclab.sensors import Camera, CameraCfg
        import re
        paths = [articulation_path] if isinstance(articulation_path, str) else list(articulation_path)
        if not paths:
            raise ValueError('At least one camera path is required')
        self.num_cameras = len(paths)
        self.camera_paths = [path + '/link1/link2/link3/link4/link5/gripper/D405/Camera' for path in paths]
        self.output_indices = None
        source = trimesh.load_mesh(cfg.CAMERA_MESH_PATH, process=False)
        for articulation_path in paths:
            gripper = articulation_path + '/link1/link2/link3/link4/link5/gripper'
            path = gripper + '/D405'
            stage = sim_utils.get_current_stage()
            mount = UsdGeom.Xform.Define(stage, path)
            mount.AddTranslateOp().Set(Gf.Vec3d(*cfg.CAMERA_POSITION_M))
            x, y, z, w = Rotation.from_quat(cfg.CAMERA_QUAT_XYZW).as_quat()
            mount.AddOrientOp().Set(Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z))))
            mesh = UsdGeom.Mesh.Define(stage, path + '/HousingAndBracket')
            mesh.CreatePointsAttr(source.vertices.astype(np.float32).tolist())
            mesh.CreateFaceVertexCountsAttr([3] * len(source.faces))
            mesh.CreateFaceVertexIndicesAttr(source.faces.flatten().tolist())
            mesh.CreateSubdivisionSchemeAttr('none')
            mesh.CreateDisplayColorAttr([Gf.Vec3f(.16, .16, .18)])
        camera_path = paths[0] + '/link1/link2/link3/link4/link5/gripper/D405/Camera'
        if self.num_cameras > 1:
            camera_path = re.sub(r'/env_\d+/', '/env_.*/', camera_path)
            if any(re.fullmatch(camera_path, p + '/link1/link2/link3/link4/link5/gripper/D405/Camera') is None for p in paths):
                raise ValueError('Batched camera paths must share one env_<index> hierarchy')
        self.camera = Camera(CameraCfg(prim_path=camera_path,
            # The collector owns the 30 Hz schedule. A terminal substep may be
            # earlier than the next period; it must still get a fresh image.
            width=cfg.CAMERA_WIDTH, height=cfg.CAMERA_HEIGHT, update_period=0.,
            data_types=['rgb', 'distance_to_image_plane'],
            offset=CameraCfg.OffsetCfg(pos=(0., 0., 0.), rot=(0., 0., 0., 1.), convention='ros'),
            spawn=sim_utils.PinholeCameraCfg(focal_length=cfg.CAMERA_FOCAL_LENGTH_MM,
                horizontal_aperture=cfg.CAMERA_HORIZONTAL_APERTURE_MM,
                vertical_aperture=cfg.CAMERA_VERTICAL_APERTURE_MM,
                clipping_range=cfg.CAMERA_CLIPPING_M)))
        self.last_timestamp = None

    @staticmethod
    def match_camera_order(expected, actual):
        if len(actual) != len(expected) or len(set(actual)) != len(actual) or set(actual) != set(expected):
            raise RuntimeError('Camera paths do not match environment rows')
        return [actual.index(path) for path in expected]

    @staticmethod
    def numpy(value):
        raw = value.torch if hasattr(value, 'torch') else value
        return raw.detach().cpu().numpy().copy()

    def capture(self, sim, physics_timestamp_s, *, render=True):
        if self.num_cameras != 1:
            raise ValueError('Use capture_all for batched cameras')
        return self.capture_all(sim, physics_timestamp_s, render=render)[0]

    def capture_all(self, sim, physics_timestamp_s, *, render=True):
        if self.last_timestamp is not None and physics_timestamp_s < self.last_timestamp:
            raise ValueError('Camera capture timestamp moved backwards')
        elapsed = 0. if self.last_timestamp is None else physics_timestamp_s - self.last_timestamp
        if render:
            sim.render()
        self.camera.update(elapsed, force_recompute=True)
        if self.output_indices is None:
            import isaaclab.sim as sim_utils
            actual = list(self.camera._view.prim_paths)
            rendered = [str(prim.GetPath()) for prim in sim_utils.find_matching_prims(
                self.camera.cfg.prim_path, sim_utils.get_current_stage())]
            if actual != rendered:
                raise RuntimeError('Renderer and pose-view camera order differ; refusing mislabelled images')
            self.output_indices = self.match_camera_order(self.camera_paths, actual)
        rgb = self.numpy(self.camera.data.output['rgb'])[..., :3].astype(np.uint8)
        depth = self.numpy(self.camera.data.output['distance_to_image_plane'])[..., 0].astype(np.float32)
        valid = np.isfinite(depth) & (depth > 0)
        depth[~valid] = 0.
        self.last_timestamp = physics_timestamp_s
        frames = self.numpy(self.camera.frame)
        return [dict(rgb=rgb[i], depth=depth[i], depth_valid=valid[i].astype(np.uint8),
            timestamp_s=float(physics_timestamp_s), frame_id=int(frames[i])) for i in self.output_indices]

    def metadata(self, env_id=0):
        return dict(model='ideal_rendered_D405', noise_model='none', depth_type='optical_axis_z_metres',
            timestamp_source='physics time at explicit forced-render capture; not hardware exposure time',
            width=cfg.CAMERA_WIDTH, height=cfg.CAMERA_HEIGHT,
            prim_path=self.camera_paths[env_id], sensor_output_index=self.output_indices[env_id],
            intrinsic_matrix=self.numpy(self.camera.data.intrinsic_matrices)[self.output_indices[env_id]].tolist(),
            position_in_gripper_m=list(cfg.CAMERA_POSITION_M),
            quaternion_in_gripper_xyzw=list(cfg.CAMERA_QUAT_XYZW))


# --- HDF5 episode validation and persistence ---


def array(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


class EpisodeRecorder:
    SCHEMA = 'yam_cartesian_rgbd_pilot_v1'

    def __init__(self, path, metadata, *, compression='gzip', buffer_rows=16):
        if compression not in ('gzip', 'lzf', None) or buffer_rows < 1:
            raise ValueError('Invalid writer compression/buffer size')
        self.compression = compression
        self.servo_period = metadata.get('servo_period_s', 1/240)
        self.buffer_rows = buffer_rows
        self.pending = {}
        self.counts = {}
        self.path = Path(path)
        self.partial = self.path.with_suffix('.partial.hdf5')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() or self.partial.exists():
            raise FileExistsError('Episode output exists; never overwrite a previous episode')
        self.file = h5py.File(self.partial, 'x')
        self.file.attrs.update(schema_version=self.SCHEMA, complete=False,
            metadata_json=json.dumps(metadata), coordinate_frame='BASE', quaternion_order='XYZW')
        self.steps = 0
        self.last_timestamp = None
        self.last_camera_frame = None
        self.episode_id = metadata.get('episode_id')
        self.last_episode_time = None

    def _append(self, name, value):
        value = np.asarray(value).copy()
        self.pending.setdefault(name, []).append(value)
        self.counts[name] = self.counts.get(name, 0) + 1
        if len(self.pending[name]) >= self.buffer_rows:
            self._flush_field(name)

    def _flush_field(self, name):
        rows = self.pending.get(name, [])
        if not rows:
            return
        values = np.stack(rows)
        value = rows[0]
        if name not in self.file:
            image = name.startswith('obs/images/')
            self.file.create_dataset(name, shape=(0, *value.shape), maxshape=(None, *value.shape),
                dtype=value.dtype, chunks=(1, *value.shape) if image else True,
                compression=self.compression if image else None,
                compression_opts=1 if image and self.compression == 'gzip' else None)
        data = self.file[name]
        start = data.shape[0]
        data.resize(start + len(rows), axis=0)
        data[start:] = values
        self.pending[name] = []

    def flush(self):
        for name in list(self.pending):
            self._flush_field(name)
        self.file.flush()

    def _validate_observation(self, obs, frame):
        timestamp = float(array(obs['physics_timestamp_s']))
        episode_time = float(array(obs['episode_time_s']))
        if not np.isfinite([timestamp, episode_time]).all() or episode_time < 0:
            raise ValueError('Invalid observation time')
        if 'episode_id' in obs and self.episode_id is not None and int(array(obs['episode_id'])) != self.episode_id:
            raise ValueError('Episode changed without closing its recorder')
        if self.last_episode_time is not None and (episode_time <= self.last_episode_time
            or abs((episode_time - self.last_episode_time) - (timestamp - self.last_timestamp)) > 1e-8):
            raise ValueError('Episode and physics clocks are misaligned')
        if self.last_timestamp is not None and timestamp <= self.last_timestamp:
            raise ValueError('Observation timestamps must strictly increase within an episode')
        if abs(timestamp - frame['timestamp_s']) > 1e-9:
            raise ValueError('Image and robot observation timestamps differ')
        if self.last_camera_frame is not None and frame['frame_id'] <= self.last_camera_frame:
            raise ValueError('Repeated/stale camera frame')
        rgb, depth, valid = frame['rgb'], frame['depth'], frame['depth_valid']
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError('RGB must be HxWx3 uint8')
        if depth.dtype != np.float32 or depth.shape != rgb.shape[:2] or valid.shape != depth.shape or valid.dtype != np.uint8:
            raise ValueError('Depth/mask must align with RGB, depth float32 metres')
        if not np.isfinite(depth).all() or (depth < 0).any():
            raise ValueError('Invalid depths must be zero with a false validity mask')
        if not np.isin(valid, [0, 1]).all() or (depth[valid == 0] != 0).any() or (depth[valid == 1] <= 0).any():
            raise ValueError('Depth validity mask does not match stored metric depth')
        for key, shape in (('joint_position', (6,)), ('joint_velocity', (6,)),
                           ('gripper_opening', (1,)), ('ee_pose_base', (7,))):
            value = array(obs[key])
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError('Invalid observation field: ' + key)
        return timestamp

    def _observation(self, obs, frame):
        timestamp = self._validate_observation(obs, frame)
        for key in ('joint_position', 'joint_velocity', 'gripper_opening', 'ee_pose_base'):
            self._append('obs/' + key, array(obs[key]).astype(np.float32))
        for key in ('peg_pose_base', 'peg_linear_velocity', 'hole_position_base'):
            self._append('privileged/' + key, array(obs['privileged'][key]).astype(np.float32))
        for key, value in obs.get('diagnostics', {}).items():
            self._append('diagnostics/' + key, array(value).astype(np.float32))
        for key in ('rgb', 'depth', 'depth_valid'):
            self._append('obs/images/' + key, frame[key])
        self._append('time/observation_s', np.float64(timestamp))
        self._append('time/episode_s', np.float64(array(obs['episode_time_s'])))
        self._append('time/camera_s', np.float64(frame['timestamp_s']))
        self._append('time/camera_frame', np.int64(frame['frame_id']))
        self.last_timestamp = timestamp
        self.last_camera_frame = frame['frame_id']
        self.last_episode_time = float(array(obs['episode_time_s']))
        if 'episode_id' in obs:
            self.episode_id = int(array(obs['episode_id']))

    def begin(self, obs, frame):
        if self.last_timestamp is not None:
            raise RuntimeError('Episode already started')
        self._observation(obs, frame)

    def append(self, action, commanded_pose, obs_next, frame_next, *, servo_actions, servo_timestamps,
               servo_joint_positions, terminated, truncated, reward, servo_measurement_timestamps=None):
        if self.last_timestamp is None:
            raise RuntimeError('begin() before append()')
        self._validate_observation(obs_next, frame_next)
        action, commanded_pose = array(action), array(commanded_pose)
        servo_actions, servo_timestamps = array(servo_actions), array(servo_timestamps)
        servo_joint_positions = array(servo_joint_positions)
        measured_times = (servo_timestamps + self.servo_period if servo_measurement_timestamps is None
            else array(servo_measurement_timestamps))
        if action.shape != (7,) or commanded_pose.shape != (7,) or not np.isfinite(action).all() or not np.isfinite(commanded_pose).all():
            raise ValueError('Invalid policy action/commanded pose')
        if servo_actions.shape != (len(servo_timestamps), 7) or servo_joint_positions.shape != (len(servo_timestamps), 6):
            raise ValueError('Servo arrays are misaligned')
        if not all(np.isfinite(value).all() for value in (servo_actions, servo_timestamps, servo_joint_positions)):
            raise ValueError('Servo arrays must be finite')
        end = float(array(obs_next['physics_timestamp_s']))
        if len(servo_timestamps) < 1 or servo_timestamps[0] != self.last_timestamp or (np.diff(servo_timestamps) <= 0).any() or servo_timestamps[-1] >= end:
            raise ValueError('Servo commands must span this transition, before its next observation')
        if measured_times.shape != servo_timestamps.shape or not np.isfinite(measured_times).all() or (measured_times > end + 1e-9).any() or (measured_times <= servo_timestamps).any():
            raise ValueError('Servo measurement timestamps must fall inside their recorded transition')
        offset = self.counts.get('servo/joint_actions', 0)
        self._append('actions/cartesian', action.astype(np.float32))
        self._append('commands/ee_target_base', commanded_pose.astype(np.float32))
        self._append('time/action_start_s', np.float64(self.last_timestamp))
        self._append('time/action_end_s', np.float64(end))
        self._append('terminated', np.bool_(terminated))
        self._append('truncated', np.bool_(truncated))
        self._append('rewards', np.float32(reward))
        self._append('servo/start_index', np.int64(offset))
        self._append('servo/count', np.int64(len(servo_timestamps)))
        for command, timestamp, measured, measured_time in zip(servo_actions, servo_timestamps, servo_joint_positions, measured_times):
            self._append('servo/joint_actions', command.astype(np.float32))
            self._append('servo/command_timestamp_s', np.float64(timestamp))
            self._append('servo/measured_joint_position', measured.astype(np.float32))
            self._append('servo/measured_timestamp_s', np.float64(measured_time))
        self._observation(obs_next, frame_next)
        self.steps += 1

    def finish(self, *, success, reason):
        if self.steps < 1 or self.counts.get('obs/joint_position', 0) != self.steps + 1:
            raise ValueError('Expected T actions and T+1 observations')
        self.flush()
        if not bool(self.file['terminated'][-1] or self.file['truncated'][-1]):
            raise ValueError('Cannot finalize an episode without a terminal transition')
        self.file.attrs.update(complete=True, episode_success=bool(success), terminal_reason=str(reason), transitions=self.steps)
        self.file.flush()
        self.file.close()
        if self.path.exists():
            raise FileExistsError('Final episode already exists')
        self.partial.rename(self.path)

    def close_partial(self):
        if self.file.id.valid:
            self.flush()
            self.file.close()


class WristCameraBatch:
    """One render per timestamp for all wrist cameras; no physics stepping."""

    def __init__(self, articulation_paths):
        self.camera = WristCamera(articulation_paths)
        self.timestamp = None
        self.frames = None

    def capture(self, sim, timestamp, *, force=False):
        if force or self.timestamp != timestamp:
            self.frames = self.camera.capture_all(sim, timestamp)
            self.timestamp = timestamp
        return self.frames

    def metadata(self, env_id):
        return self.camera.metadata(env_id)
