"""YAM scene, joint environment and Cartesian control contract.

Sections are ordered scene -> joint step/reset -> Cartesian adapter.
Simulator-only imports occur when constructing a scene, after AppLauncher.
"""

from dataclasses import replace
import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation
import env_config as cfg
from env_task import peg_geometry, release_ready, contact_gate_pass, geometry_diagnostic_thresholds


def tensor(value):
    return value.torch if hasattr(value, "torch") else value


def get_current_stage():
    from isaacsim.core.utils.stage import get_current_stage as native_stage
    return native_stage()


def close_simulation(app):
    """Stop PhysX and detach callbacks before unloading Kit (Lab 2.1)."""
    import isaaclab.sim as sim_utils
    sim = sim_utils.SimulationContext.instance()
    try:
        if sim is not None:
            sim.stop()
            sim.clear_instance()
    finally:
        app.close(wait_for_replicator=False)


def wxyz_to_xyzw(q):
    """Isaac Lab 2.1 native quaternion -> public dataset convention."""
    return q[..., [1, 2, 3, 0]]


def xyzw_to_wxyz(q):
    """Public dataset convention -> Isaac Lab 2.1 native quaternion."""
    return q[..., [3, 0, 1, 2]]


def gravity_compensation(art):
    """Use native PhysX generalized gravity forces; validate root layout."""
    force = art.root_physx_view.get_gravity_compensation_forces()
    root_dofs = force.shape[-1] - art.num_joints
    if root_dofs not in (0, 6) or (art.is_fixed_base and root_dofs != 0):
        raise RuntimeError('Unexpected native gravity force layout')
    return force[:, root_dofs:]


class _WxyzIKBoundary:
    """Keep public XYZW actions independent of the backend controller format."""
    def __init__(self, controller):
        self.controller = controller

    def reset(self, env_ids=None):
        return self.controller.reset(env_ids)

    def set_command(self, command, *args, **kwargs):
        native = torch.cat((command[..., :3], xyzw_to_wxyz(command[..., 3:7])), dim=-1)
        return self.controller.set_command(native, *args, **kwargs)

    def compute(self, ee_pos, ee_quat, jacobian, joint_pos):
        return self.controller.compute(ee_pos, xyzw_to_wxyz(ee_quat), jacobian, joint_pos)


def create_simulation(device):
    """Pinned Lab 2.1 / Sim 4.5 backend; no Isaac Sim 6 physics options."""
    from importlib.metadata import version
    # v2.1.0 is the repository release; its extension distribution is 0.36.21.
    if not version('isaacsim').startswith('4.5.') or version('isaaclab') != '0.36.21':
        raise RuntimeError('compat/isaaclab-2.1 requires Isaac Sim 4.5.x and Isaac Lab 2.1.0; use main for Sim 6')
    import isaaclab.sim as sim_utils
    from pxr import PhysxSchema
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
        dt=cfg.PHYSICS_DT_S, device=device, enable_scene_query_support=True,
        physx=sim_utils.PhysxCfg(
            min_position_iteration_count=cfg.SCENE_MIN_POSITION_ITERATIONS,
            min_velocity_iteration_count=cfg.SCENE_MIN_VELOCITY_ITERATIONS)))
    physics = PhysxSchema.PhysxSceneAPI.Get(get_current_stage(), sim.cfg.physics_prim_path)
    for method, enabled in (
        ('CreateEnableExternalForcesEveryIterationAttr', cfg.TGS_EXTERNAL_FORCES_EVERY_ITERATION),
        ('CreateSolveArticulationContactLastAttr', cfg.SOLVE_ARTICULATION_CONTACT_LAST)):
        create = getattr(physics, method, None)
        if create is not None:
            create().Set(enabled)
        elif enabled:
            raise RuntimeError(method + ' is unsupported in this PhysX build; do not silently change contact settings')
    print('[BACKEND] Isaac Lab 2.1 / Sim 4.5; native WXYZ, public XYZW; '
          'legacy solver options are configured explicitly', flush=True)
    return sim


class ParallelYamScene:
    """Fixed workpiece per environment; reset selected robot/peg states only.

    Workpiece relocation and expert episode state will be added with the task
    scheduler. Reset does not restart SimulationContext or change other rows.
    """

    def __init__(self, sim, num_envs=4, clearance_mm=0.5, workpiece_positions=None):
        # Simulator imports stay lazy so contracts and CPU tests need no Kit app.
        from pxr import Gf, UsdGeom, UsdPhysics
        import isaaclab.sim as sim_utils
        from isaaclab.assets import RigidObject, RigidObjectCfg
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from robot_yam import YamRobot, YamRobotConfig
        if num_envs < 1 or clearance_mm not in (1.0, 0.5, 0.25, 0.1):
            raise ValueError("Invalid environment count or clearance")
        self.sim = sim
        self.workpiece_positions = np.asarray(
            [cfg.WORKPIECE_POSITION_M] * num_envs if workpiece_positions is None else workpiece_positions,
            dtype=float)
        if self.workpiece_positions.shape != (num_envs, 3) or not np.isfinite(self.workpiece_positions).all():
            raise ValueError("workpiece_positions must have shape (num_envs, 3) and be finite")
        self.heterogeneous_holes = not np.allclose(self.workpiece_positions,
            self.workpiece_positions[:1], atol=1e-12, rtol=0)
        self.scene = InteractiveScene(InteractiveSceneCfg(
            num_envs=num_envs, env_spacing=2.0, replicate_physics=False))
        source = self.scene.env_prim_paths[0]
        robot_cfg = YamRobotConfig(
            prim_path=source + "/YAM",
            finger_static_friction=cfg.CONTACT_STATIC_FRICTION,
            finger_dynamic_friction=cfg.CONTACT_DYNAMIC_FRICTION,
            gripper_effort_limit_sim=cfg.GRIPPER_FORCE_LIMIT_N,
            gripper_velocity_limit_sim=cfg.GRIPPER_VELOCITY_LIMIT_M_S,
            gripper_stiffness=cfg.GRIPPER_STIFFNESS_N_M,
            gripper_damping=cfg.GRIPPER_DAMPING_N_S_M)
        YamRobot.spawn_asset(robot_cfg)
        table = sim_utils.CuboidCfg(size=cfg.TABLE_SIZE_M,
                                    collision_props=sim_utils.CollisionPropertiesCfg())
        table.func(source + "/Table", table, translation=cfg.TABLE_POSITION_M)
        mesh_data = trimesh.load_mesh(cfg.WORKPIECE_PATH)
        if not mesh_data.is_watertight or not np.allclose(mesh_data.extents,
                [cfg.WORKPIECE_DIAMETER_M, cfg.WORKPIECE_DIAMETER_M, cfg.WORKPIECE_HEIGHT_M], atol=0.0001):
            raise ValueError("Invalid workpiece mesh dimensions or topology")
        vertices = mesh_data.vertices
        ring = vertices[(abs(vertices[:, 2] - vertices[:, 2].min()) < 1e-7)
            & (abs(np.linalg.norm(vertices[:, :2], axis=1) - cfg.HOLE_DIAMETER_MM / 2000) < 1e-6), :2]
        if len(ring) < 3:
            raise ValueError("Cannot identify STL hole aperture")
        ring = ring[np.argsort(np.arctan2(ring[:, 1], ring[:, 0]))]
        next_vertex = np.roll(ring, -1, axis=0)
        self.hole_radius_m = float(np.min(abs(ring[:, 0] * next_vertex[:, 1]
            - ring[:, 1] * next_vertex[:, 0]) / np.linalg.norm(next_vertex - ring, axis=1)))
        mesh = UsdGeom.Mesh.Define(get_current_stage(), source + "/Workpiece")
        mesh.CreatePointsAttr(mesh_data.vertices.astype(np.float32).tolist())
        mesh.CreateFaceVertexCountsAttr([3] * len(mesh_data.faces))
        mesh.CreateFaceVertexIndicesAttr(mesh_data.faces.flatten().tolist())
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDisplayColorAttr([Gf.Vec3f(*cfg.WORKPIECE_COLOR)])
        mesh.AddTranslateOp().Set(Gf.Vec3d(*self.workpiece_positions[0]))
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("none")
        sim_utils.define_collision_properties(source + "/Workpiece",
            sim_utils.CollisionPropertiesCfg(contact_offset=cfg.OBJECT_CONTACT_OFFSET_M, rest_offset=0.0))
        peg_spawn = sim_utils.CylinderCfg(
            radius=(cfg.HOLE_DIAMETER_MM - 2 * clearance_mm) / 2000,
            height=cfg.PEG_LENGTH_M,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=cfg.PEG_COLOR),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=cfg.PEG_MAX_DEPENETRATION_M_S,
                solver_position_iteration_count=cfg.PEG_POSITION_ITERATIONS,
                solver_velocity_iteration_count=cfg.PEG_VELOCITY_ITERATIONS),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=cfg.OBJECT_CONTACT_OFFSET_M, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=cfg.CONTACT_STATIC_FRICTION,
                dynamic_friction=cfg.CONTACT_DYNAMIC_FRICTION,
                restitution=cfg.CONTACT_RESTITUTION),
            mass_props=sim_utils.MassPropertiesCfg(mass=cfg.PEG_MASS_KG))
        peg_spawn.func(source + "/Peg", peg_spawn, translation=cfg.PEG_POSITION_M)
        # PhysX replication copies the prototype's internal actor transforms.
        # Different hole locations require USD clones parsed independently.
        # Articulation/peg tensors are still batched in the same simulator.
        self.scene.clone_environments(copy_from_source=True)
        # Sim 4.5 requires reset Xform stacks on nested rigid bodies. Independent
        # clones must be baked after placement, before parsing/starting PhysX.
        for path in self.scene.env_prim_paths:
            YamRobot.prepare_legacy_rigid_hierarchy(path + '/YAM')
        # Author each static hole pose before PhysX starts. Never move static
        # colliders in a running episode; task scheduling uses fixed rows.
        for path, position in zip(self.scene.env_prim_paths, self.workpiece_positions):
            get_current_stage().GetPrimAtPath(path + "/Workpiece").GetAttribute(
                "xformOp:translate").Set(Gf.Vec3d(*position))
        self.scene.filter_collisions()
        regex = self.scene.env_regex_ns
        self.robot = YamRobot(replace(robot_cfg, prim_path=regex + "/YAM"), spawn=False)
        self.peg = RigidObject(RigidObjectCfg(
            prim_path=regex + "/Peg", spawn=None,
            init_state=RigidObjectCfg.InitialStateCfg(pos=cfg.PEG_POSITION_M)))
        self.num_envs = num_envs
        self.episode_count = torch.zeros(num_envs, device=sim.device, dtype=torch.long)

    def validate_workpiece_colliders(self):
        """Query actual PhysX surfaces, not merely authored USD transforms."""
        from omni.physx import get_physx_scene_query_interface
        query = get_physx_scene_query_interface()
        origins = self.scene.env_origins.detach().cpu().numpy()
        checks = []
        for env_id, (origin, position) in enumerate(zip(origins, self.workpiece_positions)):
            point = origin + position + [.01, 0., cfg.WORKPIECE_HEIGHT_M + .01]
            hit = query.raycast_closest(tuple(point), (0., 0., -1.), .02)
            expected = self.scene.env_prim_paths[env_id] + "/Workpiece"
            passed = bool(hit.get("hit", False) and str(hit.get("collision", "")) == expected
                and abs(hit["position"][2] - (origin[2] + position[2] + cfg.WORKPIECE_HEIGHT_M)) < 1e-5)
            checks.append(dict(env_id=env_id, passed=passed,
                expected_collider=expected, actual_collider=str(hit.get("collision", "")),
                hit_position=list(hit["position"]) if hit.get("hit") else None))
        return checks

    def reset(self, env_ids=None):
        """Reset explicit rows, including velocities, targets and effort buffers."""
        ids = torch.as_tensor(list(range(self.num_envs)) if env_ids is None else env_ids,
                              device=self.sim.device, dtype=torch.long).flatten()
        if ids.numel() == 0:
            return
        if ids.unique().numel() != ids.numel() or bool(((ids < 0) | (ids >= self.num_envs)).any()):
            raise ValueError("env_ids must contain unique valid environment indices")
        art = self.robot.articulation
        q = tensor(art.data.default_joint_pos)[ids].clone()
        zero = torch.zeros_like(q)
        art.reset(env_ids=ids)
        art.write_joint_state_to_sim(position=q, velocity=zero, env_ids=ids)
        art.set_joint_position_target(target=q, env_ids=ids)
        art.set_joint_velocity_target(target=zero, env_ids=ids)
        art.set_joint_effort_target(target=zero, env_ids=ids)
        limits = torch.full_like(q, self.robot.config.gripper_effort_limit_sim)
        limits[:, self.robot.arm_joint_indices] = self.robot.config.arm_effort_limit_sim
        art.write_joint_effort_limit_to_sim(limits=limits, env_ids=ids)
        state = tensor(self.peg.data.default_root_state)[ids].clone()
        state[:, :3] += self.scene.env_origins[ids]
        self.peg.reset(env_ids=ids)
        self.peg.write_root_pose_to_sim(root_pose=state[:, :7], env_ids=ids)
        self.peg.write_root_velocity_to_sim(root_velocity=state[:, 7:], env_ids=ids)
        self.episode_count[ids] += 1

    def step(self, render=False, profiler=None):
        from contextlib import nullcontext
        with profiler.section('scene.write') if profiler else nullcontext():
            self.robot.write_data_to_sim()
            self.peg.write_data_to_sim()
        with profiler.section('scene.physics') if profiler else nullcontext():
            self.sim.step(render=render)
        with profiler.section('scene.update') if profiler else nullcontext():
            self.robot.update(self.sim.get_physics_dt())
            self.peg.update(self.sim.get_physics_dt())


# --- Joint-space environment and task lifecycle ---


class YamPickInsertEnv:
    ACTION_SCHEMA = 'joint_position_opening_v1'

    def __init__(self, scene, clearance_mm=.5, control_period_s=None, max_episode_seconds=120.):
        self.scene = scene
        self.robot = scene.robot
        self.num_envs = scene.num_envs
        self.device = scene.sim.device
        self.physics_dt = scene.sim.get_physics_dt()
        self.control_period = cfg.CONTROL_PERIOD_S if control_period_s is None else control_period_s
        self.substeps = round(self.control_period / self.physics_dt)
        if self.substeps < 1 or abs(self.substeps * self.physics_dt - self.control_period) > 1e-9:
            raise ValueError('Control period must be a positive integer number of physics steps')
        if max_episode_seconds <= 0:
            raise ValueError('Episode timeout must be positive')
        self.max_episode_seconds = max_episode_seconds
        self.peg_diameter = (cfg.HOLE_DIAMETER_MM - 2 * clearance_mm) / 1000
        self.global_steps = 0
        self.episode_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.truncated = self.terminated.clone()
        self.success = self.terminated.clone()
        self.was_lifted = np.zeros(self.num_envs, dtype=bool)
        self.settle_seconds = np.zeros(self.num_envs)
        self.max_radial_violation = np.zeros(self.num_envs)
        self.max_bottom_penetration = np.zeros(self.num_envs)
        self.reasons = [''] * self.num_envs
        self.terminal_observations = {}
        self._initialized = False
        self.last_action = torch.zeros((self.num_envs, 7), device=self.device)
        self.legacy_pose_reads = False  # paired performance reference only
        self.profile_seconds = None

    def reset(self, env_ids=None):
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else torch.as_tensor(env_ids, device=self.device)
        if ids.ndim != 1 or ids.dtype not in (torch.int32, torch.int64):
            raise ValueError('env_ids must be a one-dimensional integer list/tensor')
        ids = ids.long()
        if ids.unique().numel() != ids.numel() or bool(((ids < 0) | (ids >= self.num_envs)).any()):
            raise ValueError('Invalid or duplicate env_ids')
        if not self._initialized and ids.numel() != self.num_envs:
            raise ValueError('First reset must initialize all environments')
        self.scene.reset(ids)
        # Refresh native FK without advancing physics or touching episode clocks.
        self.scene.sim.forward()
        self.episode_steps[ids] = 0
        self.terminated[ids] = False
        self.truncated[ids] = False
        self.success[ids] = False
        for index in ids.cpu().tolist():
            self.was_lifted[index] = False
            self.settle_seconds[index] = 0.
            self.max_radial_violation[index] = 0.
            self.max_bottom_penetration[index] = 0.
            self.reasons[index] = ''
            self.terminal_observations.pop(index, None)
        q = tensor(self.robot.articulation.data.joint_pos)
        self.last_action[ids, :6] = q[ids][:, self.robot.arm_joint_indices]
        finger = q[ids][:, self.robot.gripper_joint_indices].mean(dim=1)
        self.last_action[ids, 6] = (-2 * finger - self.robot.config.gripper_contact_overlap_m).clamp(0., self.robot.max_gripper_opening_m)
        self._initialized = True
        return self.get_observations(), dict(reset_env_ids=ids.clone(), action_schema=self.ACTION_SCHEMA)

    def _validate_actions(self, actions):
        if not isinstance(actions, torch.Tensor) or actions.shape != (self.num_envs, 7):
            raise ValueError('actions must be a Tensor of shape (num_envs, 7): joint1..6 rad, total jaw opening m')
        if not actions.is_floating_point() or not bool(torch.isfinite(actions).all()):
            raise ValueError('actions must be finite floating-point values')
        actions = actions.detach().to(device=self.device, dtype=torch.float32).clone()
        limits = tensor(self.robot.articulation.data.joint_pos_limits)[:, self.robot.arm_joint_indices]
        if bool(((actions[:, :6] < limits[..., 0]) | (actions[:, :6] > limits[..., 1])).any()):
            raise ValueError('Arm action exceeds joint limits; no silent clipping')
        if bool(((actions[:, 6] < 0) | (actions[:, 6] > self.robot.max_gripper_opening_m)).any()):
            raise ValueError('Gripper action exceeds valid total opening range')
        return actions

    def _apply_actions(self, actions):
        art = self.robot.articulation
        q = tensor(art.data.joint_pos_target).clone()
        q[:, self.robot.arm_joint_indices] = actions[:, :6]
        finger = (-.5 * (actions[:, 6] + self.robot.config.gripper_contact_overlap_m)).clamp(
            self.robot.config.gripper_joint_open_m, self.robot.config.gripper_joint_closed_m)
        q[:, self.robot.gripper_joint_indices] = finger[:, None]
        art.set_joint_position_target(target=q)
        ff = gravity_compensation(art).clone()
        if not cfg.GRAVITY_COMPENSATION_ENABLED:
            ff.zero_()
        ff[:, self.robot.gripper_joint_indices] = 0.
        self.robot.set_feedforward_effort_target(ff)

    def get_observations(self):
        if not self._initialized:
            raise RuntimeError('reset() is required before observation/step')
        art = self.robot.articulation
        peg = self.scene.peg
        # Native simulator poses, not a second FK implementation. CPU pose
        # conversion is intentionally explicit; vectorization is a later change.
        origins = self.scene.scene.env_origins
        peg_pose = torch.cat([tensor(peg.data.root_pos_w) - origins,
            wxyz_to_xyzw(tensor(peg.data.root_quat_w))], dim=1)
        if not self.legacy_pose_reads and hasattr(self.robot, 'get_ee_poses_base'):
            ee = self.robot.get_ee_poses_base().detach().clone()
        else:
            poses = [self.robot.get_ee_pose_base(index) for index in range(self.num_envs)]
            ee = torch.as_tensor(np.asarray([np.r_[p, r.as_quat()] for p, r in poses]),
                device=self.device, dtype=torch.float32)
        def on_device(value):
            return torch.as_tensor(np.asarray(value), dtype=torch.float32, device=self.device).clone()
        q = tensor(art.data.joint_pos)
        opening = (-2 * q[:, self.robot.gripper_joint_indices].mean(dim=1)
            - self.robot.config.gripper_contact_overlap_m).clamp(0., self.robot.max_gripper_opening_m)
        return dict(
            joint_position=q[:, self.robot.arm_joint_indices].detach().clone(),
            joint_velocity=tensor(art.data.joint_vel)[:, self.robot.arm_joint_indices].detach().clone(),
            gripper_opening=opening[:, None].detach().clone(), ee_pose_base=ee,
            previous_action=self.last_action.clone(),
            physics_timestamp_s=torch.full((self.num_envs,), self.global_steps * self.physics_dt, device=self.device, dtype=torch.float64),
            episode_time_s=self.episode_steps.double() * self.physics_dt,
            episode_id=self.scene.episode_count.clone(),
            diagnostics={name: tensor(getattr(art.data, name)).detach().clone()
                for name in ('joint_pos', 'joint_vel', 'joint_pos_target', 'applied_torque', 'computed_torque')},
            privileged=dict(peg_pose_base=peg_pose.detach().clone(),
                peg_linear_velocity=tensor(peg.data.root_lin_vel_w).detach().clone(),
                hole_position_base=on_device(self.scene.workpiece_positions)))

    def _update_task(self):
        obs = self.get_observations()
        pose = obs['privileged']['peg_pose_base'].cpu().numpy()
        speed = obs['privileged']['peg_linear_velocity'].norm(dim=1).cpu().numpy()
        opened = obs['gripper_opening'][:, 0].cpu().numpy() >= cfg.GRASP_OPENING_M - cfg.CLOSE_GAP_TOLERANCE_M
        active = (~(self.terminated | self.truncated)).cpu().tolist()
        for index in range(self.num_envs):
            if not active[index]:
                continue
            self.was_lifted[index] |= pose[index, 2] >= cfg.MIN_LIFTED_PEG_CENTER_Z_M
            top = self.scene.workpiece_positions[index].copy()
            top[2] += cfg.WORKPIECE_HEIGHT_M
            metric = peg_geometry(pose[index, :3], Rotation.from_quat(pose[index, 3:]).apply([0, 0, 1]),
                top, self.scene.hole_radius_m, self.peg_diameter)
            if self.was_lifted[index] and metric['depth'] > 0:
                table_top = cfg.TABLE_POSITION_M[2] + cfg.TABLE_SIZE_M[2] / 2
                self.max_radial_violation[index] = max(self.max_radial_violation[index],
                    max(0., metric['radial_envelope'] - self.scene.hole_radius_m))
                self.max_bottom_penetration[index] = max(self.max_bottom_penetration[index], max(0., table_top - metric['lowest_z']))
            ready = self.was_lifted[index] and release_ready(opened[index], metric['tip'][2], speed[index],
                metric['geometry_inside'], metric['depth'])
            self.settle_seconds[index] = self.settle_seconds[index] + self.physics_dt if ready else 0.
            if ready and self.settle_seconds[index] >= cfg.RELEASE_SETTLE_HOLD_S:
                self.terminated[index] = True
                passed = contact_gate_pass(self.max_radial_violation[index], self.max_bottom_penetration[index])
                self.success[index] = passed
                self.reasons[index] = 'success' if passed else 'contact_violation'
            elif self.episode_steps[index].item() * self.physics_dt >= self.max_episode_seconds:
                self.truncated[index] = True
                self.reasons[index] = 'timeout'
        return obs

    def step(self, actions, *, render=False, physics_observer=None):
        import time
        clock = time.perf_counter
        def charge(name, start):
            if self.profile_seconds is not None:
                self.profile_seconds[name] = self.profile_seconds.get(name, 0.) + clock() - start
        if not self._initialized:
            raise RuntimeError('reset() is required before step')
        actions = self._validate_actions(actions)
        active = ~(self.terminated | self.truncated)
        before_done = ~active
        actions[~active] = self.last_action[~active]
        self.last_action = actions
        stamp = clock()
        self._apply_actions(actions)
        charge('apply_joint_and_gravity', stamp)
        for substep in range(self.substeps):
            stamp = clock()
            self.scene.step(render=render and substep == self.substeps - 1)
            charge('physics_write_step_update', stamp)
            self.global_steps += 1
            self.episode_steps[~(self.terminated | self.truncated)] += 1
            stamp = clock()
            obs = self._update_task()
            charge('observations_and_task', stamp)
            done = self.terminated | self.truncated
            for index in torch.where(done & ~before_done)[0].cpu().tolist():
                self.terminal_observations.setdefault(index, self._observation_row(obs, index))
            if physics_observer is not None:
                stamp = clock()
                physics_observer(self.physics_dt, obs)
                charge('physics_observer', stamp)
        done = self.terminated | self.truncated
        return obs, (self.success & ~before_done).float(), self.terminated.clone(), self.truncated.clone(), dict(
            success=self.success.clone(), reason=self.reasons.copy(),
            geometry_diagnostic_thresholds=geometry_diagnostic_thresholds(),
            max_radial_violation_m=self.max_radial_violation.copy(),
            max_bottom_penetration_m=self.max_bottom_penetration.copy(),
            newly_done=(done & ~before_done).clone(), action_schema=self.ACTION_SCHEMA,
            final_observation={index: value for index, value in self.terminal_observations.items()
                if bool((done & ~before_done)[index])})

    def report_failure(self, env_ids, reason):
        """Latch controller/IK failure; caller must still explicitly reset."""
        obs = self.get_observations()
        for index in env_ids:
            if bool(self.terminated[index] | self.truncated[index]):
                continue
            self.terminated[index] = True
            self.success[index] = False
            self.reasons[index] = str(reason)
            self.terminal_observations[index] = self._observation_row(obs, index)

    @staticmethod
    def _observation_row(obs, index):
        return {name: (YamPickInsertEnv._observation_row(value, index) if isinstance(value, dict)
            else value[index].clone()) for name, value in obs.items()}


# --- Cartesian policy contract and native DIK adapter ---


def compose_target(measured_pose, action):
    """p*=p(t)+dp; R*=Exp(dr_BASE) R(t). Quaternion storage is XYZW."""
    pose = np.asarray(measured_pose, dtype=float)
    action = np.asarray(action, dtype=float)
    target = pose.copy()
    target[:, :3] += action[:, :3]
    target[:, 3:] = (Rotation.from_rotvec(action[:, 3:6]) * Rotation.from_quat(pose[:, 3:])).as_quat()
    return target


def encode_reference(measured_pose, commanded_pose, opening):
    """Encode an issued reference, never the subsequently observed motion."""
    measured, commanded = np.asarray(measured_pose), np.asarray(commanded_pose)
    rotation = (Rotation.from_quat(commanded[:, 3:]) * Rotation.from_quat(measured[:, 3:]).inv()).as_rotvec()
    return np.c_[commanded[:, :3] - measured[:, :3], rotation, np.asarray(opening)]


def bound_expert_reference(action):
    """Teacher's explicit reference governor, BEFORE an action is issued/logged.

    This is not silent clipping in the environment. Both nominal and issued
    references remain distinguishable; policies must still supply valid actions.
    """
    issued = np.asarray(action, dtype=float).copy()
    limited = np.zeros(len(issued), dtype=bool)
    for block, bound in ((slice(0, 3), cfg.CARTESIAN_MAX_TRANSLATION_M),
                         (slice(3, 6), cfg.CARTESIAN_MAX_ROTATION_RAD)):
        magnitude = np.linalg.norm(issued[:, block], axis=1)
        limited |= magnitude > bound
        issued[:, block] *= np.minimum(1., bound / np.maximum(magnitude, 1e-12))[:, None]
    return issued, limited


class CartesianYamEnv:
    ACTION_SCHEMA = 'base_measured_delta_rotvec_absolute_opening_v1'

    def __init__(self, low_level_env, *, controller=None, jacobian_provider=None):
        self.env = low_level_env
        self.num_envs = low_level_env.num_envs
        self.device = low_level_env.device
        self.period = cfg.POLICY_PERIOD_S
        if not any(abs(low_level_env.control_period - 1 / hz) < 1e-12 for hz in (60, 120, 240)):
            raise ValueError('Servo rate must be 60, 120 or 240 Hz')
        self.servo_steps = round(self.period / low_level_env.control_period)
        if self.servo_steps not in (2, 4, 8) or abs(self.servo_steps * low_level_env.control_period - self.period) > 1e-12:
            raise ValueError('Expected 2, 4 or 8 servo steps per 30 Hz action')
        if controller is None:
            from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
            controller = DifferentialIKController(DifferentialIKControllerCfg(command_type='pose',
                use_relative_mode=False, ik_method='dls', ik_params={'lambda_val': cfg.CARTESIAN_IK_DAMPING}),
                self.num_envs, self.device)
            controller = _WxyzIKBoundary(controller)
        self.controller = controller
        self.jacobian_provider = jacobian_provider or self.env.robot.get_ee_jacobian_base
        self.last_joint_target = None
        self.boundary_observation = None

    def reset(self, env_ids=None):
        obs, info = self.env.reset(env_ids)
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else torch.as_tensor(env_ids, device=self.device)
        if self.last_joint_target is None:
            self.last_joint_target = obs['joint_position'].clone()
        else:
            self.last_joint_target[ids] = obs['joint_position'][ids]
        self.controller.reset(ids)
        self.boundary_observation = obs
        return obs, dict(**info, policy_action_schema=self.ACTION_SCHEMA, policy_period_s=self.period)

    def get_observations(self):
        if self.boundary_observation is None:
            raise RuntimeError('Reset the Cartesian wrapper before use')
        return self.env.get_observations()

    def _validate(self, actions):
        if not isinstance(actions, torch.Tensor) or actions.shape != (self.num_envs, 7):
            raise ValueError('Expected (num_envs, 7): dp_BASE[m], drotvec_BASE[rad], opening[m]')
        if not actions.is_floating_point() or not bool(torch.isfinite(actions).all()):
            raise ValueError('Cartesian actions must be finite floating-point values')
        actions = actions.detach().to(device=self.device, dtype=torch.float32).clone()
        if bool((actions[:, :3].norm(dim=1) > cfg.CARTESIAN_MAX_TRANSLATION_M + 1e-7).any()):
            raise ValueError('Translation delta exceeds contract bound; no silent clipping')
        if bool((actions[:, 3:6].norm(dim=1) > cfg.CARTESIAN_MAX_ROTATION_RAD + 1e-7).any()):
            raise ValueError('Rotation delta exceeds contract bound; no silent clipping')
        if bool(((actions[:, 6] < 0) | (actions[:, 6] > self.env.robot.max_gripper_opening_m)).any()):
            raise ValueError('Opening must be absolute total aperture in metres')
        return actions

    def step(self, actions, *, render=False, physics_observer=None):
        if self.boundary_observation is None:
            raise RuntimeError('reset() required')
        actions = self._validate(actions)
        obs = self.boundary_observation
        if not torch.equal(obs['episode_id'], self.env.scene.episode_count):
            raise RuntimeError('Reset through the Cartesian wrapper, not its underlying environment')
        if bool((obs['physics_timestamp_s'] != self.env.global_steps * self.env.physics_dt).any()):
            raise RuntimeError('Low-level environment advanced outside the Cartesian wrapper')
        # Freeze the absolute target ONCE at the policy boundary.
        target_array = compose_target(obs['ee_pose_base'].detach().cpu().numpy(), actions.detach().cpu().numpy())
        target = torch.as_tensor(target_array, device=self.device, dtype=torch.float32)
        self.controller.set_command(target)
        before_done = (self.env.terminated | self.env.truncated).clone()
        reward = torch.zeros(self.num_envs, device=self.device)
        final = {}
        joint_commands = []
        servo_times = []
        servo_states = []
        slew_flags = []
        limit_flags = []
        for servo in range(self.servo_steps):
            proposal = self.controller.compute(obs['ee_pose_base'][:, :3], obs['ee_pose_base'][:, 3:],
                self.jacobian_provider(), obs['joint_position'])
            limit = cfg.ARM_COMMAND_SPEED_RAD_S * self.env.control_period
            slew_flags.append((proposal - self.last_joint_target).abs() > limit)
            q_target = self.last_joint_target + (proposal - self.last_joint_target).clamp(-limit, limit)
            bounds = tensor(self.env.robot.articulation.data.joint_pos_limits)[:, self.env.robot.arm_joint_indices]
            limit_flags.append((q_target < bounds[..., 0]) | (q_target > bounds[..., 1]))
            q_target = torch.maximum(torch.minimum(q_target, bounds[..., 1]), bounds[..., 0])
            joint_action = torch.cat([q_target, actions[:, 6:]], dim=1)
            servo_times.append(obs['physics_timestamp_s'].clone())
            obs, current_reward, terminated, truncated, info = self.env.step(joint_action,
                render=render and servo == self.servo_steps - 1, physics_observer=physics_observer)
            self.last_joint_target = obs['previous_action'][:, :6].clone()
            joint_commands.append(obs['previous_action'].clone())
            servo_states.append(obs['joint_position'].clone())
            reward += current_reward
            final.update(info['final_observation'])
        self.boundary_observation = obs
        info.update(action_schema=self.ACTION_SCHEMA, policy_period_s=self.period,
            servo_steps=self.servo_steps, commanded_pose_base=target.clone(),
            policy_action=actions.clone(), servo_joint_actions=torch.stack(joint_commands),
            servo_command_timestamp_s=torch.stack(servo_times),
            servo_measured_joint_position=torch.stack(servo_states),
            servo_slew_limited=torch.stack(slew_flags), servo_joint_limit_hit=torch.stack(limit_flags),
            newly_done=((terminated | truncated) & ~before_done), final_observation=final)
        return obs, reward, terminated, truncated, info
