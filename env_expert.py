"""One episode's pick-and-insert state, optionally selecting a batched scene row."""

import env_config as cfg
from env_task import peg_geometry, release_ready, geometry_diagnostic_thresholds
import numpy as np
import xml.etree.ElementTree as ET
import json
import time
from pathlib import Path
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from env_yam import tensor


class GraspValidation:
    """Grasp, lift, transfer, align, and insert a peg with bounded stages."""

    def __init__(self, robot, peg, clearance_mm=0.5, yaw_deg=cfg.GRASP_YAW_DEG,
                 preload_mm=cfg.GRASP_PRELOAD_MM, hole_radius_m=None, *, env_id=0,
                 env_origin=None, output_dir=None, workpiece_position=None, defer_commands=False,
                 observation=None):
        self.env_id = int(env_id)
        self.env_origin = np.zeros(3) if env_origin is None else np.asarray(env_origin, dtype=float)
        self.output_dir = Path(cfg.OUTPUT_DIR if output_dir is None else output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.workpiece_position = np.asarray(cfg.WORKPIECE_POSITION_M if workpiece_position is None else workpiece_position, dtype=float)
        self.defer_commands = defer_commands
        self.pending_joint_command = None
        self.observation = observation
        self.observation_source = "shared_env" if observation is not None else "legacy_simulator"
        if observation is not None and not defer_commands:
            raise ValueError("Shared-observation expert must defer all simulator writes")
        self.robot = robot
        self.peg = peg
        self.time = 0.0
        self.stage_time = 0.0
        self.stage_index = 0
        self.finished = False
        self.stable_time = 0.0
        self.close_stable_time = 0.0
        self.stage_wall_start = time.perf_counter()
        self.stage_results = []
        self.arm_peak_estimated_effort_nm = np.zeros(6)
        self.arm_peak_requested_effort_nm = np.zeros(6)
        self.arm_saturation_time_s = np.zeros(6)
        self.arm_peak_joint_tracking_error_rad = np.zeros(6)
        self.last_log_time = -cfg.TRACK_LOG_PERIOD_S
        self.hold_reference = None
        self.max_hold_slip = 0.0
        self.grasp_success = False
        self.insertion_success = False
        self.release_wait_time = 0.0
        self.released_to_bottom = False
        self.preinsertion_xy_error_m = None
        self.preinsertion_sample_time_s = None
        self.tip_in_ee = None
        self.last_insertion_ik_time = -1.0
        self.last_alignment_time = -1.0
        self.last_transfer_ik_time = -1.0
        self.transfer_start_position = None
        self.transfer_goal_position = None
        self.transfer_duration_s = 0.0
        self.align_start_position = None
        self.align_goal_position = None
        self.align_approach_duration_s = 0.0
        self.last_align_approach_ik_time = -1.0
        self.alignment_feedback_enabled = False
        self.measured_tip = None
        self.measured_axis = None
        self.insertion_depth_m = 0.0
        self.max_insertion_radial_violation_m = 0.0
        self.max_insertion_bottom_penetration_m = 0.0
        self.axis_xy_error_m = None
        self.hole_radius_m = cfg.HOLE_DIAMETER_MM / 2000.0 if hole_radius_m is None else hole_radius_m
        self.clearance_m = self.hole_radius_m - (cfg.HOLE_DIAMETER_MM - 2.0 * clearance_mm) / 2000.0
        if self.clearance_m <= 0.0:
            raise ValueError("Peg does not fit inside the STL aperture.")
        self.trace_path = self.output_dir / "pick_insert_trace.jsonl"
        self.trace_path.write_text("", encoding="utf-8")
        if cfg.RUN_INSERTION:
            (self.output_dir / "pick_insert_result.json").write_text(
                json.dumps(dict(status="running", task_success=None)), encoding="utf-8")
        self.last_trace_time = -cfg.TRACE_PERIOD_S
        self.hole_top = self.workpiece_position.copy()
        self.hole_top[2] += cfg.WORKPIECE_HEIGHT_M
        self.orientation = (Rotation.from_euler("z", yaw_deg, degrees=True)
                            * Rotation.from_euler("x", 180, degrees=True))
        self.peg_diameter = (cfg.HOLE_DIAMETER_MM - 2.0 * clearance_mm) / 1000.0
        self.yaw_deg = yaw_deg
        self.preload_mm = preload_mm
        if not 0.0 <= preload_mm < self.peg_diameter * 1000.0:
            raise ValueError("Grasp preload must be nonnegative and smaller than peg diameter.")
        closing_opening = self.peg_diameter - preload_mm / 1000.0
        initial_peg = self._peg_position() if observation is not None else np.asarray(cfg.PEG_POSITION_M)
        self.stages = [
            ("approach", np.array([*initial_peg[:2], cfg.GRASP_APPROACH_Z_M]), cfg.APPROACH_OPENING_M),
            ("descend", np.array([*initial_peg[:2], cfg.GRASP_HEIGHT_M]), cfg.APPROACH_OPENING_M),
            ("close", np.array([*initial_peg[:2], cfg.GRASP_HEIGHT_M]), closing_opening),
            ("lift", np.array([*initial_peg[:2], cfg.GRASP_APPROACH_Z_M]), closing_opening),
            ("hold", np.array([*initial_peg[:2], cfg.GRASP_APPROACH_Z_M]), closing_opening),
        ]
        root = ET.parse(robot.config.cumotion_dir / robot.config.urdf_filename).getroot()
        self.joints = [root.find(f"joint[@name='joint{i}']") for i in range(1, 7)]
        self.lower = np.array([float(j.find('limit').get('lower')) for j in self.joints])
        self.upper = np.array([float(j.find('limit').get('upper')) for j in self.joints])
        self.commanded_q = np.array([robot.config.initial_joint_positions[f"joint{i}"] for i in range(1, 7)])
        if observation is not None:
            self.commanded_q = observation["joint_position"][self.env_id].detach().cpu().numpy().copy()
        self.target_q = self.solve_ik(self.stages[0][1])
        print("[EXPERT] Starting approach", flush=True)

    def set_observation(self, observation):
        if self.observation_source != "shared_env":
            raise ValueError("Legacy expert cannot silently switch observation sources")
        if observation is None:
            raise ValueError("Shared observation is required")
        if getattr(self, "observation", None) is not None:
            previous_id = int(self.observation["episode_id"][self.env_id].item())
            current_id = int(observation["episode_id"][self.env_id].item())
            previous_time = float(self.observation["physics_timestamp_s"][self.env_id].item())
            current_time = float(observation["physics_timestamp_s"][self.env_id].item())
            if current_id != previous_id or current_time < previous_time:
                raise ValueError("Episode mismatch or observation moved backwards; reset the expert explicitly")
        self.observation = observation
        self.workpiece_position = observation["privileged"]["hole_position_base"][self.env_id].detach().cpu().numpy().copy()
        self.hole_top = self.workpiece_position.copy()
        self.hole_top[2] += cfg.WORKPIECE_HEIGHT_M

    def _joint_values(self, name, ids=None):
        if getattr(self, "observation", None) is not None:
            value = self.observation["diagnostics"][name][self.env_id]
        else:
            raw = getattr(self.robot.articulation.data, name)
            value = (raw.torch if hasattr(raw, "torch") else raw)[self.env_id]
        return value if ids is None else value[ids]

    def _peg_position(self):
        if getattr(self, "observation", None) is not None:
            return self.observation["privileged"]["peg_pose_base"][self.env_id, :3].detach().cpu().numpy()
        raw = self.peg.data.root_pos_w
        value = raw.torch if hasattr(raw, "torch") else raw
        return value[self.env_id].detach().cpu().numpy() - self.env_origin

    def _peg_axis(self):
        if getattr(self, "observation", None) is not None:
            value = self.observation["privileged"]["peg_pose_base"][self.env_id, 3:]
        else:
            raw = self.peg.data.root_quat_w
            value = (raw.torch if hasattr(raw, "torch") else raw)[self.env_id][[1, 2, 3, 0]]
        return Rotation.from_quat(value.detach().cpu().numpy()).apply([0., 0., 1.])

    def _peg_speed(self):
        if getattr(self, "observation", None) is not None:
            value = self.observation["privileged"]["peg_linear_velocity"][self.env_id]
        else:
            raw = self.peg.data.root_lin_vel_w
            value = (raw.torch if hasattr(raw, "torch") else raw)[self.env_id]
        return float(value.norm().item())

    def _ee_pose(self):
        if getattr(self, "observation", None) is not None:
            pose = self.observation["ee_pose_base"][self.env_id].detach().cpu().numpy()
            return pose[:3].copy(), Rotation.from_quat(pose[3:])
        return self.robot.get_ee_pose_base() if self.env_id == 0 else self.robot.get_ee_pose_base(self.env_id)

    def _opening(self):
        if getattr(self, "observation", None) is not None:
            return float(self.observation["gripper_opening"][self.env_id, 0].item())
        return self.robot.get_gripper_opening() if self.env_id == 0 else self.robot.get_gripper_opening(self.env_id)

    def fk(self, q):
        transform = np.eye(4)
        for joint, angle in zip(self.joints, q):
            origin = joint.find("origin")
            rotation = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy"), sep=" "))
            axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
            local = np.eye(4)
            local[:3, :3] = (rotation * Rotation.from_rotvec(axis * angle)).as_matrix()
            local[:3, 3] = np.fromstring(origin.get("xyz"), sep=" ")
            transform = transform @ local
        return (transform[:3, 3] + transform[:3, :3] @ self.robot.config.ee_position_in_gripper_m,
                Rotation.from_matrix(transform[:3, :3]))

    def solve_ik(self, target, warm_start_only=False):
        def residual(q):
            position, orientation = self.fk(q)
            return np.r_[position - target, 0.1 * (orientation.inv() * self.orientation).as_rotvec()]
        seeds = [self.commanded_q, np.array([-0.27, 1.72, 1.57, -1.42, 0.0, 1.3]),
                 np.array([-0.27, 1.72, 1.57, -1.42, 0.0, -0.27])]
        if warm_start_only:
            seeds = seeds[:1]
        solutions = [least_squares(residual, np.clip(seed, self.lower + 1e-6, self.upper - 1e-6),
                                   bounds=(self.lower, self.upper), max_nfev=250) for seed in seeds]
        solution = min(solutions, key=lambda result: np.linalg.norm(result.fun))
        if np.linalg.norm(solution.fun) > 0.0001:
            raise RuntimeError(f"IK failed for {target}: {solution.fun}")
        return solution.x

    @staticmethod
    def _cartesian_duration(distance, speed, acceleration):
        return max(cfg.TRANSFER_IK_PERIOD_S, 1.875 * distance / speed,
                   np.sqrt((10.0 / np.sqrt(3.0)) * distance / acceleration))

    def command(self, dt):
        self.time += dt
        name, _, opening = self.stages[self.stage_index]
        if name == "transfer" and self.stage_time - self.last_transfer_ik_time >= cfg.TRANSFER_IK_PERIOD_S:
            u = np.clip(self.stage_time / self.transfer_duration_s, 0.0, 1.0)
            blend = 10 * u**3 - 15 * u**4 + 6 * u**5
            ee_target = self.transfer_start_position + blend * (self.transfer_goal_position - self.transfer_start_position)
            self.target_q = self.solve_ik(ee_target, warm_start_only=True)
            self.stages[self.stage_index] = (name, ee_target, opening)
            self.last_transfer_ik_time = self.stage_time
        if name == "align" and not self.alignment_feedback_enabled and self.stage_time - self.last_align_approach_ik_time >= cfg.TRANSFER_IK_PERIOD_S:
            u = np.clip(self.stage_time / self.align_approach_duration_s, 0.0, 1.0)
            blend = 10 * u**3 - 15 * u**4 + 6 * u**5
            ee_target = self.align_start_position + blend * (self.align_goal_position - self.align_start_position)
            self.target_q = self.solve_ik(ee_target, warm_start_only=True)
            self.stages[self.stage_index] = (name, ee_target, opening)
            self.last_align_approach_ik_time = self.stage_time
        if name == "align" and self.alignment_feedback_enabled and self.stage_time - self.last_alignment_time >= cfg.ALIGN_FEEDBACK_PERIOD_S:
            actual_position, actual_rotation = self._ee_pose()
            tip_relative = actual_rotation.inv().apply(self.measured_tip - actual_position)
            correction = Rotation.align_vectors(np.array([[0.0, 0.0, 1.0]]), self.measured_axis[None, :])[0]
            previous_rotation = self.orientation
            # Increment the existing command from measured peg error. Blending
            # absolute poses leaks away the correction for steady tracking bias.
            self.orientation = Rotation.from_rotvec(cfg.ALIGN_FEEDBACK_GAIN * correction.as_rotvec()) * previous_rotation
            tip_target = self.hole_top + np.array([0.0, 0.0, cfg.PREINSERT_TIP_GAP_M])
            previous_target = self.stages[self.stage_index][1]
            rotation_shift = self.orientation.apply(tip_relative) - previous_rotation.apply(tip_relative)
            ee_target = previous_target + cfg.ALIGN_FEEDBACK_GAIN * (tip_target - self.measured_tip) - rotation_shift
            self.target_q = self.solve_ik(ee_target, warm_start_only=True)
            self.stages[self.stage_index] = (name, ee_target, opening)
            self.last_alignment_time = self.stage_time
        if name == "insert" and self.stage_time - self.last_insertion_ik_time >= cfg.INSERTION_IK_PERIOD_S:
            travel = min(cfg.INSERTION_SPEED_M_S * self.stage_time,
                         cfg.PREINSERT_TIP_GAP_M + cfg.INSERTION_COMMAND_DEPTH_M)
            tip_target = self.hole_top + np.array([0.0, 0.0, cfg.PREINSERT_TIP_GAP_M - travel])
            ee_target = tip_target - self.orientation.apply(self.tip_in_ee)
            self.target_q = self.solve_ik(ee_target, warm_start_only=True)
            self.stages[self.stage_index] = (name, ee_target, opening)
            self.last_insertion_ik_time = self.stage_time
        self.commanded_q += np.clip(self.target_q - self.commanded_q, -cfg.ARM_COMMAND_SPEED_RAD_S * dt, cfg.ARM_COMMAND_SPEED_RAD_S * dt)
        command = {f"joint{i+1}": float(q) for i, q in enumerate(self.commanded_q)}
        finger_target = self.robot.opening_to_joint_position(opening)
        command.update(joint7=finger_target, joint8=finger_target)
        self.pending_joint_command = command
        if self.defer_commands:
            return
        self.robot.set_joint_position_target(command)
        articulation = self.robot.articulation
        from env_yam import gravity_compensation
        compensation = gravity_compensation(articulation).clone()
        if not cfg.GRAVITY_COMPENSATION_ENABLED:
            compensation.zero_()
        compensation[:, self.robot.gripper_joint_indices] = 0.0
        self.robot.set_feedforward_effort_target(compensation)

    def observe(self, dt):
        self.stage_time += dt
        name, target, _ = self.stages[self.stage_index]
        arm_ids = self.robot.arm_joint_indices
        estimated = self._joint_values("applied_torque", arm_ids).detach().cpu().numpy()
        requested = self._joint_values("computed_torque", arm_ids).detach().cpu().numpy()
        q_error = (self._joint_values("joint_pos_target", arm_ids) - self._joint_values("joint_pos", arm_ids)).detach().cpu().numpy()
        self.arm_peak_estimated_effort_nm = np.maximum(self.arm_peak_estimated_effort_nm, np.abs(estimated))
        self.arm_peak_requested_effort_nm = np.maximum(self.arm_peak_requested_effort_nm, np.abs(requested))
        self.arm_saturation_time_s += (np.abs(requested) >= self.robot.config.arm_effort_limit_sim) * dt
        self.arm_peak_joint_tracking_error_rad = np.maximum(self.arm_peak_joint_tracking_error_rad, np.abs(q_error))
        position, orientation = self._ee_pose()
        error = float(np.linalg.norm(position - target))
        angle = float(np.degrees((orientation.inv() * self.orientation).magnitude()))
        reached = error < cfg.POSITION_TOLERANCE_M and angle < cfg.ORIENTATION_TOLERANCE_DEG
        if name == "align" and self.stage_time >= self.align_approach_duration_s and reached:
            # Do not integrate the large approach transient into a settled-pose
            # precision correction (integral windup during the descent).
            self.alignment_feedback_enabled = True
        peg_position = self._peg_position()
        peg_axis = self._peg_axis()
        geometry = peg_geometry(peg_position, peg_axis, self.hole_top, self.hole_radius_m, self.peg_diameter)
        peg_axis, peg_tip = geometry['axis'], geometry['tip']
        tilt_deg = geometry['tilt_deg']
        self.measured_tip = peg_tip.copy()
        self.measured_axis = peg_axis.copy()
        self.insertion_depth_m = geometry['depth']
        self.axis_xy_error_m = geometry['axis_xy_error']
        geometry_inside = geometry['geometry_inside']
        radial_envelope = geometry['radial_envelope']
        peg_relative = orientation.inv().apply(peg_position - position)
        if name in {"insert", "release"} and self.insertion_depth_m > 0.0:
            self.max_insertion_radial_violation_m = max(
                self.max_insertion_radial_violation_m, max(0.0, radial_envelope - self.hole_radius_m))
            # A tilted flat cylinder reaches below its axis endpoint by r*sin(tilt).
            lowest_z = peg_tip[2] - self.peg_diameter / 2 * np.sqrt(max(0.0, 1 - peg_axis[2] ** 2))
            table_top = cfg.TABLE_POSITION_M[2] + cfg.TABLE_SIZE_M[2] / 2
            self.max_insertion_bottom_penetration_m = max(
                self.max_insertion_bottom_penetration_m, max(0.0, table_top - lowest_z))
        if self.time - self.last_trace_time >= cfg.TRACE_PERIOD_S:
            trace = dict(time_s=self.time, stage=name, target_ee_m=target.tolist(),
                         actual_ee_m=position.tolist(), tracking_error_m=error,
                         orientation_error_deg=angle, peg_center_m=peg_position.tolist(),
                         peg_axis=peg_axis.tolist(), peg_tip_m=peg_tip.tolist(),
                         axis_xy_error_m=self.axis_xy_error_m,
                         insertion_depth_m=self.insertion_depth_m, peg_tilt_deg=tilt_deg,
                         geometry_inside_hole=geometry_inside,
                         gripper_opening_m=self._opening())
            if getattr(self, 'observation', None) is not None:
                trace.update(physics_timestamp_s=float(self.observation['physics_timestamp_s'][self.env_id].item()),
                    episode_id=int(self.observation['episode_id'][self.env_id].item()))
            with self.trace_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(trace) + "\n")
            self.last_trace_time = self.time
        if self.time - self.last_log_time >= cfg.TRACK_LOG_PERIOD_S:
            print(f"[TRACK] {name}: error={error*1000:.2f} mm; angle={angle:.2f} deg; peg_z={peg_position[2]*1000:.2f} mm", flush=True)
            ids = self.robot.gripper_joint_indices
            print("[GRIP] requested=", self.stages[self.stage_index][2],
                  "target=", self._joint_values("joint_pos_target", ids).tolist(),
                  "actual=", self._joint_values("joint_pos", ids).tolist(),
                  "drive_force_estimate=", self._joint_values("applied_torque", ids).tolist(), flush=True)
            self.last_log_time = self.time
            if name in {"transfer", "align", "insert"}:
                print(f"[INSERT] axis_xy_error_mm={self.axis_xy_error_m * 1000 if self.axis_xy_error_m is not None else None}; depth_mm={self.insertion_depth_m * 1000:.3f}; tilt_deg={tilt_deg:.3f}", flush=True)
        if name in {"transfer", "align"}:
            self.max_hold_slip = max(self.max_hold_slip, float(np.linalg.norm(peg_relative - self.hold_reference)))
            if self.max_hold_slip > cfg.MAX_HOLD_SLIP_M:
                self.finish(False, f"Peg slipped during {name}")
                return
        if name == "hold":
            self.max_hold_slip = max(self.max_hold_slip, float(np.linalg.norm(peg_relative - self.hold_reference)))
            if peg_position[2] < cfg.MIN_LIFTED_PEG_CENTER_Z_M or self.max_hold_slip > cfg.MAX_HOLD_SLIP_M:
                self.finish(False, "Peg dropped or slipped during hold")
                return
        self.stable_time = self.stable_time + dt if reached else 0.0
        if self.stage_time > cfg.STAGE_TIMEOUT_S:
            print(f"[EXPERT FAIL] {name}: position_error={error:.6f} m; orientation_error={angle:.3f} deg", flush=True)
            self.finish(False, f"{name} tracking timeout")
            return
        if name == "close":
            ids = self.robot.gripper_joint_indices
            gap = self._opening()
            finger_speed = float(self._joint_values("joint_vel", ids).abs().max().item())
            drive_force = self._joint_values("applied_torque", ids)
            loaded = bool((drive_force > cfg.CLOSE_MIN_DRIVE_FORCE_N).all().item())
            settled = (
                reached and loaded
                and abs(gap - self.peg_diameter) <= cfg.CLOSE_GAP_TOLERANCE_M
                and finger_speed <= cfg.CLOSE_MAX_FINGER_SPEED_M_S
            )
            self.close_stable_time = self.close_stable_time + dt if settled else 0.0
            advance = self.close_stable_time >= cfg.CLOSE_STABLE_S
        elif name == "hold":
            advance = reached and self.stage_time >= cfg.LIFT_HOLD_S
        elif name == "align":
            aligned = (self.alignment_feedback_enabled and self.axis_xy_error_m is not None
                       and self.axis_xy_error_m <= min(cfg.ALIGN_MAX_AXIS_ERROR_M, self.clearance_m * cfg.ALIGN_CLEARANCE_FRACTION)
                       and tilt_deg <= cfg.ALIGN_MAX_TILT_DEG
                       and 0.0 < peg_tip[2] - self.hole_top[2] <= 2.0 * cfg.PREINSERT_TIP_GAP_M)
            self.close_stable_time = self.close_stable_time + dt if aligned and reached else 0.0
            advance = self.close_stable_time >= cfg.REACHED_HOLD_S
        elif name == "insert":
            descent_done = self.stage_time >= (cfg.PREINSERT_TIP_GAP_M + cfg.INSERTION_COMMAND_DEPTH_M) / cfg.INSERTION_SPEED_M_S
            at_release_pose = descent_done and geometry_inside and error <= cfg.INSERTION_RELEASE_POSITION_TOLERANCE_M
            self.close_stable_time = self.close_stable_time + dt if at_release_pose else 0.0
            advance = self.close_stable_time >= cfg.INSERTION_VERIFY_HOLD_S
        elif name == "release":
            opened = self._opening() >= cfg.GRASP_OPENING_M - cfg.CLOSE_GAP_TOLERANCE_M
            if cfg.RELEASE_MODE == "settled":
                peg_speed = self._peg_speed()
                ready = self._release_ready(opened, peg_tip[2], peg_speed, geometry_inside)
                self.release_wait_time = self.release_wait_time + dt if ready else 0.0
                advance = ready and self.release_wait_time >= cfg.RELEASE_SETTLE_HOLD_S
            else:
                self.release_wait_time = self.release_wait_time + dt if opened else 0.0
                advance = opened and self.release_wait_time >= cfg.RELEASE_WAIT_S
        elif name == "lift":
            advance = self.stable_time >= cfg.LIFT_REACHED_HOLD_S
        elif name == "transfer":
            end_reached = self.stage_time >= self.transfer_duration_s and reached
            self.close_stable_time = self.close_stable_time + dt if end_reached else 0.0
            advance = self.close_stable_time >= cfg.REACHED_HOLD_S
        else:
            advance = self.stable_time >= cfg.REACHED_HOLD_S
        if not advance:
            return
        self.stage_results.append(dict(stage=name, position_error_m=error, orientation_error_deg=angle,
                                       peg_center_m=peg_position.tolist(), peg_tilt_deg=tilt_deg,
                                       peg_tip_m=peg_tip.tolist(), axis_xy_error_m=self.axis_xy_error_m,
                                       insertion_depth_m=self.insertion_depth_m,
                                       gripper_opening_m=self._opening(),
                                       drive_force_estimate_n=[float(self._joint_values("applied_torque", [i])[0].item())
                                                               for i in self.robot.gripper_joint_indices],
                                       gripper_joints_m=[float(self._joint_values("joint_pos", [i])[0].item())
                                                         for i in self.robot.gripper_joint_indices],
                                       duration_s=self.stage_time,
                                       wall_duration_s=time.perf_counter() - self.stage_wall_start))
        if name == "lift":
            if peg_position[2] < cfg.MIN_LIFTED_PEG_CENTER_Z_M:
                self.finish(False, "Peg was not lifted")
                return
            self.hold_reference = peg_relative.copy()
        if name == "hold":
            self.grasp_success = True
            if not cfg.RUN_INSERTION:
                self.finish(True, f"Lift hold passed ({cfg.LIFT_HOLD_S:.2f} s)")
                return
            self.tip_in_ee = orientation.inv().apply(peg_tip - position)
            correction = Rotation.align_vectors(np.array([[0.0, 0.0, 1.0]]), peg_axis[None, :])[0]
            self.orientation = correction * orientation
            opening = self.stages[self.stage_index][2]
            transfer_tip = np.array([*self.hole_top[:2], peg_tip[2]])
            transfer_target = transfer_tip - self.orientation.apply(self.tip_in_ee)
            self.transfer_start_position = position.copy()
            self.transfer_goal_position = transfer_target.copy()
            distance = float(np.linalg.norm(transfer_target - position))
            self.transfer_duration_s = self._cartesian_duration(distance, cfg.TRANSFER_SPEED_M_S, cfg.TRANSFER_ACCELERATION_M_S2)
            align_tip = self.hole_top + np.array([0.0, 0.0, cfg.PREINSERT_TIP_GAP_M])
            align_target = align_tip - self.orientation.apply(self.tip_in_ee)
            self.align_goal_position = align_target.copy()
            self.stages.extend([("transfer", position.copy(), opening),
                                ("align", align_target, opening),
                                ("insert", align_target.copy(), opening),
                                ("release", align_target.copy(), cfg.GRASP_OPENING_M)])
        if name == "transfer":
            self.align_start_position = position.copy()
            distance = float(np.linalg.norm(self.align_goal_position - position))
            self.align_approach_duration_s = self._cartesian_duration(
                distance, cfg.ALIGN_APPROACH_SPEED_M_S, cfg.ALIGN_APPROACH_ACCELERATION_M_S2)
            self.stages[self.stage_index + 1] = ("align", position.copy(), self.stages[self.stage_index][2])
        if name == "align":
            # Fixed measurement condition: stabilized at 3 mm nominal tip gap,
            # immediately before insertion commands start.
            self.preinsertion_xy_error_m = self.axis_xy_error_m
            self.preinsertion_sample_time_s = self.time
            # Effective command-to-tip transform keeps insertion continuous and
            # retains the static tracking-bias correction learned by alignment.
            self.tip_in_ee = self.orientation.inv().apply(peg_tip - target)
        if name == "insert":
            self.stages[self.stage_index + 1] = ("release", target.copy(), cfg.GRASP_OPENING_M)
        if name == "release":
            peg_speed = self._peg_speed()
            table_top = cfg.TABLE_POSITION_M[2] + cfg.TABLE_SIZE_M[2] / 2.0
            self.released_to_bottom = bool(
                abs(peg_tip[2] - table_top) <= cfg.RELEASE_BOTTOM_TOLERANCE_M
                and peg_speed <= cfg.RELEASE_MAX_PEG_SPEED_M_S
            )
            self.insertion_success = bool(
                self.released_to_bottom and geometry_inside
                and self.insertion_depth_m >= cfg.INSERTION_SUCCESS_DEPTH_M
            )
            self.finish(self.insertion_success,
                        "Released peg settled at bottom and passed depth geometry checks"
                        if self.insertion_success else "Released peg failed bottom/depth/geometry checks")
            return
        self.stage_index += 1
        self.target_q = self.solve_ik(self.stages[self.stage_index][1])
        self.stage_time = 0.0
        self.stage_wall_start = time.perf_counter()
        self.stable_time = 0.0
        self.close_stable_time = 0.0
        print(f"[EXPERT] Starting {self.stages[self.stage_index][0]}", flush=True)

    def _release_ready(self, opened, tip_z, peg_speed, geometry_inside):
        return release_ready(opened, tip_z, peg_speed, geometry_inside, self.insertion_depth_m)

    def finish(self, success, reason):
        self.finished = True
        result = dict(observation_source=self.observation_source, env_id=self.env_id, env_origin_m=self.env_origin.tolist(), success=success, reason=reason, stages=self.stage_results,
                      status="completed",
                      grasp_success=self.grasp_success,
                      insertion_success=self.insertion_success,
                      task_success=self.grasp_success and self.insertion_success,
                      failure_stage=None if success else self.stages[self.stage_index][0],
                      trace_file=str(self.trace_path.name),
                      preinsertion_xy_error_m=self.preinsertion_xy_error_m,
                      preinsertion_sample_time_s=self.preinsertion_sample_time_s,
                      preinsertion_condition="Stable nominal 3 mm peg-tip gap, before first insertion command",
                      final_insertion_depth_m=self.insertion_depth_m,
                      max_insertion_radial_violation_m=self.max_insertion_radial_violation_m,
                      max_insertion_bottom_penetration_m=self.max_insertion_bottom_penetration_m,
                      final_axis_xy_error_m=self.axis_xy_error_m,
                      hole_collision_inscribed_radius_m=self.hole_radius_m,
                      effective_radial_clearance_m=self.hole_radius_m - self.peg_diameter / 2.0,
                      target_insertion_depth_m=cfg.INSERTION_SUCCESS_DEPTH_M,
                      commanded_insertion_depth_m=cfg.INSERTION_COMMAND_DEPTH_M,
                      insertion_speed_m_s=cfg.INSERTION_SPEED_M_S,
                      transfer_reference='quintic Cartesian interpolation with fixed orientation',
                      transfer_reference_duration_s=self.transfer_duration_s,
                      align_approach_reference_duration_s=self.align_approach_duration_s,
                      release_wait_s=cfg.RELEASE_WAIT_S,
                      release_mode=cfg.RELEASE_MODE,
                      release_settle_hold_s=cfg.RELEASE_SETTLE_HOLD_S,
                      approach_opening_m=cfg.APPROACH_OPENING_M,
                      arm_command_speed_rad_s=cfg.ARM_COMMAND_SPEED_RAD_S,
                      released_to_bottom=self.released_to_bottom,
                      hole_top_m=self.hole_top.tolist(),
                      max_hold_slip_m=self.max_hold_slip,
                      gravity_compensation_enabled=cfg.GRAVITY_COMPENSATION_ENABLED,
                      grasp_yaw_deg=self.yaw_deg, closing_preload_mm=self.preload_mm,
                      peg_diameter_m=self.peg_diameter,
                      gripper_force_limit_per_finger_n=self.robot.config.gripper_effort_limit_sim,
                      gripper_speed_limit_m_s=self.robot.config.gripper_velocity_limit_sim,
                      gripper_stiffness_n_m=self.robot.config.gripper_stiffness,
                      gripper_damping_n_s_m=self.robot.config.gripper_damping,
                      arm_motor_effort_limit_nm=self.robot.config.arm_effort_limit_sim,
                      arm_velocity_limit_rad_s=self.robot.config.arm_velocity_limit_sim,
                      arm_peak_estimated_effort_nm=self.arm_peak_estimated_effort_nm.tolist(),
                      arm_peak_requested_effort_nm=self.arm_peak_requested_effort_nm.tolist(),
                      arm_saturation_time_s=self.arm_saturation_time_s.tolist(),
                      arm_peak_joint_tracking_error_rad=self.arm_peak_joint_tracking_error_rad.tolist(),
                      arm_effort_note='Implicit-PD torque estimates; drive limit reserves abs(feedforward) so total motor torque is bounded. Not a contact-force measurement.',
                      finger_collision_model="proximal_convexDecomposition+nonoverlapping_planar_distal_faces",
                      physics_dt_s=cfg.PHYSICS_DT_S, control_period_s=cfg.CONTROL_PERIOD_S,
                      peg_solver_position_iterations=cfg.PEG_POSITION_ITERATIONS,
                      peg_solver_velocity_iterations=cfg.PEG_VELOCITY_ITERATIONS,
                      scene_min_position_iterations=cfg.SCENE_MIN_POSITION_ITERATIONS,
                      scene_min_velocity_iterations=cfg.SCENE_MIN_VELOCITY_ITERATIONS,
                      tgs_external_forces_every_iteration=cfg.TGS_EXTERNAL_FORCES_EVERY_ITERATION,
                      solve_articulation_contact_last=cfg.SOLVE_ARTICULATION_CONTACT_LAST,
                      object_contact_offset_m=cfg.OBJECT_CONTACT_OFFSET_M,
                      geometry_diagnostic_thresholds=geometry_diagnostic_thresholds(),
                      controller=("numerical_ik_joint_position_with_gravity_feedforward"
                                  if cfg.GRAVITY_COMPENSATION_ENABLED
                                  else "numerical_ik_joint_position"))
        path = self.output_dir / "grasp_validation_result.json"
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        if cfg.RUN_INSERTION:
            (self.output_dir / "pick_insert_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        label = "TASK" if cfg.RUN_INSERTION else "GRASP"
        print(f"[{label} {'PASS' if success else 'FAIL'}] {reason}; report={path}", flush=True)


# --- Multi-environment expert scheduling ---


class ParallelExpert:
    def __init__(self, scene, output_dir, clearance_mm=0.5, *, observation=None, flat_single_report=False):
        self.scene = scene
        self.output_dir = Path(output_dir)
        self.flat_single_report = flat_single_report
        if flat_single_report and scene.num_envs != 1:
            raise ValueError('Flat reports are only valid for a single environment')
        self.origins = scene.scene.env_origins.detach().cpu().numpy()
        self.clearance_mm = clearance_mm
        self.observation = observation
        self.shared_observation_mode = observation is not None
        self.episode_ids = [0] * scene.num_envs
        self.experts = []
        for env_id in range(scene.num_envs):
            self.experts.append(self._new_expert(env_id))
        self.target = (observation['diagnostics']['joint_pos'] if observation is not None
            else tensor(scene.robot.articulation.data.joint_pos)).clone()

    def _new_expert(self, env_id):
        folder = self.output_dir if self.flat_single_report else self.output_dir / f"env_{env_id:03d}"
        if self.episode_ids[env_id]:
            folder /= f"episode_{self.episode_ids[env_id]:03d}"
        return GraspValidation(self.scene.robot, self.scene.peg, self.clearance_mm,
            hole_radius_m=self.scene.hole_radius_m, env_id=env_id,
            env_origin=self.origins[env_id], workpiece_position=self.scene.workpiece_positions[env_id],
            output_dir=folder, defer_commands=True, observation=self.observation)

    def reset(self, env_ids, *, reset_scene=True, observation=None):
        """Start fresh episodes only in completed rows, retaining old reports.

        Hole locations stay fixed. This is not a grid relocation scheduler.
        """
        ids = [int(index) for index in env_ids]
        if len(set(ids)) != len(ids) or any(index < 0 or index >= len(self.experts) for index in ids):
            raise ValueError("Invalid or duplicate environment index")
        if any(not self.experts[index].finished for index in ids):
            raise ValueError("Finish an active episode before resetting it")
        if self.shared_observation_mode:
            if reset_scene or observation is None:
                raise ValueError('Call env.reset(ids), then expert.reset(ids, reset_scene=False, observation=obs)')
            self.observation = observation
        if reset_scene:
            self.scene.reset(ids)
        for env_id in ids:
            self.episode_ids[env_id] += 1
            self.experts[env_id] = self._new_expert(env_id)
        self.target[ids] = (observation['diagnostics']['joint_pos'][ids] if self.shared_observation_mode
            else tensor(self.scene.robot.articulation.data.joint_pos)[ids])

    @property
    def finished(self):
        return all(expert.finished for expert in self.experts)

    def _accept_observation(self, observation):
        if self.shared_observation_mode:
            if observation is None:
                raise ValueError('Shared environment observation is required')
            # Download each batched field once, not repeatedly for every row
            # and getter in the CPU expert. Keep original public snapshots intact.
            if not getattr(self, 'batch_cpu_snapshots', True):
                self._host_observation = observation
                self._last_source_observation = observation
            elif getattr(self, '_last_source_observation', None) is not observation:
                import torch
                groups = {}
                def gather(value, path=()):
                    if isinstance(value, dict):
                        for key, item in value.items():
                            gather(item, path + (key,))
                    else:
                        groups.setdefault((value.dtype, value.device), []).append((path, value.detach()))
                gather(observation)
                host = {}
                # One transfer per dtype/device group, independent of env count.
                for fields in groups.values():
                    packed = torch.cat([value.reshape(-1) for _, value in fields]).cpu()
                    offset = 0
                    for path, value in fields:
                        target = host
                        for key in path[:-1]:
                            target = target.setdefault(key, {})
                        target[path[-1]] = packed[offset:offset + value.numel()].reshape(value.shape)
                        offset += value.numel()
                self._host_observation = host
                self._last_source_observation = observation
            self.observation = self._host_observation
            for expert in self.experts:
                if not expert.finished:
                    expert.set_observation(self.observation)
        elif observation is not None:
            raise ValueError('Construct the expert with an observation to select shared mode')

    def get_actions(self, dt, *, observation=None):
        """Generate low-level joint/opening actions without writing to PhysX."""
        self._accept_observation(observation)
        return self.command(dt, apply=False)

    def command(self, dt, *, apply=True):
        if self.shared_observation_mode and apply:
            raise ValueError('Shared-mode expert cannot apply simulator commands; use env.step(actions)')
        art = self.scene.robot.articulation
        for expert in self.experts:
            if expert.finished:
                continue
            try:
                expert.command(dt)
                for name, value in expert.pending_joint_command.items():
                    self.target[expert.env_id, art.joint_names.index(name)] = value
            except (RuntimeError, ValueError) as error:
                expert.finish(False, f"Controller failed: {error}")
        # One position-target write and one gravity/drive-budget write for all
        # rows. Completed environments hold their last target until run ends.
        if not apply:
            import torch
            opening = (-2 * self.target[:, self.scene.robot.gripper_joint_indices].mean(dim=1)
                - self.scene.robot.config.gripper_contact_overlap_m).clamp(0., self.scene.robot.max_gripper_opening_m)
            return torch.cat([self.target[:, self.scene.robot.arm_joint_indices], opening[:, None]], dim=1)
        art.set_joint_position_target(target=self.target)
        from env_yam import gravity_compensation
        compensation = gravity_compensation(art).clone()
        if not cfg.GRAVITY_COMPENSATION_ENABLED:
            compensation.zero_()
        compensation[:, self.scene.robot.gripper_joint_indices] = 0.0
        self.scene.robot.set_feedforward_effort_target(compensation)

    def get_cartesian_actions(self, dt, *, observation):
        """Express the expert's issued reference as BASE deltas at policy rate.

        FK maps the planned joint command, not an observed future state. The
        Cartesian controller, not the expert, executes it through native DIK.
        """
        import numpy as np
        import torch
        from env_yam import encode_reference, bound_expert_reference, compose_target
        joint_action = self.get_actions(dt, observation=observation)
        q = joint_action[:, :6].detach().cpu().numpy()
        poses = []
        for index, expert in enumerate(self.experts):
            position, rotation = expert.fk(q[index])
            poses.append(np.r_[position, rotation.as_quat()])
        self.last_nominal_cartesian_reference = torch.as_tensor(np.asarray(poses), device=joint_action.device, dtype=joint_action.dtype)
        encoded = encode_reference(observation['ee_pose_base'].detach().cpu().numpy(), np.asarray(poses),
            joint_action[:, 6].detach().cpu().numpy())
        encoded, self.last_cartesian_reference_limited = bound_expert_reference(encoded)
        issued = compose_target(observation['ee_pose_base'].detach().cpu().numpy(), encoded)
        self.last_cartesian_reference = torch.as_tensor(issued, device=joint_action.device, dtype=joint_action.dtype)
        return torch.as_tensor(encoded, device=joint_action.device, dtype=joint_action.dtype)

    def observe(self, dt, observation=None):
        self._accept_observation(observation)
        for expert in self.experts:
            if expert.finished:
                continue
            try:
                expert.observe(dt)
            except (RuntimeError, ValueError) as error:
                expert.finish(False, f"Observation failed: {error}")

    def finish_pending(self, reason):
        for expert in self.experts:
            if not expert.finished:
                expert.finish(False, reason)
