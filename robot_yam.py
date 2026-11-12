# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reusable YAM robot wrapper for Isaac Lab 2.1 joint-space and Cartesian control.

Import this module only after :class:`isaaclab.app.AppLauncher` has created the
Isaac Sim application. Cartesian control is owned by env_yam using the native DifferentialIKController.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull

import isaaclab.sim as sim_utils
from isaacsim.core.utils.stage import get_current_stage
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema


PROJECT_ROOT = Path(__file__).resolve().parent
YAM_ASSET_ROOT = PROJECT_ROOT / "assets" / "yam_robot_with_gripper"
YAM_USD_PATH = YAM_ASSET_ROOT / "usd" / "yam_with_linear_gripper.usd"


@dataclass
class YamRobotConfig:
    """Configuration used to spawn and control one YAM robot."""

    prim_path: str = "/World/YAM"
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    usd_path: Path = YAM_USD_PATH
    articulation_base_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    articulation_base_rot_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    finger_static_friction: float = 4.0
    finger_dynamic_friction: float = 3.0
    finger_restitution: float = 0.0
    # Decomposed convex shapes avoid the SDF contact failure observed on release.
    finger_collision_max_hulls: int = 64
    finger_collision_hull_vertex_limit: int = 64
    finger_collision_voxel_resolution: int = 1000000
    finger_collision_error_percentage: float = 0.1
    finger_planar_zone_start_m: float = 0.1325
    initial_joint_positions: dict[str, float] = field(
        default_factory=lambda: {
            "joint1": 0.0,
            "joint2": 0.5,
            "joint3": 0.5,
            "joint4": 0.0,
            "joint5": 0.0,
            "joint6": 0.0,
            "joint7": 0.0,
            "joint8": 0.0,
        }
    )
    # All six revolute joints in the supplied URDF specify 10 Nm.
    arm_effort_limit_sim: float = 10.0
    arm_velocity_limit_sim: float = 3.0
    arm_stiffness: float = 400.0
    arm_damping: float = 40.0
    gripper_effort_limit_sim: float = 5.0
    gripper_velocity_limit_sim: float = 10.0
    gripper_stiffness: float = 105.2631579
    gripper_damping: float = 5.468
    ee_position_in_gripper_m: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.14472], dtype=np.float64)
    )
    gripper_joint_open_m: float = -0.0475
    gripper_joint_closed_m: float = 0.0
    gripper_contact_overlap_m: float = 0.001039
    cumotion_dir: Path = YAM_ASSET_ROOT / "config" / "cumotion"
    urdf_filename: str = "yam.urdf"
    xrdf_filename: str = "yam.xrdf"
    rmpflow_filename: str = "rmp_flow.yaml"
    tool_frame: str = "gripper"


class YamRobot:
    """Spawn and control one YAM articulation.

    The caller owns the simulation lifecycle. Construct the object before
    ``sim.reset()``. CartesianYamEnv supplies EE control after reset.
    """

    ARM_JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 7))
    GRIPPER_JOINT_NAMES = ("joint7", "joint8")
    TIP_BODY_NAMES = ("tip_left", "tip_right")

    @staticmethod
    def _arm_effort_budget(feedforward, effort_limit):
        """Reserve implicit-drive effort for additive motor feedforward."""
        bounded = feedforward.clamp(-effort_limit, effort_limit)
        return bounded, (effort_limit - bounded.abs()).clamp_min(0.0)

    def set_feedforward_effort_target(self, efforts):
        """Bound arm PD drive + additive torque by the configured motor cap.

        PhysX implicit drive maxForce does not itself bound an additional
        effort target. Conservatively reserve |feedforward| from the drive.
        Fingers keep their existing drive limits and receive zero gravity FF
        in the expert. This method does not change gains or position targets.
        """
        target = efforts.clone()
        ids = self.arm_joint_indices
        bounded, drive_limits = self._arm_effort_budget(target[:, ids], self.config.arm_effort_limit_sim)
        target[:, ids] = bounded
        self.articulation.write_joint_effort_limit_to_sim(limits=drive_limits, joint_ids=ids)
        self.articulation.set_joint_effort_target(target)
        self.arm_feedforward_effort = bounded.clone()
        self.arm_drive_effort_limits = drive_limits.clone()

    @staticmethod
    def _clip_proximal_mesh(vertices, faces, z_cut):
        """Keep z <= z_cut and cap this finger's convex cut cross-section.

        This is specific to the solid distal finger section, not a general
        boolean operation for meshes containing holes at the cut plane.
        """
        output, cuts = [], []
        for face in faces:
            polygon = [np.asarray(vertices[i], dtype=float) for i in face]
            clipped = []
            for a, b in zip(polygon, polygon[1:] + polygon[:1]):
                inside_a, inside_b = a[2] <= z_cut, b[2] <= z_cut
                if inside_a:
                    clipped.append(a)
                if inside_a != inside_b:
                    intersection = a + (z_cut - a[2]) / (b[2] - a[2]) * (b - a)
                    clipped.append(intersection)
                    cuts.append(intersection)
            for i in range(1, len(clipped) - 1):
                output.extend([clipped[0], clipped[i], clipped[i + 1]])
        if len(cuts) >= 3:
            section = np.unique(np.round(cuts, 10), axis=0)
            hull = ConvexHull(section[:, :2])
            center = np.mean(section[hull.vertices], axis=0)
            # Keep collinear boundary vertices to avoid cap T-junctions.
            angles = np.arctan2(section[:, 1] - center[1], section[:, 0] - center[0])
            ring = section[np.argsort(angles)]
            for a, b in zip(ring, np.roll(ring, -1, axis=0)):
                output.extend([center, a, b])
        if not output:
            raise ValueError('Finger clipping removed the entire collider')
        points, inverse = np.unique(np.round(output, 10), axis=0, return_inverse=True)
        triangles = inverse.reshape(-1, 3)
        area2 = np.linalg.norm(np.cross(points[triangles[:, 1]] - points[triangles[:, 0]],
                                        points[triangles[:, 2]] - points[triangles[:, 0]]), axis=1)
        return points, triangles[area2 > 1e-14]

    @classmethod
    def spawn_asset(cls, config: YamRobotConfig) -> None:
        """Author a prototype without creating an articulation view (for cloning)."""
        prototype = cls.__new__(cls)
        prototype.config = config
        prototype._spawn_usd()

    def __init__(self, config: YamRobotConfig | None = None, *, spawn: bool = True):
        self.config = config or YamRobotConfig()
        if spawn:
            self._spawn_usd()
            self.prepare_legacy_rigid_hierarchy(self.config.prim_path)
        self.articulation = Articulation(self._make_articulation_cfg())

    @staticmethod
    def prepare_legacy_rigid_hierarchy(prim_path):
        """Bake world poses before resetting nested rigid Xforms for Sim 4.5.

        Called after independent environment cloning, before PhysX starts.
        Joint/body paths and rigid local coordinate frames remain unchanged.
        Only the current stage is edited; source USD files are never saved.
        """
        stage = get_current_stage()
        cache = UsdGeom.XformCache()
        base = stage.GetPrimAtPath(prim_path + '/Geometry/world/base')
        authored_self_collision = base.GetAttribute('newton:selfCollisionEnabled')
        enabled = bool(authored_self_collision.Get()) if authored_self_collision.IsValid() else False
        PhysxSchema.PhysxArticulationAPI.Apply(base).CreateEnabledSelfCollisionsAttr(enabled)
        bodies = [p for p in Usd.PrimRange(stage.GetPrimAtPath(prim_path))
                  if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        world = [(p, cache.GetLocalToWorldTransform(p)) for p in bodies]
        for prim, matrix in world:
            xform = UsdGeom.Xformable(prim)
            xform.ClearXformOpOrder()
            xform.AddTransformOp(opSuffix='lab21WorldPose').Set(matrix)
            xform.SetResetXformStack(True)
        anchor = UsdPhysics.FixedJoint.Get(stage, prim_path + '/Physics/world_to_base')
        targets = anchor.GetBody1Rel().GetTargets()
        if len(targets) != 1:
            raise ValueError('Expected a single world-to-base anchor target')
        base_world = dict((str(p.GetPath()), m) for p, m in world)[str(targets[0])]
        anchor.GetBody0Rel().ClearTargets(True)
        # Sim 4.5 resolves an unbound joint side in its parent Xform frame;
        # using world coordinates here would apply the env translation twice.
        anchor_frame = base_world * cache.GetLocalToWorldTransform(anchor.GetPrim().GetParent()).GetInverse()
        anchor.GetLocalPos0Attr().Set(Gf.Vec3f(anchor_frame.Transform(Gf.Vec3d(anchor.GetLocalPos1Attr().Get()))))
        anchor.GetLocalRot0Attr().Set(Gf.Quatf(anchor_frame.ExtractRotationQuat() * Gf.Quatd(anchor.GetLocalRot1Attr().Get())))

    @property
    def articulation_root_path(self) -> str:
        """Articulation root inside the spawned YAM USD."""
        return f"{self.config.prim_path}/Geometry/world/base"

    def _spawn_usd(self) -> None:
        """Spawn the configured USD and apply the finger collision settings."""
        if not self.config.usd_path.is_file():
            raise FileNotFoundError(self.config.usd_path)
        usd_cfg = sim_utils.UsdFileCfg(
            usd_path=str(self.config.usd_path),
            activate_contact_sensors=True,
        )
        usd_cfg.func(
            self.config.prim_path,
            usd_cfg,
            translation=self.config.translation,
            orientation=tuple(self.config.orientation_xyzw[i] for i in (3, 0, 1, 2)),
        )
        self._configure_finger_colliders()
        self._configure_visual_materials()

    def _configure_visual_materials(self) -> None:
        """Apply an approximate standard YAM black/white finish to CAD meshes."""
        stage = get_current_stage()
        materials = {}
        for name, color in {
            "Dark": (0.025, 0.028, 0.032),
            "White": (0.72, 0.74, 0.76),
        }.items():
            path = f"{self.config.prim_path}/VisualMaterials/{name}"
            cfg = sim_utils.PreviewSurfaceCfg(
                diffuse_color=color, roughness=0.55, metallic=0.05,
            )
            cfg.func(path, cfg)
            materials[name] = UsdShade.Material.Get(stage, path)
        root = stage.GetPrimAtPath(self.articulation_root_path)
        # Visual copies are instanceable. De-instance them to author display
        # overrides independently from physics collision meshes.
        for prim in list(Usd.PrimRange(root)):
            if prim.IsInstance():
                prim.SetInstanceable(False)
        for prim in Usd.PrimRange(root):
            if not prim.IsA(UsdGeom.Mesh) or prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            mesh = UsdGeom.Mesh(prim)
            mesh.CreateSubdivisionSchemeAttr("none")
            mesh.CreateDoubleSidedAttr(True)
            link_name = prim.GetName()
            finish = "White" if link_name in {"link2", "link3"} else "Dark"
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(materials[finish])

    def _configure_finger_colliders(self) -> None:
        """Apply decomposed convex collision and planar contact faces."""
        stage = get_current_stage()
        gripper_path = (
            f"{self.articulation_root_path}/link1/link2/link3/link4/link5/gripper"
        )
        material_path = f"{self.config.prim_path}/PhysicsMaterials/FingerGrip"
        material_cfg = sim_utils.RigidBodyMaterialCfg(
            static_friction=self.config.finger_static_friction,
            dynamic_friction=self.config.finger_dynamic_friction,
            restitution=self.config.finger_restitution,
        )
        material_cfg.func(material_path, material_cfg)

        for tip_name in self.TIP_BODY_NAMES:
            collision_instance_path = f"{gripper_path}/{tip_name}/{tip_name}_1"
            collision_instance = stage.GetPrimAtPath(collision_instance_path)
            if not collision_instance.IsValid():
                raise RuntimeError(
                    f"YAM finger collision instance not found: {collision_instance_path}"
                )
            collision_instance.SetInstanceable(False)
            collision_mesh_path = f"{collision_instance_path}/{tip_name}"
            collision_mesh = stage.GetPrimAtPath(collision_mesh_path)
            if not collision_mesh.IsValid():
                raise RuntimeError(
                    f"YAM finger collision mesh not found: {collision_mesh_path}"
                )

            # The distal CAD surface and an overlapping analytic contact face
            # must not both generate contacts. Crop only the collision mesh,
            # retaining the original visual mesh and rigid-body mass/inertia.
            mesh = UsdGeom.Mesh(collision_mesh)
            cache = UsdGeom.XformCache()
            matrix = np.asarray(cache.GetLocalToWorldTransform(collision_mesh)
                                * cache.GetLocalToWorldTransform(stage.GetPrimAtPath(gripper_path)).GetInverse())
            local_points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
            gripper_points = (np.c_[local_points, np.ones(len(local_points))] @ matrix)[:, :3]
            if not 0.1445 < gripper_points[:, 2].max() < 0.1450:
                raise ValueError(f'Unexpected {tip_name} collider frame; refusing to crop it')
            counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
            if not np.all(counts == 3):
                raise ValueError('Finger collision mesh must be triangulated')
            faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
            proximal, faces = self._clip_proximal_mesh(gripper_points, faces, self.config.finger_planar_zone_start_m)
            points = (np.c_[proximal, np.ones(len(proximal))] @ np.linalg.inv(matrix))[:, :3]
            mesh.GetPointsAttr().Set(points.astype(np.float32).tolist())
            mesh.GetFaceVertexCountsAttr().Set([3] * len(faces))
            mesh.GetFaceVertexIndicesAttr().Set(faces.flatten().tolist())
            # Lab 2.1 has no mesh-collision config helper. Author the same
            # decomposition parameters through Sim 4.5's native USD schema.
            UsdPhysics.MeshCollisionAPI.Apply(collision_mesh).CreateApproximationAttr('convexDecomposition')
            decomposition = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(collision_mesh)
            decomposition.CreateHullVertexLimitAttr(self.config.finger_collision_hull_vertex_limit)
            decomposition.CreateMaxConvexHullsAttr(self.config.finger_collision_max_hulls)
            decomposition.CreateVoxelResolutionAttr(self.config.finger_collision_voxel_resolution)
            decomposition.CreateErrorPercentageAttr(self.config.finger_collision_error_percentage)
            decomposition.CreateShrinkWrapAttr(True)
            approximation = UsdPhysics.MeshCollisionAPI(
                collision_mesh
            ).GetApproximationAttr().Get()
            if approximation != "convexDecomposition":
                raise RuntimeError(
                    f"YAM {tip_name} collision approximation was not applied: "
                    f"expected 'convexDecomposition', got {approximation!r}"
                )
            sim_utils.bind_physics_material(
                collision_mesh_path,
                material_path,
                stage=stage,
            )
            sim_utils.define_collision_properties(
                collision_mesh_path,
                sim_utils.CollisionPropertiesCfg(contact_offset=0.0001, rest_offset=0.0),
            )
            # Analytic approximation of the planar inner contact faces measured
            # from the STL at distal Z=0.1325..0.1445 m in gripper coordinates.
            # Thin contact faces are more reliable as boxes than sampled SDFs.
            if tip_name == "tip_left":
                origin = np.array([-0.0450619, 0.0226677, 0.0545600])
                rotation = Rotation.from_euler("xyz", [-1.4328917, 1.5707889, 1.7087083])
                center = np.array([0.0014419, -0.001325, 0.13861])
            else:
                origin = np.array([0.0450620, -0.0226681, 0.0545598])
                rotation = Rotation.from_euler("xyz", [-1.5707937, 0.0000037, -1.5707927])
                center = np.array([-0.0014419, 0.001325, 0.13861])
            pad = sim_utils.CuboidCfg(
                size=(0.0039228, 0.0155, 0.01222), visible=False,
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.0001, rest_offset=0.0),
                physics_material=material_cfg,
            )
            pad.func(f"{gripper_path}/{tip_name}/PlanarContactFace", pad,
                     translation=tuple(rotation.inv().apply(center - origin)),
                     orientation=tuple(rotation.inv().as_quat()[[3, 0, 1, 2]]))
            print(
                f"[YAM COLLISION] {tip_name}: verified approximation={approximation}; "
                f"planar contact faces; "
                f"friction static={self.config.finger_static_friction}, "
                f"dynamic={self.config.finger_dynamic_friction}, "
                f"restitution={self.config.finger_restitution}",
                flush=True,
            )

    def _make_articulation_cfg(self) -> ArticulationCfg:
        """Build Isaac Lab articulation and actuator settings from ``config``."""
        return ArticulationCfg(
            prim_path=self.articulation_root_path,
            spawn=None,
            init_state=ArticulationCfg.InitialStateCfg(
                pos=self.config.articulation_base_pos,
                rot=self.config.articulation_base_rot_wxyz,
                joint_pos=self.config.initial_joint_positions,
            ),
            actuators={
                "arm": ImplicitActuatorCfg(
                    joint_names_expr=["joint[1-6]"],
                    effort_limit_sim=self.config.arm_effort_limit_sim,
                    velocity_limit_sim=self.config.arm_velocity_limit_sim,
                    stiffness=self.config.arm_stiffness,
                    damping=self.config.arm_damping,
                ),
                "gripper": ImplicitActuatorCfg(
                    joint_names_expr=["joint[7-8]"],
                    effort_limit_sim=self.config.gripper_effort_limit_sim,
                    velocity_limit_sim=self.config.gripper_velocity_limit_sim,
                    stiffness=self.config.gripper_stiffness,
                    damping=self.config.gripper_damping,
                ),
            },
        )

    @property
    def max_gripper_opening_m(self) -> float:
        """Maximum total finger opening [m]."""
        return 2.0 * abs(self.config.gripper_joint_open_m) - self.config.gripper_contact_overlap_m

    @property
    def arm_joint_indices(self) -> list[int]:
        return [self.articulation.joint_names.index(name) for name in self.ARM_JOINT_NAMES]

    @property
    def gripper_joint_indices(self) -> list[int]:
        return [self.articulation.joint_names.index(name) for name in self.GRIPPER_JOINT_NAMES]

    @property
    def tool_body_index(self) -> int:
        return self.articulation.body_names.index(self.config.tool_frame)

    def body_index(self, name: str) -> int:
        """Return the articulation body index for ``name``."""
        return self.articulation.body_names.index(name)

    def ee_pose_to_gripper_pose(
        self,
        position_base_ee: np.ndarray,
        rotation_base_ee: Rotation,
    ) -> tuple[np.ndarray, Rotation]:
        """Convert a manipulation EE pose to the XRDF gripper-link pose."""
        position = np.asarray(position_base_ee, dtype=np.float64) - rotation_base_ee.apply(
            self.config.ee_position_in_gripper_m
        )
        return position.astype(np.float32), rotation_base_ee

    def gripper_pose_to_ee_pose(
        self,
        position_base_gripper: np.ndarray,
        rotation_base_gripper: Rotation,
    ) -> tuple[np.ndarray, Rotation]:
        """Convert the physical gripper-link pose to the manipulation EE pose."""
        position = np.asarray(position_base_gripper, dtype=np.float64) + rotation_base_gripper.apply(
            self.config.ee_position_in_gripper_m
        )
        return position.astype(np.float32), rotation_base_gripper

    def get_ee_poses_base(self) -> torch.Tensor:
        """All native TCP poses on device, XYZW, without per-row CPU transfers."""
        from isaaclab.utils.math import quat_apply, subtract_frame_transforms
        data = self.articulation.data
        position = data.body_pos_w[:, self.tool_body_index]
        rotation = data.body_quat_w[:, self.tool_body_index]
        offset = torch.as_tensor(self.config.ee_position_in_gripper_m,
            dtype=position.dtype, device=position.device).expand_as(position)
        tcp = position + quat_apply(rotation, offset)
        p, q = subtract_frame_transforms(data.root_pos_w,
            data.root_quat_w, tcp, rotation)
        return torch.cat([p, q[:, [1, 2, 3, 0]]], dim=-1)

    def get_ee_pose_base(self, env_id: int = 0) -> tuple[np.ndarray, Rotation]:
        """Return the current manipulation EE pose expressed in robot BASE."""
        pose = self.get_ee_poses_base()[env_id].detach().cpu().numpy()
        return pose[:3], Rotation.from_quat(pose[3:])

    def get_ee_jacobian_base(self) -> torch.Tensor:
        """PhysX COM Jacobian shifted to TCP, expressed in BASE (Lab 2.1).

        Raw PhysX columns include six root DOFs only for floating articulations.
        The TCP offset is relative to the actor origin, not its centre of mass.
        Validate this boundary with test_cartesian_adapter before collecting.
        """
        from isaaclab.utils.math import quat_apply, quat_conjugate
        art = self.articulation
        if not art.is_fixed_base and not getattr(self, '_cartesian_anchor_checked', False):
            base_paths = sim_utils.find_matching_prim_paths(art.cfg.prim_path)
            if len(base_paths) != art.data.root_quat_w.shape[0]:
                raise ValueError('Cannot resolve every articulation world anchor')
            for base_path in base_paths:
                anchor_path = str(base_path).removesuffix('/Geometry/world/base') + '/Physics/world_to_base'
                anchor = UsdPhysics.FixedJoint.Get(get_current_stage(), anchor_path)
                if (not anchor or anchor.GetJointEnabledAttr().Get() is False
                        or str(base_path) not in [str(p) for p in anchor.GetBody1Rel().GetTargets()]):
                    raise ValueError('Cartesian controller requires world-anchored YAM assets')
            self._cartesian_anchor_checked = True
        raw = art.root_physx_view.get_jacobians()
        root_columns = raw.shape[-1] - art.num_joints
        if root_columns != (0 if art.is_fixed_base else 6):
            raise RuntimeError('Unexpected PhysX Jacobian layout')
        columns = [root_columns + index for index in self.arm_joint_indices]
        body = self.tool_body_index - int(art.is_fixed_base)
        jacobian = raw[:, body, :, :][:, :, columns].clone()
        offset = torch.as_tensor(self.config.ee_position_in_gripper_m, device=jacobian.device,
                                 dtype=jacobian.dtype).expand(jacobian.shape[0], -1)
        offset = offset - art.data.com_pos_b[:, self.tool_body_index]
        offset_world = quat_apply(art.data.body_quat_w[:, self.tool_body_index], offset)
        jacobian[:, :3] += torch.cross(jacobian[:, 3:].transpose(1, 2),
            offset_world[:, None, :].expand(-1, len(columns), -1), dim=-1).transpose(1, 2)
        for block in (slice(0, 3), slice(3, 6)):
            vectors = jacobian[:, block, :].transpose(1, 2)
            quaternions = quat_conjugate(art.data.root_quat_w)[:, None, :].expand(-1, len(columns), -1)
            jacobian[:, block, :] = quat_apply(quaternions.reshape(-1, 4),
                vectors.reshape(-1, 3)).reshape(-1, len(columns), 3).transpose(1, 2)
        return jacobian

    def opening_to_joint_position(self, opening_m: float) -> float:
        """Map total symmetric jaw opening [m] to either prismatic joint [m]."""
        opening_m = float(np.clip(opening_m, 0.0, self.max_gripper_opening_m))
        return float(
            np.clip(
                -0.5 * (opening_m + self.config.gripper_contact_overlap_m),
                self.config.gripper_joint_open_m,
                self.config.gripper_joint_closed_m,
            )
        )

    def set_gripper_opening_target(self, opening_m: float) -> torch.Tensor:
        """Command a total symmetric jaw opening [m] while holding arm targets."""
        target = self.articulation.data.joint_pos.clone()
        target[0, self.gripper_joint_indices] = self.opening_to_joint_position(opening_m)
        self.set_joint_position_target(target)
        return target

    def get_gripper_opening(self, env_id: int = 0) -> float:
        """Return the joint-coordinate-derived total jaw opening [m]."""
        mean_joint_position = float(
            self.articulation.data.joint_pos[env_id, self.gripper_joint_indices].mean().item()
        )
        return float(np.clip(-2.0 * mean_joint_position - self.config.gripper_contact_overlap_m,
                             0.0, self.max_gripper_opening_m))

    def set_joint_position_target(
        self,
        positions: torch.Tensor | Sequence[float] | dict[str, float],
        joint_names: Sequence[str] | None = None,
    ) -> torch.Tensor:
        """Set full or partial joint-space position targets."""
        target = self.articulation.data.joint_pos.clone()
        if isinstance(positions, dict):
            for name, value in positions.items():
                target[0, self.articulation.joint_names.index(name)] = float(value)
        elif isinstance(positions, torch.Tensor):
            if positions.shape == target.shape:
                target = positions.to(device=target.device, dtype=target.dtype)
            else:
                names = list(joint_names) if joint_names is not None else list(self.articulation.joint_names)
                values = positions.flatten()
                if values.numel() != len(names):
                    raise ValueError("Joint target length does not match joint_names.")
                for name, value in zip(names, values):
                    target[0, self.articulation.joint_names.index(name)] = float(value)
        else:
            names = list(joint_names) if joint_names is not None else list(self.articulation.joint_names)
            values = list(positions)
            if len(values) != len(names):
                raise ValueError("Joint target length does not match joint_names.")
            for name, value in zip(names, values):
                target[0, self.articulation.joint_names.index(name)] = float(value)
        self.articulation.set_joint_position_target(target=target)
        return target

    def write_data_to_sim(self) -> None:
        """Write the latest articulation command buffers to PhysX."""
        self.articulation.write_data_to_sim()

    def update(self, physics_dt_s: float) -> None:
        """Update articulation state buffers after a physics step."""
        self.articulation.update(physics_dt_s)
