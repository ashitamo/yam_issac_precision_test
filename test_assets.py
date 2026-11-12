"""Read-only asset audit, plus a local JSON report. Does not require Isaac startup."""
import json
import ast
import importlib.metadata
import platform
import sys
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
from pxr import Usd, UsdUtils
import env_config as cfg


def main():
    versions = {}
    for package in ['numpy', 'scipy', 'torch', 'trimesh', 'isaacsim', 'isaaclab']:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = 'not registered as a distribution'
    asset = cfg.PROJECT_ROOT / 'assets/yam_robot_with_gripper'
    usd = asset / 'usd/yam_with_linear_gripper.usd'
    stage = Usd.Stage.Open(str(usd))
    root = ET.parse(asset / 'config/cumotion/yam.urdf').getroot()
    source = ast.parse((cfg.PROJECT_ROOT / 'robot_yam.py').read_text(encoding='utf-8'))
    config_class = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == 'YamRobotConfig')
    arm_defaults = {node.target.id: ast.literal_eval(node.value) for node in config_class.body
                    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
                    and node.target.id in {'arm_effort_limit_sim', 'arm_velocity_limit_sim'}}
    joints = []
    for joint in root.findall('joint'):
        if joint.get('type') not in {'revolute', 'prismatic'}:
            continue
        name = joint.get('name')
        prim = stage.GetPrimAtPath('/output/Physics/' + name)
        actual = np.array([prim.GetAttribute('physics:lowerLimit').Get(), prim.GetAttribute('physics:upperLimit').Get()], dtype=float)
        if joint.get('type') == 'revolute':
            actual = np.deg2rad(actual)
        expected = np.array([float(joint.find('limit').get(k)) for k in ['lower', 'upper']])
        joints.append(dict(name=name, type=joint.get('type'), urdf_limits=expected.tolist(), usd_limits=actual.tolist(),
                           urdf_effort_limit=float(joint.find('limit').get('effort')),
                           urdf_velocity_limit=float(joint.find('limit').get('velocity')),
                           position_limits_match=bool(np.allclose(actual, expected, atol=1e-5, rtol=0))))
    _, _, unresolved = UsdUtils.ComputeAllDependencies(str(usd))
    workpiece = trimesh.load_mesh(cfg.WORKPIECE_PATH)
    points = workpiece.vertices
    mask = (np.abs(points[:, 2]) < 1e-7) & (np.abs(np.linalg.norm(points[:, :2], axis=1) - .005) < 1e-6)
    ring = points[mask, :2]
    ring = ring[np.argsort(np.arctan2(ring[:, 1], ring[:, 0]))]
    nxt = np.roll(ring, -1, axis=0)
    hole_radius = float(np.min(np.abs(ring[:, 0] * nxt[:, 1] - ring[:, 1] * nxt[:, 0]) / np.linalg.norm(nxt - ring, axis=1)))
    report = dict(python_version=sys.version, platform=platform.platform(), package_versions=versions,
                  joints=joints, all_position_limits_match=all(j['position_limits_match'] for j in joints),
                  unresolved_usd_dependencies=[str(x) for x in unresolved],
                  actuator_limit_overrides=dict(arm=arm_defaults,
                                                gripper_effort_n=cfg.GRIPPER_FORCE_LIMIT_N,
                                                gripper_velocity_m_s=cfg.GRIPPER_VELOCITY_LIMIT_M_S),
                  arm_motor_effort_matches_urdf=all(j['urdf_effort_limit'] == arm_defaults['arm_effort_limit_sim']
                                                    for j in joints if j['type'] == 'revolute'),
                  workpiece_watertight=bool(workpiece.is_watertight), workpiece_extents_m=workpiece.extents.tolist(),
                  nominal_hole_radius_m=.005, actual_inscribed_hole_radius_m=hole_radius,
                  effective_clearances_mm={str(c): (hole_radius - (.01 - 2*c/1000)/2)*1000 for c in [1., .5, .25, .1]},
                  limitations=[
                      'Nominal radial clearances differ slightly from the faceted STL aperture.',
                      'Arm motor effort cap follows URDF (10 Nm); gains, conservative velocity and finger settings remain simulation approximations, not hardware calibration.',
                      'Table/workpiece physics material uses simulator defaults; no hardware material identification.',
                      'D405 is ideal RGB/depth with visual-only housing/bracket, not a calibrated stereo-noise/mass model.',
                      'No ACT/DP training or Cartesian torque impedance controller yet; RGB-D pilot recorder exists.',
                      'Grid candidate seen/unseen labels are not finalized train/validation/evaluation splits.',
                  ])
    output = cfg.PROJECT_ROOT / 'validation/asset_audit.json'
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    if (not report['all_position_limits_match'] or not report['arm_motor_effort_matches_urdf']
            or unresolved or not report['workpiece_watertight']):
        raise SystemExit('Asset audit failed')


if __name__ == '__main__':
    main()
