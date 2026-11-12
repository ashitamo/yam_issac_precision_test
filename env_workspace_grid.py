"""Workspace grid plus CPU-only kinematics used for screening."""

from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import argparse
import json
import env_config as cfg


class YamKinematics:
    def __init__(self, urdf_path: Path, tcp=(0.0, 0.0, 0.14472)):
        root = ET.parse(urdf_path).getroot()
        joints = [root.find(f"joint[@name='joint{i}']") for i in range(1, 7)]
        self.origins = [np.fromstring(j.find('origin').get('xyz'), sep=' ') for j in joints]
        self.rotations = [Rotation.from_euler('xyz', np.fromstring(j.find('origin').get('rpy'), sep=' ')).as_matrix() for j in joints]
        axes = [np.fromstring(j.find('axis').get('xyz'), sep=' ') for j in joints]
        self.skews = [np.array([[0., -a[2], a[1]], [a[2], 0., -a[0]], [-a[1], a[0], 0.]]) for a in axes]
        self.lower = np.array([float(j.find('limit').get('lower')) for j in joints])
        self.upper = np.array([float(j.find('limit').get('upper')) for j in joints])
        self.tcp = np.asarray(tcp)

    def fk(self, q):
        p, r = np.zeros(3), np.eye(3)
        for origin, rotation, skew, angle in zip(self.origins, self.rotations, self.skews, q):
            p = p + r @ origin
            r = r @ rotation @ (np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew))
        return p + r @ self.tcp, Rotation.from_matrix(r)

    def solve(self, target, orientation, seed):
        def residual(q):
            p, r = self.fk(q)
            return np.r_[p - target, 0.1 * (r.inv() * orientation).as_rotvec()]
        best = None
        for initial in [seed, [0., 1.7, 1.5, -1.4, 0., 0.], [0., 2.0, 1.0, -1.0, 0., 0.]]:
            fit = least_squares(residual, np.clip(initial, self.lower + 1e-7, self.upper - 1e-7),
                                bounds=(self.lower, self.upper), max_nfev=150)
            p, r = self.fk(fit.x)
            error = float(np.linalg.norm(p - target))
            angle = float(np.degrees((r.inv() * orientation).magnitude()))
            item = dict(q=fit.x.tolist(), position_error_m=error, orientation_error_deg=angle,
                        reachable=bool(error <= 0.0001 and angle <= 0.05))
            if best is None or np.linalg.norm(fit.fun) < best[0]:
                best = (np.linalg.norm(fit.fun), item)
            if item['reachable']:
                return item
        return best[1]


# --- Grid manifest and command-line screening ---

MANIFEST_PATH = cfg.PROJECT_ROOT / 'validation' / 'grid_manifest.json'


def make_manifest():
    points = []
    center = np.asarray(cfg.GRID_CENTER_XY_M)
    for row in range(cfg.GRID_SIDE_COUNT):
        for col in range(cfg.GRID_SIDE_COUNT):
            xy = center + (np.array([col, row]) / (cfg.GRID_SIDE_COUNT - 1) - 0.5) * cfg.GRID_WIDTH_M
            # Use the largest peg for a clearance-independent eligibility mask.
            gap = float(np.linalg.norm(xy - cfg.PEG_POSITION_M[:2]) - cfg.WORKPIECE_DIAMETER_M / 2 - .0098 / 2)
            points.append(dict(grid_id=f'r{row:02d}_c{col:02d}', row=row, col=col,
                               hole_position_m=[float(xy[0]), float(xy[1]), cfg.WORKPIECE_POSITION_M[2]],
                               initial_peg_surface_gap_m=gap, initial_scene_clear=gap > 2 * cfg.OBJECT_CONTACT_OFFSET_M,
                               candidate_group='seen' if (row + col) % 2 == 0 else 'unseen'))
    return dict(schema_version=1, frame='YAM_BASE', width_m=cfg.GRID_WIDTH_M,
                side_count=cfg.GRID_SIDE_COUNT, spacing_m=cfg.GRID_WIDTH_M / (cfg.GRID_SIDE_COUNT - 1),
                center_xy_m=list(cfg.GRID_CENTER_XY_M), split_status='candidate_only_not_training_split', points=points)


def get_grid_point(grid_id):
    # Generate from the same constants even before the first screening run.
    manifest = make_manifest()
    return next(p for p in manifest['points'] if p['grid_id'] == grid_id)


def screen_workspace(output_dir=MANIFEST_PATH.parent):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = make_manifest()
    (output_dir / 'grid_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    model = YamKinematics(cfg.PROJECT_ROOT / 'assets/yam_robot_with_gripper/config/cumotion/yam.urdf')
    orientation = Rotation.from_euler('z', cfg.GRASP_YAW_DEG, degrees=True) * Rotation.from_euler('x', 180, degrees=True)
    seed = np.array([0., .5, .5, 0., 0., 0.])
    peg_xy = np.array(cfg.PEG_POSITION_M[:2])
    grasp_tip_offset = cfg.GRASP_HEIGHT_M - (cfg.PEG_POSITION_M[2] - cfg.PEG_LENGTH_M / 2)
    records = []
    for point in manifest['points']:
        xy = np.array(point['hole_position_m'][:2])
        top_z = point['hole_position_m'][2] + cfg.WORKPIECE_HEIGHT_M
        samples = [('approach', [*peg_xy, cfg.GRASP_APPROACH_Z_M]),
                   ('grasp', [*peg_xy, cfg.GRASP_HEIGHT_M]),
                   ('lift', [*peg_xy, cfg.GRASP_APPROACH_Z_M])]
        samples += [(f'transfer_{k}', [*(peg_xy + (xy - peg_xy) * k / 5), cfg.GRASP_APPROACH_Z_M]) for k in range(1, 6)]
        z0 = top_z + cfg.PREINSERT_TIP_GAP_M + grasp_tip_offset
        z1 = top_z - cfg.INSERTION_COMMAND_DEPTH_M + grasp_tip_offset
        samples += [(f'insertion_{k}', [*xy, z]) for k, z in enumerate(np.linspace(z0, z1, 6))]
        q = seed.copy()
        checks = []
        for name, target in samples:
            item = model.solve(np.array(target), orientation, q)
            item.update(stage=name, target_ee_m=list(target))
            checks.append(item)
            if item['reachable']:
                q = np.array(item['q'])
        records.append(dict(**point, kinematically_reachable=all(c['reachable'] for c in checks), checks=checks))
        print('[GRID]', point['grid_id'], records[-1]['kinematically_reachable'], flush=True)
    report = dict(scope='Nominal fixed-orientation sampled FK/IK only; NOT collision-free or full-expert certification',
                  sample_count_per_position=len(samples), total=len(records),
                  reachable=sum(r['kinematically_reachable'] for r in records),
                  initial_geometry_clear=sum(r['initial_scene_clear'] for r in records),
                  nominally_eligible=sum(r['kinematically_reachable'] and r['initial_scene_clear'] for r in records), records=records)
    (output_dir / 'grid_reachability.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    # Simple portable map, blue = nominal IK pass, red = fail.
    circles = ''.join(f'<circle cx="{30+r["col"]*25}" cy="{430-r["row"]*25}" r="6" fill="{"#1679b2" if r["kinematically_reachable"] else "#ce3030"}"><title>{r["grid_id"]}</title></circle>' for r in records)
    (output_dir / 'grid_reachability.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg" width="480" height="480"><rect width="100%" height="100%" fill="white"/><text x="20" y="20">Nominal IK screening. Blue pass, red fail.</text>' + circles + '</svg>', encoding='utf-8')
    print(f'[GRID SUMMARY] {report["reachable"]}/{report["total"]} nominal IK passes', flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=Path, default=MANIFEST_PATH.parent)
    screen_workspace(parser.parse_args().output_dir)
