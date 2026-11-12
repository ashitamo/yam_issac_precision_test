"""Shared measured task geometry. No simulator, IK or dataset dependency."""
import numpy as np
import env_config as cfg


def geometry_diagnostic_thresholds():
    return dict(radial_m=cfg.RADIAL_GEOMETRY_TOLERANCE_M, bottom_m=cfg.BOTTOM_GEOMETRY_TOLERANCE_M)


def contact_gate_pass(radial, bottom, thresholds=None):
    thresholds = geometry_diagnostic_thresholds() if thresholds is None else thresholds
    return bool(radial is not None and bottom is not None
        and np.isfinite([radial, bottom]).all()
        and radial <= thresholds['radial_m'] and bottom <= thresholds['bottom_m'])


def peg_geometry(position, axis, hole_top, hole_radius, peg_diameter):
    position, axis, hole_top = map(lambda value: np.asarray(value, dtype=float), (position, axis, hole_top))
    axis = axis / np.linalg.norm(axis)
    if axis[2] < 0:
        axis = -axis
    tip = position - .5 * cfg.PEG_LENGTH_M * axis
    depth = max(0., float(hole_top[2] - tip[2]))
    xy_error = None
    envelope = float(np.linalg.norm(position[:2] - hole_top[:2])) + cfg.PEG_LENGTH_M / 2 + peg_diameter / 2
    inside = False
    if axis[2] > .1:
        entrance = position + axis * ((hole_top[2] - position[2]) / axis[2])
        xy_error = float(np.linalg.norm(entrance[:2] - hole_top[:2]))
        envelope = max(xy_error, float(np.linalg.norm(tip[:2] - hole_top[:2]))) + peg_diameter / (2 * axis[2])
        inside = bool(envelope <= hole_radius and depth > 0
            and tip[2] >= hole_top[2] - cfg.WORKPIECE_HEIGHT_M - cfg.OBJECT_CONTACT_OFFSET_M
            and position[2] + .5 * cfg.PEG_LENGTH_M * axis[2] >= hole_top[2])
    lowest_z = tip[2] - peg_diameter / 2 * np.sqrt(max(0., 1 - axis[2] ** 2))
    return dict(axis=axis, tip=tip, depth=depth, axis_xy_error=xy_error,
        radial_envelope=envelope, geometry_inside=inside,
        tilt_deg=float(np.degrees(np.arccos(np.clip(axis[2], 0., 1.)))), lowest_z=float(lowest_z))


def release_ready(opened, tip_z, peg_speed, geometry_inside, insertion_depth):
    table_top = cfg.TABLE_POSITION_M[2] + cfg.TABLE_SIZE_M[2] / 2
    return bool(opened and geometry_inside and insertion_depth >= cfg.INSERTION_SUCCESS_DEPTH_M
        and abs(tip_z - table_top) <= cfg.RELEASE_BOTTOM_TOLERANCE_M
        and peg_speed <= cfg.RELEASE_MAX_PEG_SPEED_M_S)
