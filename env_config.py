"""Fixed experiment settings, grouped by subsystem. Lengths use metres."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "latest"

# Candidate hole workspace. Exact coordinates, no placement perturbation.
# Keep the fixed peg outside the hole workspace, while retaining nominal IK reach.
GRID_CENTER_XY_M = (0.265, 0.065)
GRID_WIDTH_M = 0.2
GRID_SIDE_COUNT = 17

# Simulation and overview camera.
PHYSICS_DT_S = 1.0 / 240.0
CONTROL_PERIOD_S = 1.0 / 240.0
# Policy Cartesian contract; raw SI units, independent of physics/servo rate.
POLICY_PERIOD_S = 1.0 / 30.0
CARTESIAN_MAX_TRANSLATION_M = 0.02
CARTESIAN_MAX_ROTATION_RAD = 0.2
CARTESIAN_IK_DAMPING = 0.005
SCENE_MIN_POSITION_ITERATIONS = 64
SCENE_MIN_VELOCITY_ITERATIONS = 16
# Sim 4.5 lacks newer solver controls: a different physics baseline.
# Repeat native contact validation before production collection.
TGS_EXTERNAL_FORCES_EVERY_ITERATION = False
SOLVE_ARTICULATION_CONTACT_LAST = False
RENDER_EVERY_N_STEPS = 8
OVERVIEW_EYE_M = (0.8, 0.8, 0.6)
OVERVIEW_TARGET_M = (0.3, 0.0, 0.05)
LIGHT_INTENSITY = 2500.0

# D405 optical frame relative to the gripper. Quaternion order is XYZW.
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 360
CAMERA_PERIOD_S = 1.0 / 30.0
CAMERA_POSITION_M = (0.0, 0.070435, 0.077006)
CAMERA_QUAT_XYZW = (0.0, 0.216440, -0.976296, 0.0)
CAMERA_FOCAL_LENGTH_MM = 1.93
CAMERA_HORIZONTAL_APERTURE_MM = 3.896
CAMERA_VERTICAL_APERTURE_MM = 2.140
CAMERA_CLIPPING_M = (0.001, 10.0)
CAMERA_MESH_PATH = PROJECT_ROOT / "assets/d405/d405_wrist_linear_4310.stl"

# Table, fixed workpiece, and dynamic peg.
TABLE_SIZE_M = (1.0, 0.8, 0.05)
TABLE_POSITION_M = (0.3, 0.0, -0.025)
WORKPIECE_PATH = PROJECT_ROOT / "assets" / "workpiece.stl"
WORKPIECE_POSITION_M = (0.35, 0.08, 0.0)
WORKPIECE_DIAMETER_M = 0.032
WORKPIECE_HEIGHT_M = 0.032
HOLE_DIAMETER_MM = 10.0
WORKPIECE_COLOR = (0.008, 0.008, 0.008)
PEG_LENGTH_M = 0.05
PEG_POSITION_M = (0.3, -0.08, 0.025)
PEG_MASS_KG = 0.025
PEG_COLOR = (0.8, 0.8, 0.8)
PEG_MAX_DEPENETRATION_M_S = 0.05
PEG_POSITION_ITERATIONS = 64
PEG_VELOCITY_ITERATIONS = 16
OBJECT_CONTACT_OFFSET_M = 0.00002
# User-accepted geometry diagnostic tolerances; NOT PhysX contact/rest offsets.
RADIAL_GEOMETRY_TOLERANCE_M = 0.00005
BOTTOM_GEOMETRY_TOLERANCE_M = 0.00005

# Gripper drive tuning. Per-finger force [N], velocity [m/s], Kp [N/m], Kd [N s/m].
GRIPPER_FORCE_LIMIT_N = 1.5
GRIPPER_VELOCITY_LIMIT_M_S = 0.02
GRIPPER_STIFFNESS_N_M = 1000.0
GRIPPER_DAMPING_N_S_M = 17.0
CONTACT_STATIC_FRICTION = 1.0
CONTACT_DYNAMIC_FRICTION = 0.8
CONTACT_RESTITUTION = 0.0

# Grasp trajectory and acceptance thresholds.
GRASP_YAW_DEG = 90.0
# Keeps the peg stable with the decomposed-convex finger colliders.
GRASP_PRELOAD_MM = 2.0
GRASP_OPENING_M = 0.06
APPROACH_OPENING_M = GRASP_OPENING_M
GRASP_APPROACH_Z_M = 0.14
GRASP_HEIGHT_M = 0.038
ARM_COMMAND_SPEED_RAD_S = 0.5
# Smooth Cartesian carry reference, with zero endpoint velocity/acceleration.
TRANSFER_SPEED_M_S = 0.10
TRANSFER_ACCELERATION_M_S2 = 0.20
TRANSFER_IK_PERIOD_S = 1.0 / 30.0
ALIGN_APPROACH_SPEED_M_S = 0.05
ALIGN_APPROACH_ACCELERATION_M_S2 = 0.10
GRAVITY_COMPENSATION_ENABLED = True
POSITION_TOLERANCE_M = 0.002
ORIENTATION_TOLERANCE_DEG = 2.0
REACHED_HOLD_S = 0.5
# Advance after the fingers establish a stable loaded grasp.
CLOSE_STABLE_S = 0.15
CLOSE_GAP_TOLERANCE_M = 0.0005
CLOSE_MAX_FINGER_SPEED_M_S = 0.001
# Sim 4.5 reports noisy solver velocities at stationary contacts. Assess
# settling from the position envelope over this interval, not one solver tick.
CLOSE_POSITION_WINDOW_S = 0.05
CLOSE_MIN_DRIVE_FORCE_N = 0.05
SAVE_VALIDATION_IMAGES = False
LIFT_REACHED_HOLD_S = 0.15
LIFT_HOLD_S = 0.15
STAGE_TIMEOUT_S = 20.0
MIN_LIFTED_PEG_CENTER_Z_M = 0.075
MAX_HOLD_SLIP_M = 0.002
TRACK_LOG_PERIOD_S = 2.0

# Pick-and-insert continuation after the grasp validation.
RUN_INSERTION = True
PREINSERT_TIP_GAP_M = 0.003
# Command depth and final success depth are independent.
INSERTION_COMMAND_DEPTH_M = 0.0305
INSERTION_SUCCESS_DEPTH_M = 0.030
INSERTION_SPEED_M_S = 0.020
INSERTION_IK_PERIOD_S = 1.0 / 30.0
INSERTION_VERIFY_HOLD_S = 0.15
INSERTION_RELEASE_POSITION_TOLERANCE_M = 0.0005
# Set to 0.0 to skip the extra post-opening delay during rollouts.
RELEASE_WAIT_S = 3.0
RELEASE_MODE = "fixed"
RELEASE_SETTLE_HOLD_S = 0.1
RELEASE_BOTTOM_TOLERANCE_M = 0.0002
RELEASE_MAX_PEG_SPEED_M_S = 0.005
TRACE_PERIOD_S = 0.05
ALIGN_MAX_TILT_DEG = 0.05
ALIGN_MAX_AXIS_ERROR_M = 0.00002
ALIGN_CLEARANCE_FRACTION = 0.5
ALIGN_FEEDBACK_PERIOD_S = 0.1
ALIGN_FEEDBACK_GAIN = 0.2

# Opt-in reference-generation experiment. Does not alter actuator/physics limits,
# insertion speed, depth, alignment tolerances or final contact acceptance gates.
EXPERT_FAST_OVERRIDES = dict(
    APPROACH_OPENING_M=0.020,
    ARM_COMMAND_SPEED_RAD_S=0.75,
    TRANSFER_SPEED_M_S=0.15,
    TRANSFER_ACCELERATION_M_S2=0.30,
    ALIGN_APPROACH_SPEED_M_S=0.075,
    ALIGN_APPROACH_ACCELERATION_M_S2=0.15,
    REACHED_HOLD_S=0.15,
    CLOSE_STABLE_S=0.10,
    LIFT_REACHED_HOLD_S=0.10,
    LIFT_HOLD_S=0.0,
    RELEASE_MODE="settled",
)

# Explicit numerical-performance experiments, not actuator/material changes.
# The collector restricts low_iter_candidate to nominal 0.50 mm clearance.
COLLECTION_SOLVER_PROFILES = {
    'baseline': {},
    'low_iter_candidate': dict(
        SCENE_MIN_POSITION_ITERATIONS=16, SCENE_MIN_VELOCITY_ITERATIONS=4,
        PEG_POSITION_ITERATIONS=16, PEG_VELOCITY_ITERATIONS=4),
}
