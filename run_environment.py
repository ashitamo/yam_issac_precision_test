"""YAM precision experiment scene. Run with the Isaac Lab Python environment."""

import argparse
import traceback
from pathlib import Path

import env_config as cfg

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--clearance_mm", type=float, choices=[1.0, 0.5, 0.25, 0.1], default=0.5)
parser.add_argument("--max_steps", type=int, default=0, help="Physics steps to run; zero keeps the viewer open.")
parser.add_argument("--expert", action="store_true", help="Run the single-position grasp/lift validation.")
parser.add_argument("--grid_id", default=None, help="Exact hole position, e.g. r08_c08, from the fixed candidate grid.")
parser.add_argument("--output_dir", type=Path, default=cfg.OUTPUT_DIR, help="Separate per-episode reports and traces.")
parser.add_argument("--camera_snapshot", type=Path, default=None,
                    help="Optional RGB snapshot after camera warmup.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
cfg.OUTPUT_DIR = args.output_dir.resolve()
cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
if args.grid_id is not None:
    from env_workspace_grid import get_grid_point
    cfg.WORKPIECE_POSITION_M = tuple(get_grid_point(args.grid_id)['hole_position_m'])
args.enable_cameras = True
print("[INIT] Starting Isaac Sim; RTX shader compilation may take time.", flush=True)
launcher = AppLauncher(args)
app = launcher.app
print("[INIT] Isaac Sim application ready.", flush=True)

import numpy as np
import torch
import isaaclab.sim as sim_utils
from env_yam import ParallelYamScene, YamPickInsertEnv
from env_expert import ParallelExpert
from env_dataset import WristCamera


def main():
    """Viewer and legacy validation CLI, backed by the shared environment."""
    from env_yam import create_simulation
    sim = create_simulation(args.device)
    sim.set_camera_view(cfg.OVERVIEW_EYE_M, cfg.OVERVIEW_TARGET_M)
    light = sim_utils.DomeLightCfg(intensity=cfg.LIGHT_INTENSITY)
    light.func("/World/Light", light)
    scene = ParallelYamScene(sim, num_envs=1, clearance_mm=args.clearance_mm)
    # The batched robot config contains a regex, not an authorable USD path.
    camera = WristCamera(scene.scene.env_prim_paths[0] + '/YAM/Geometry/world/base')
    sim.reset()
    env = YamPickInsertEnv(scene, clearance_mm=args.clearance_mm, max_episode_seconds=180.)
    obs, _ = env.reset()
    hold_action = torch.cat([obs['joint_position'], obs['gripper_opening']], dim=1).clone()
    expert = (ParallelExpert(scene, cfg.OUTPUT_DIR, args.clearance_mm,
        observation=obs, flat_single_report=True) if args.expert else None)
    print("[READY] Shared scene running. Close the window to exit.", flush=True)
    captured_stages = 0
    last_camera_time = None
    snapshot_saved = False
    while app.is_running() and (args.max_steps == 0 or env.global_steps < args.max_steps):
        if expert is None:
            action = hold_action
            callback = None
        else:
            action = expert.get_actions(env.control_period, observation=obs)
            callback = expert.observe
        render = env.global_steps % cfg.RENDER_EVERY_N_STEPS == 0
        obs, _, terminated, truncated, _ = env.step(action, render=render, physics_observer=callback)
        now = env.global_steps * env.physics_dt
        controller = expert.experts[0] if expert is not None else None
        stage_changed = controller is not None and len(controller.stage_results) > captured_stages
        snapshot_due = args.camera_snapshot is not None and not snapshot_saved and env.global_steps >= 16
        image_due = cfg.SAVE_VALIDATION_IMAGES and stage_changed
        frame_due = last_camera_time is None or now - last_camera_time >= cfg.CAMERA_PERIOD_S - 1e-9
        if frame_due or snapshot_due or image_due:
            frame = camera.capture(sim, now)
            if last_camera_time is None:
                print("[D405] RGB shape:", frame['rgb'].shape, "depth shape:", frame['depth'].shape, flush=True)
            last_camera_time = now
            if snapshot_due or image_due:
                from PIL import Image
                if snapshot_due:
                    args.camera_snapshot.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(frame['rgb']).save(args.camera_snapshot)
                    snapshot_saved = True
                if image_due:
                    image_dir = cfg.OUTPUT_DIR / "grasp_validation_images"
                    image_dir.mkdir(exist_ok=True)
                    Image.fromarray(frame['rgb']).save(image_dir / (controller.stage_results[-1]['stage'] + ".png"))
                    captured_stages = len(controller.stage_results)
        if controller is not None:
            if bool((terminated | truncated)[0]) and not controller.finished:
                controller.insertion_success = bool(env.success[0])
                controller.released_to_bottom = bool(env.success[0])
                controller.finish(bool(env.success[0]) and controller.grasp_success, env.reasons[0])
            if controller.finished:
                break
    if expert is not None:
        expert.finish_pending("Run ended before validation completed")
    print(f"[EXIT] Completed {env.global_steps} physics steps", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        app.close()
