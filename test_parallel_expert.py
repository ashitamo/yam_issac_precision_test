"""Run a small batch of full tasks in one Isaac Sim process (no cameras)."""
import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--profile", action="store_true", help="Save wall-clock sections and cProfile call timings.")
    parser.add_argument("--expert_profile", choices=["baseline", "fast"], default="baseline")
    parser.add_argument("--physics_profile", choices=["baseline", "half_dt", "strong_solver"], default="baseline")
    parser.add_argument("--grid_ids", nargs="+", default=["r08_c08"])
    parser.add_argument("--clearance_mm", type=float, choices=[1., .5, .25, .1], default=.5)
    parser.add_argument("--max_sim_seconds", type=float, default=120.)
    parser.add_argument("--output_dir", type=Path, default=Path(__file__).parent / "outputs" / "parallel_expert")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs < 1 or len(args.grid_ids) not in (1, args.num_envs) or args.max_sim_seconds <= 0:
        parser.error("Provide one grid_id or exactly num_envs grid_ids, and positive count/duration")
    args.output_dir = args.output_dir.resolve()
    # Never overwrite a previous completed run accidentally.
    if (args.output_dir / "summary.json").exists() or any(args.output_dir.glob("env_*/pick_insert_result.json")):
        parser.error("Output already contains a run; choose a new --output_dir")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    grid_ids = args.grid_ids * args.num_envs if len(args.grid_ids) == 1 else args.grid_ids
    from env_workspace_grid import get_grid_point
    positions = [get_grid_point(grid_id)["hole_position_m"] for grid_id in grid_ids]
    import env_config as cfg
    overrides = cfg.EXPERT_FAST_OVERRIDES if args.expert_profile == "fast" else {}
    for name, value in overrides.items():
        setattr(cfg, name, value)
    from test_physics import PROFILES
    physics = PROFILES[args.physics_profile]
    cfg.PHYSICS_DT_S = 1. / physics['hz']
    cfg.PEG_POSITION_ITERATIONS = physics['peg_position']
    cfg.PEG_VELOCITY_ITERATIONS = physics['peg_velocity']
    cfg.SCENE_MIN_POSITION_ITERATIONS = physics['scene_position']
    cfg.SCENE_MIN_VELOCITY_ITERATIONS = physics['scene_velocity']
    (args.output_dir / "run.json").write_text(json.dumps(dict(
        status="starting", grid_ids=grid_ids, clearance_mm=args.clearance_mm,
        expert_profile=args.expert_profile, overrides=overrides,
        physics_profile=args.physics_profile, physics=physics,
        controller="existing SciPy IK per environment; batched physics and commands"), indent=2), encoding="utf-8")
    launcher = AppLauncher(args)
    profiler = None
    try:
        import isaaclab.sim as sim_utils
        from env_yam import create_simulation
        import env_config as cfg
        from env_task import contact_gate_pass, geometry_diagnostic_thresholds
        from env_yam import ParallelYamScene
        from env_expert import ParallelExpert
        sim = create_simulation(args.device)
        light = sim_utils.DomeLightCfg(intensity=cfg.LIGHT_INTENSITY)
        light.func("/World/Light", light)
        sim.set_camera_view((4., 4., 3.), (0., 0., 0.))
        scene = ParallelYamScene(sim, args.num_envs, args.clearance_mm, positions)
        sim.reset()
        scene.reset()
        collider_checks = scene.validate_workpiece_colliders()
        (args.output_dir / "collider_checks.json").write_text(json.dumps(collider_checks, indent=2), encoding="utf-8")
        if not all(check["passed"] for check in collider_checks):
            raise RuntimeError("Actual PhysX workpiece poses do not match requested holes; see collider_checks.json")
        expert = ParallelExpert(scene, args.output_dir, args.clearance_mm)
        control_every = round(cfg.CONTROL_PERIOD_S / cfg.PHYSICS_DT_S)
        if control_every < 1 or abs(control_every * cfg.PHYSICS_DT_S - cfg.CONTROL_PERIOD_S) > 1e-9:
            raise ValueError("Physics dt must divide the control period")
        start = time.perf_counter()
        steps = 0
        max_steps = round(args.max_sim_seconds / cfg.PHYSICS_DT_S)
        if args.profile:
            from env_profiling import RolloutProfiler
            profiler = RolloutProfiler()
        with (profiler.rollout() if profiler else nullcontext()), (profiler.engine_calls(sim) if profiler else nullcontext()):
            while launcher.app.is_running() and steps < max_steps and not expert.finished:
                if steps % control_every == 0:
                    with profiler.section('command') if profiler else nullcontext():
                        expert.command(cfg.CONTROL_PERIOD_S)
                with profiler.section('scene') if profiler else nullcontext():
                    scene.step(render=not args.headless and steps % cfg.RENDER_EVERY_N_STEPS == 0,
                               profiler=profiler)
                with profiler.section('observe') if profiler else nullcontext():
                    expert.observe(cfg.PHYSICS_DT_S)
                steps += 1
        expert.finish_pending("Batch ended before task completed")
        results = []
        for env_id, grid_id in enumerate(grid_ids):
            path = args.output_dir / f"env_{env_id:03d}" / "pick_insert_result.json"
            result = json.loads(path.read_text(encoding="utf-8"))
            results.append(dict(env_id=env_id, grid_id=grid_id, task_success=result["task_success"],
                contact_gate_pass=contact_gate_pass(result["max_insertion_radial_violation_m"],
                    result["max_insertion_bottom_penetration_m"], result['geometry_diagnostic_thresholds']),
                reason=result["reason"], simulated_seconds=expert.experts[env_id].time,
                result_file=str(path.relative_to(args.output_dir))))
        summary = dict(status="completed", num_envs=args.num_envs, physics_steps=steps,
            expert_profile=args.expert_profile, clearance_mm=args.clearance_mm,
            physics_profile=args.physics_profile, physics=physics,
            geometry_diagnostic_thresholds=geometry_diagnostic_thresholds(),
            physical_workpiece_poses_verified=True,
            wall_seconds=time.perf_counter() - start, results=results,
            all_passed=all(r["task_success"] and r["contact_gate_pass"] for r in results))
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
        if not summary["all_passed"]:
            raise RuntimeError("Some parallel tasks failed; see summary.json")
    except Exception:
        import traceback
        traceback.print_exc()
        raise
    finally:
        if profiler is not None:
            profiler.save(args.output_dir)
        launcher.app.close()


if __name__ == "__main__":
    main()
