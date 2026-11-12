"""Short native batched scene/reset smoke test; does not run insertion episodes."""
import argparse
import json
import time
from pathlib import Path


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--output_dir", type=Path, default=Path(__file__).parent / "outputs" / "parallel_smoke")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.steps < 2 or args.num_envs < 1:
        parser.error("steps >= 2 and num_envs >= 1 required")
    launcher = AppLauncher(args)
    try:
        import torch
        import isaaclab.sim as sim_utils
        from env_yam import create_simulation
        import env_config as cfg
        from env_yam import ParallelYamScene, tensor
        sim = create_simulation(args.device)
        light = sim_utils.DomeLightCfg(intensity=cfg.LIGHT_INTENSITY)
        light.func("/World/Light", light)
        sim.set_camera_view((4., 4., 3.), (0., 0., 0.))
        scene = ParallelYamScene(sim, args.num_envs)
        sim.reset()
        scene.reset()
        art = scene.robot.articulation
        assert tensor(art.data.joint_pos).shape[0] == args.num_envs
        before = tensor(art.data.joint_pos).clone()
        # Perturb only row zero; reset must restore it, without touching others.
        moved = before[:1].clone()
        moved[:, 0] += 0.05
        ids = torch.tensor([0], device=sim.device)
        art.write_joint_state_to_sim(position=moved, velocity=torch.zeros_like(moved), env_ids=ids)
        unaffected = tensor(art.data.joint_pos)[1:].clone()
        scene.reset([0])
        torch.testing.assert_close(tensor(art.data.joint_pos)[0], before[0])
        torch.testing.assert_close(tensor(art.data.joint_pos)[1:], unaffected, rtol=0, atol=0)
        start = time.perf_counter()
        for step in range(args.steps):
            scene.step(render=not args.headless and step % cfg.RENDER_EVERY_N_STEPS == 0)
        elapsed = time.perf_counter() - start
        q = tensor(art.data.joint_pos)
        local_peg = tensor(scene.peg.data.root_pos_w) - scene.scene.env_origins
        assert torch.isfinite(q).all() and torch.isfinite(local_peg).all()
        # Identical scenes should evolve similarly in their own local frames.
        q_spread = float((q - q[:1]).abs().max())
        peg_spread = float((local_peg - local_peg[:1]).abs().max())
        # Exercise reset again after contacts have evolved, including peg velocity.
        q_other = q[1:].clone()
        peg_other = tensor(scene.peg.data.root_pos_w)[1:].clone()
        vel_other = tensor(scene.peg.data.root_vel_w)[1:].clone()
        scene.reset([0])
        torch.testing.assert_close(tensor(art.data.joint_pos)[1:], q_other, rtol=0, atol=0)
        torch.testing.assert_close(tensor(scene.peg.data.root_pos_w)[1:], peg_other, rtol=0, atol=0)
        torch.testing.assert_close(tensor(scene.peg.data.root_vel_w)[1:], vel_other, rtol=0, atol=0)
        torch.testing.assert_close(tensor(scene.peg.data.root_vel_w)[0], torch.zeros(6, device=sim.device))
        passed = q_spread < 0.001 and peg_spread < 0.001
        report = dict(passed=passed, num_envs=args.num_envs,
            single_simulator=True, batched_articulation=True, cameras_enabled=False,
            partial_reset_preserves_other_joint_states=True,
            partial_reset_preserves_other_peg_states=True,
            episode_counts=scene.episode_count.cpu().tolist(),
            max_joint_difference_rad=q_spread, max_peg_local_difference_m=peg_spread,
            physics_steps=args.steps, wall_seconds=elapsed,
            environment_steps_per_second=args.steps * args.num_envs / elapsed,
            scope="reset and identical-scene smoke test only; not insertion validation")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
        if not passed:
            raise RuntimeError("Parallel scene equivalence check failed")
    finally:
        launcher.app.close()


if __name__ == "__main__":
    main()
