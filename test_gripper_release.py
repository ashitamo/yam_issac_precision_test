"""Isolated release regression, with a deliberately initialized test pose."""
import sys
import argparse
import json
import numpy as np
import env_config as cfg
import env_expert

class ReleaseFixture(env_expert.GraspValidation):
    with_peg = False
    def __init__(self, robot, peg, *args, **kwargs):
        super().__init__(robot, peg, *args, **kwargs)
        self.fixture_target = np.array([0.35022, 0.08005, 0.03948])
        self.commanded_q = self.solve_ik(self.fixture_target)
        self.target_q = self.commanded_q.copy()
        self.stages = [("release", self.fixture_target, cfg.GRASP_OPENING_M)]
        q = robot.articulation.data.joint_pos.torch.clone()
        q[0, :6] = q.new_tensor(self.commanded_q)
        q[0, robot.gripper_joint_indices] = -0.00502
        robot.articulation.write_joint_state_to_sim_index(position=q, velocity=q * 0.0)

    def observe(self, dt):
        self.stage_time += dt
        if int(self.stage_time / dt) % 120 == 0:
            data = self.robot.articulation.data
            print("[FIXTURE]", self.stage_time,
                  "q", data.joint_pos.torch[0].tolist(),
                  "gap", self.robot.get_gripper_opening(), flush=True)
        if self.stage_time >= 3.0:
            self.finished = True
            opening = self.robot.get_gripper_opening()
            passed = abs(opening - cfg.GRASP_OPENING_M) < 0.0005
            result = dict(passed=passed, opening_m=opening, target_opening_m=cfg.GRASP_OPENING_M,
                          fixture="peg_in_hole" if self.with_peg else "empty",
                          note="Initialized test fixture, not a full pick-and-insert rollout")
            (cfg.OUTPUT_DIR / (result["fixture"] + ".json")).write_text(json.dumps(result, indent=2))
            print("[FIXTURE RESULT]", result, flush=True)
            if not passed:
                raise AssertionError("Gripper release failed")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--with_peg', action='store_true')
    args = parser.parse_args()
    cfg.RENDER_EVERY_N_STEPS = 100000
    cfg.CAMERA_PERIOD_S = 100000.0
    cfg.RUN_INSERTION = False
    if args.with_peg:
        cfg.PEG_POSITION_M = (0.35, 0.08, 0.0265)
    cfg.OUTPUT_DIR = cfg.PROJECT_ROOT / 'outputs' / 'release_diagnostics'
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ReleaseFixture.with_peg = args.with_peg
    original_expert = env_expert.GraspValidation
    env_expert.GraspValidation = ReleaseFixture
    sys.argv = ['run_environment.py', '--viz', 'none', '--expert', '--max_steps', '1000']
    import run_environment
    try:
        run_environment.main()
    finally:
        env_expert.GraspValidation = original_expert
        run_environment.app.close()


if __name__ == '__main__':
    main()
