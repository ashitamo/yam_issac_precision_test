"""CPU tests. Run from project root: python -m unittest discover -s unittest -p 'unittest_*.py'."""
import ast
import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET
import numpy as np
import env_config as cfg
from env_workspace_grid import make_manifest, get_grid_point
from env_workspace_grid import YamKinematics
from env_expert import GraspValidation


class EnvironmentValidationTests(unittest.TestCase):
    def test_consolidated_modules_have_no_stale_runtime_imports(self):
        removed = {'env_parallel', 'env_cartesian', 'env_parallel_expert',
                   'env_camera', 'env_recorder', 'yam_kinematics'}
        for name in removed:
            self.assertFalse((cfg.PROJECT_ROOT / (name + '.py')).exists())
        for path in cfg.PROJECT_ROOT.glob('*.py'):
            tree = ast.parse(path.read_text(encoding='utf-8-sig'))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    self.assertNotIn(node.module, removed, str(path))
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertNotIn(alias.name, removed, str(path))
        import env_yam
        import env_dataset
        import env_expert
        import env_workspace_grid
        for module, names in ((env_yam, ('ParallelYamScene', 'YamPickInsertEnv', 'CartesianYamEnv')),
                              (env_dataset, ('WristCamera', 'EpisodeRecorder')),
                              (env_expert, ('GraspValidation', 'ParallelExpert')),
                              (env_workspace_grid, ('YamKinematics', 'make_manifest'))):
            for name in names:
                self.assertTrue(callable(getattr(module, name)))

    def test_geometry_tolerance_is_separate_from_physics_and_has_exact_boundary(self):
        from env_task import contact_gate_pass
        self.assertEqual(cfg.OBJECT_CONTACT_OFFSET_M, .00002)
        self.assertEqual(cfg.RADIAL_GEOMETRY_TOLERANCE_M, .00005)
        self.assertEqual(cfg.BOTTOM_GEOMETRY_TOLERANCE_M, .00005)
        self.assertTrue(contact_gate_pass(.00005, .00005))
        self.assertFalse(contact_gate_pass(.000050001, 0.))
        self.assertFalse(contact_gate_pass(0., .000050001))
        self.assertFalse(contact_gate_pass(float('nan'), 0.))
        self.assertFalse(contact_gate_pass(None, 0.))
        self.assertTrue(contact_gate_pass(.0000381, .0000075))
        self.assertFalse(contact_gate_pass(.0000381, .0000075, dict(radial_m=.00002, bottom_m=.00002)))

    def test_summary_preserves_legacy_threshold_instead_of_relabelling_old_data(self):
        from test_physics import summarize, write_json
        case = dict(case_id='legacy', clearance_mm=.1, grid_id='r08_c08', profile='baseline')
        result = dict(status='completed', task_success=True, reason='test',
            final_insertion_depth_m=.032, preinsertion_xy_error_m=.00001,
            max_insertion_radial_violation_m=.00003, max_insertion_bottom_penetration_m=0.)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / 'legacy'
            folder.mkdir()
            write_json(folder / 'case.json', case)
            write_json(folder / 'pick_insert_result.json', result)
            self.assertFalse(summarize(root, [case])['cases'][0]['contact_diagnostic_pass'])
            result['geometry_diagnostic_thresholds'] = dict(radial_m=.00005, bottom_m=.00005)
            write_json(folder / 'pick_insert_result.json', result)
            self.assertTrue(summarize(root, [case])['cases'][0]['contact_diagnostic_pass'])

    def test_fast_profile_does_not_change_physics_or_success_tolerances(self):
        allowed = {'APPROACH_OPENING_M', 'ARM_COMMAND_SPEED_RAD_S', 'TRANSFER_SPEED_M_S',
            'TRANSFER_ACCELERATION_M_S2', 'ALIGN_APPROACH_SPEED_M_S',
            'ALIGN_APPROACH_ACCELERATION_M_S2', 'REACHED_HOLD_S', 'CLOSE_STABLE_S',
            'LIFT_REACHED_HOLD_S', 'LIFT_HOLD_S', 'RELEASE_MODE'}
        self.assertEqual(set(cfg.EXPERT_FAST_OVERRIDES), allowed)
        self.assertEqual(cfg.RELEASE_MODE, 'fixed')
        self.assertEqual(cfg.RELEASE_WAIT_S, 3.)
        self.assertGreater(cfg.EXPERT_FAST_OVERRIDES['APPROACH_OPENING_M'], cfg.HOLE_DIAMETER_MM / 1000)
        self.assertLess(cfg.EXPERT_FAST_OVERRIDES['ARM_COMMAND_SPEED_RAD_S'], 3.)

    def test_release_ready_requires_open_bottom_depth_speed_and_hole(self):
        expert = object.__new__(GraspValidation)
        expert.insertion_depth_m = .032
        self.assertTrue(expert._release_ready(True, 0., 0., True))
        for values in ((False, 0., 0., True), (True, .001, 0., True),
                       (True, 0., .01, True), (True, 0., 0., False)):
            self.assertFalse(expert._release_ready(*values))
        expert.insertion_depth_m = .029
        self.assertFalse(expert._release_ready(True, 0., 0., True))

    def test_engine_profiler_restores_interface_on_exception(self):
        from env_profiling import RolloutProfiler
        interface = SimpleNamespace(simulate=lambda *a: None, fetch_results=lambda: None)
        manager = SimpleNamespace(_physx_sim=interface)
        profiler = RolloutProfiler()
        with self.assertRaises(ValueError):
            with profiler.engine_calls(manager):
                manager._physx_sim.simulate(.01, 0.)
                manager._physx_sim.fetch_results()
                raise ValueError('test')
        self.assertIs(manager._physx_sim, interface)
        self.assertEqual(profiler.sections['engine.simulate']['count'], 1)
        self.assertEqual(profiler.sections['engine.fetch_results']['count'], 1)

    def test_rollout_profiler_records_sections_and_saves_on_exception(self):
        from env_profiling import RolloutProfiler
        profiler = RolloutProfiler()
        with self.assertRaises(ValueError):
            with profiler.rollout():
                with profiler.section('command'):
                    sum(range(20))
                with profiler.section('observe'):
                    raise ValueError('test')
        with tempfile.TemporaryDirectory() as directory:
            report = profiler.save(directory)
            self.assertEqual(report['sections']['command']['count'], 1)
            self.assertEqual(report['sections']['observe']['count'], 1)
            self.assertGreater(report['wall_seconds'], 0.)
            self.assertTrue(report['function_timings'])
            for name in ('profile.json', 'profile.txt', 'rollout.pstats'):
                self.assertTrue((Path(directory) / name).is_file())

    def test_grid_size_and_checkerboard(self):
        points = make_manifest()['points']
        self.assertEqual(len(points), 289)
        self.assertEqual(len({p['grid_id'] for p in points}), 289)
        self.assertEqual(sum(p['candidate_group'] == 'seen' for p in points), 145)
        self.assertEqual(sum(p['candidate_group'] == 'unseen' for p in points), 144)

    def test_grid_spacing_and_extent(self):
        xy = np.array([p['hole_position_m'][:2] for p in make_manifest()['points']])
        np.testing.assert_allclose(np.ptp(xy, axis=0), [.2, .2], atol=1e-14)
        np.testing.assert_allclose(np.diff(np.unique(xy[:, 0])), .0125, atol=1e-14)
        np.testing.assert_allclose(np.diff(np.unique(xy[:, 1])), .0125, atol=1e-14)
        self.assertEqual(make_manifest(), make_manifest())

    def test_center_and_non_training_labels(self):
        np.testing.assert_allclose(get_grid_point('r08_c08')['hole_position_m'][:2], cfg.GRID_CENTER_XY_M)
        self.assertEqual(make_manifest()['split_status'], 'candidate_only_not_training_split')

    def test_all_initial_hole_positions_clear_the_fixed_peg(self):
        points = make_manifest()['points']
        self.assertTrue(all(p['initial_scene_clear'] for p in points))
        self.assertGreater(min(p['initial_peg_surface_gap_m'] for p in points), .02)

    def test_fk_matches_expert_urdf_fk(self):
        path = cfg.PROJECT_ROOT / 'assets/yam_robot_with_gripper/config/cumotion/yam.urdf'
        model = YamKinematics(path)
        # Exercise the exact FK method used by the simulator expert, without Isaac.
        expert = object.__new__(GraspValidation)
        root = ET.parse(path).getroot()
        expert.joints = [root.find(f"joint[@name='joint{i}']") for i in range(1, 7)]
        from types import SimpleNamespace
        expert.robot = SimpleNamespace(config=SimpleNamespace(ee_position_in_gripper_m=model.tcp))
        rng = np.random.default_rng(42)
        for _ in range(100):
            q = rng.uniform(model.lower, model.upper)
            p1, r1 = model.fk(q)
            p2, r2 = expert.fk(q)
            np.testing.assert_allclose(p1, p2, atol=1e-12)
            self.assertLess((r1.inv() * r2).magnitude(), 1e-12)

    def test_ik_roundtrip_and_unreachable(self):
        model = YamKinematics(cfg.PROJECT_ROOT / 'assets/yam_robot_with_gripper/config/cumotion/yam.urdf')
        q = np.array([0., 1.8, 1.4, -1.0, 0., 0.])
        p, r = model.fk(q)
        self.assertTrue(model.solve(p, r, q)['reachable'])
        self.assertFalse(model.solve(np.array([10., 0., 0.]), r, q)['reachable'])

    def test_physics_summary_rejects_missing_and_failed_results(self):
        from test_physics import summarize, PROFILES
        cases = [dict(case_id=p, clearance_mm=.1, grid_id='r08_c08', profile=p) for p in PROFILES]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.assertFalse(summarize(root, cases)['all_profile_comparisons_pass'])
            for case in cases:
                folder = root / case['case_id']
                folder.mkdir()
                (folder / 'case.json').write_text(json.dumps(case))
                result = dict(status='completed', task_success=True, final_insertion_depth_m=.032,
                              preinsertion_xy_error_m=.00002, max_insertion_radial_violation_m=0.,
                              max_insertion_bottom_penetration_m=.000001)
                (folder / 'pick_insert_result.json').write_text(json.dumps(result))
            self.assertTrue(summarize(root, cases)['all_profile_comparisons_pass'])
            result['task_success'] = False
            (folder / 'pick_insert_result.json').write_text(json.dumps(result))
            self.assertFalse(summarize(root, cases)['all_profile_comparisons_pass'])

    def make_feedback_fixture(self, enabled):
        import torch
        from scipy.spatial.transform import Rotation
        expert = object.__new__(GraspValidation)
        expert.time = 0.
        expert.env_id = 0
        expert.defer_commands = False
        expert.stage_time = 0.
        expert.stage_index = 0
        expert.stages = [('align', np.array([0., 0., .035]), .06)]
        expert.hole_top = np.array([0., 0., .032])
        expert.orientation = Rotation.identity()
        expert.alignment_feedback_enabled = enabled
        expert.align_start_position = np.array([0., 0., .035])
        expert.align_goal_position = expert.align_start_position.copy()
        expert.align_approach_duration_s = 1.0
        expert.last_align_approach_ik_time = -1.
        expert.last_alignment_time = -1.
        expert.measured_axis = np.array([0., 0., 1.])
        expert.target_q = np.zeros(6)
        expert.commanded_q = np.zeros(6)
        expert.solve_ik = lambda *a, **kw: np.zeros(6)
        articulation = SimpleNamespace(
            data=SimpleNamespace(gravity_compensation_forces=SimpleNamespace(torch=torch.zeros(1, 8))),
            num_base_dofs=0, set_joint_effort_target=lambda value: None)
        expert.robot = SimpleNamespace(
            articulation=articulation, gripper_joint_indices=[6, 7],
            get_ee_pose_base=lambda: (expert.stages[0][1] + [.0001, 0., 0.], Rotation.identity()),
            opening_to_joint_position=lambda opening: -.5 * opening,
            set_feedforward_effort_target=lambda effort: None,
            set_joint_position_target=lambda command: None)
        return expert

    def test_parallel_expert_defers_writes_and_keeps_commands_independent(self):
        first = self.make_feedback_fixture(False)
        second = self.make_feedback_fixture(False)
        for expert in (first, second):
            expert.defer_commands = True
            expert.stages[0] = ('hold', expert.stages[0][1], .06)
            expert.robot.set_joint_position_target = lambda *a: self.fail("Unexpected per-row write")
            expert.robot.set_feedforward_effort_target = lambda *a: self.fail("Unexpected per-row FF write")
        second.target_q[:] = .2
        first.command(.1)
        second.command(.1)
        self.assertEqual(first.pending_joint_command['joint1'], 0.)
        self.assertGreater(second.pending_joint_command['joint1'], 0.)
        self.assertIsNot(first.pending_joint_command, second.pending_joint_command)

    def test_parallel_expert_pose_and_opening_select_requested_row(self):
        expert = self.make_feedback_fixture(False)
        expert.env_id = 3
        expert.robot.get_ee_pose_base = lambda index: ('pose', index)
        expert.robot.get_gripper_opening = lambda index: index / 1000
        self.assertEqual(expert._ee_pose(), ('pose', 3))
        self.assertEqual(expert._opening(), .003)

    def test_parallel_expert_reset_preserves_active_rows_and_old_reports(self):
        import torch
        source = ast.parse((cfg.PROJECT_ROOT / 'env_expert.py').read_text(encoding='utf-8'))
        node = next(item for item in source.body if isinstance(item, ast.ClassDef) and item.name == 'ParallelExpert')
        namespace = dict(Path=Path, tensor=lambda value: value,
            GraspValidation=lambda *a, **kw: SimpleNamespace(finished=False, **kw))
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<parallel expert>', 'exec'), namespace)
        positions = torch.ones(2, 8)
        reset_calls = []
        def reset(ids):
            reset_calls.append(list(ids))
            positions[ids] = 0.
        scene = SimpleNamespace(num_envs=2, scene=SimpleNamespace(env_origins=torch.zeros(2, 3)),
            robot=SimpleNamespace(articulation=SimpleNamespace(data=SimpleNamespace(joint_pos=positions))),
            peg=None, hole_radius_m=.005, workpiece_positions=np.zeros((2, 3)), reset=reset)
        manager = namespace['ParallelExpert'](scene, Path('diagnostic'))
        previous = manager.experts.copy()
        previous[0].finished = True
        manager.reset([0])
        self.assertEqual(reset_calls, [[0]])
        self.assertIs(manager.experts[1], previous[1])
        self.assertIsNot(manager.experts[0], previous[0])
        self.assertEqual(manager.episode_ids, [1, 0])
        self.assertEqual(manager.experts[0].output_dir, Path('diagnostic/env_000/episode_001'))
        self.assertEqual(previous[0].output_dir, Path('diagnostic/env_000'))
        torch.testing.assert_close(manager.target[1], torch.ones(8))
        with self.assertRaises(ValueError):
            manager.reset([1])
        with self.assertRaises(ValueError):
            manager.reset([0, 0])
        self.assertEqual(reset_calls, [[0]])

    def test_alignment_corrects_constant_tracking_bias(self):
        expert = self.make_feedback_fixture(True)
        for _ in range(12):
            expert.measured_tip = expert.robot.get_ee_pose_base()[0]
            expert.command(.1)
            expert.stage_time += .1
        actual_tip = expert.robot.get_ee_pose_base()[0]
        self.assertLess(abs(actual_tip[0]), .00002)

    def test_alignment_does_not_integrate_during_approach(self):
        expert = self.make_feedback_fixture(False)
        expert.measured_tip = np.array([.01, 0., .1])
        before = expert.stages[0][1].copy()
        expert.command(.1)
        np.testing.assert_array_equal(expert.stages[0][1], before)

    def test_single_profile_is_not_a_failed_convergence_comparison(self):
        from test_physics import summarize, write_json
        case = dict(case_id='one', clearance_mm=.5, grid_id='r08_c08', profile='baseline')
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / 'one'
            folder.mkdir()
            write_json(folder / 'case.json', case)
            write_json(folder / 'pick_insert_result.json', dict(status='completed', task_success=True,
                       final_insertion_depth_m=.032, preinsertion_xy_error_m=.00001,
                       max_insertion_radial_violation_m=0., max_insertion_bottom_penetration_m=0.))
            result = summarize(root, [case])
            self.assertTrue(result['all_requested_cases_pass'])
            self.assertIsNone(result['all_profile_comparisons_pass'])
            self.assertEqual(result['progress']['fully_passed'], 1)

    def test_campaign_does_not_certify_a_partial_grid(self):
        from test_environment_A import update_review, pid_running
        from test_physics import write_json
        import os
        self.assertTrue(pid_running(os.getpid()))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'center_c0p10').mkdir()
            (root / 'grid_c0p50').mkdir()
            write_json(root / 'center_c0p10/physics_summary.json', dict(
                progress=dict(total=3, finished=3), all_requested_cases_pass=True, all_profile_comparisons_pass=True))
            write_json(root / 'grid_c0p50/physics_summary.json', dict(
                progress=dict(total=289, finished=288), all_requested_cases_pass=True))
            self.assertFalse(update_review(root, 'running', 'unit-test')['A_requested_validation_pass'])
            write_json(root / 'grid_c0p50/physics_summary.json', dict(
                progress=dict(total=289, finished=289), all_requested_cases_pass=True))
            self.assertTrue(update_review(root, 'completed', 'unit-test')['A_requested_validation_pass'])

    def test_proximal_collision_clip_preserves_closed_box(self):
        import trimesh
        from scipy.spatial import ConvexHull
        # Load the exact pure static method without importing Kit/Isaac, which
        # robot_yam normally requires to have been initialized by AppLauncher.
        source = ast.parse((cfg.PROJECT_ROOT / 'robot_yam.py').read_text(encoding='utf-8'))
        robot_class = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'YamRobot')
        method = next(n for n in robot_class.body if isinstance(n, ast.FunctionDef) and n.name == '_clip_proximal_mesh')
        method.decorator_list = []
        namespace = {'np': np, 'ConvexHull': ConvexHull}
        exec(compile(ast.Module(body=[method], type_ignores=[]), 'robot_yam.py', 'exec'), namespace)
        clip_proximal_mesh = namespace['_clip_proximal_mesh']
        box = trimesh.creation.box(extents=[.004, .015, .02])
        vertices, faces = clip_proximal_mesh(box.vertices, box.faces, .002)
        clipped = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
        self.assertLessEqual(vertices[:, 2].max(), .002 + 1e-10)
        self.assertTrue(clipped.is_watertight)
        self.assertAlmostEqual(abs(clipped.volume), .004 * .015 * .012, places=12)

    def test_arm_motor_cap_matches_urdf(self):
        source = ast.parse((cfg.PROJECT_ROOT / 'robot_yam.py').read_text(encoding='utf-8'))
        config = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'YamRobotConfig')
        cap = next(ast.literal_eval(n.value) for n in config.body if isinstance(n, ast.AnnAssign)
                   and isinstance(n.target, ast.Name) and n.target.id == 'arm_effort_limit_sim')
        root = ET.parse(cfg.PROJECT_ROOT / 'assets/yam_robot_with_gripper/config/cumotion/yam.urdf').getroot()
        for i in range(1, 7):
            self.assertEqual(cap, float(root.find(f"joint[@name='joint{i}']/limit").get('effort')))

    def test_cartesian_reference_velocity_and_acceleration_bounds(self):
        for distance, speed, acceleration in [(.253, .1, .2), (.067, .05, .1), (.001, .1, .2)]:
            duration = GraspValidation._cartesian_duration(distance, speed, acceleration)
            u = np.linspace(0., 1., 10001)
            velocity = distance / duration * (30*u**2 - 60*u**3 + 30*u**4)
            accel = distance / duration**2 * (60*u - 180*u**2 + 120*u**3)
            self.assertLessEqual(np.max(np.abs(velocity)), speed + 1e-8)
            self.assertLessEqual(np.max(np.abs(accel)), acceleration + 1e-8)

    def test_cartesian_transfer_preserves_height_and_endpoint(self):
        expert = self.make_feedback_fixture(False)
        start, goal = np.array([.3, -.08, .14]), np.array([.365, .165, .14])
        expert.stages = [('transfer', start.copy(), .06)]
        expert.transfer_start_position, expert.transfer_goal_position = start, goal
        expert.transfer_duration_s = GraspValidation._cartesian_duration(np.linalg.norm(goal-start), .1, .2)
        before = expert.orientation.as_matrix().copy()
        for fraction in np.linspace(0., 1., 11):
            expert.stage_time = fraction * expert.transfer_duration_s
            expert.last_transfer_ik_time = -1.
            expert.command(.1)
            self.assertAlmostEqual(expert.stages[0][1][2], .14, places=12)
        np.testing.assert_allclose(expert.stages[0][1], goal, atol=1e-12)
        np.testing.assert_array_equal(expert.orientation.as_matrix(), before)

    def test_feedforward_and_drive_total_stays_inside_motor_cap(self):
        import torch
        source = ast.parse((cfg.PROJECT_ROOT / 'robot_yam.py').read_text(encoding='utf-8'))
        robot = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'YamRobot')
        method = next(n for n in robot.body if isinstance(n, ast.FunctionDef) and n.name == '_arm_effort_budget')
        method.decorator_list = []
        namespace = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), 'robot_yam.py', 'exec'), namespace)
        ff, budget = namespace['_arm_effort_budget'](torch.linspace(-15., 15., 301), 10.)
        self.assertTrue(torch.all(torch.abs(ff + budget) <= 10.000001).item())
        self.assertTrue(torch.all(torch.abs(ff - budget) <= 10.000001).item())


if __name__ == '__main__':
    unittest.main()
