"""CPU checks of Lab 2.1 API boundaries; not native PhysX certification."""
import ast
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import unittest
from unittest.mock import patch
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from env_yam import wxyz_to_xyzw, xyzw_to_wxyz, gravity_compensation, _WxyzIKBoundary, create_simulation
import env_config as cfg


class CompatibilityTests(unittest.TestCase):
    def test_factory_uses_old_physx_config_and_rejects_unsupported_options(self):
        calls = []
        sim_module = ModuleType('isaaclab.sim')
        sim_module.PhysxCfg = lambda **kwargs: SimpleNamespace(**kwargs)
        sim_module.SimulationCfg = lambda **kwargs: SimpleNamespace(physics_prim_path='/physicsScene', **kwargs)
        sim_module.SimulationContext = lambda config: calls.append(config) or SimpleNamespace(cfg=config)
        sim_module.get_current_stage = lambda: None
        package = ModuleType('isaaclab')
        package.sim = sim_module
        pxr = ModuleType('pxr')
        pxr.PhysxSchema = SimpleNamespace(PhysxSceneAPI=SimpleNamespace(Get=lambda *args: SimpleNamespace()))
        with patch.dict(sys.modules, {'isaaclab': package, 'isaaclab.sim': sim_module, 'pxr': pxr}), patch(
                'importlib.metadata.version', side_effect=lambda key: '4.5.0' if key == 'isaacsim' else '2.1.0'):
            sim = create_simulation('cuda:0')
            self.assertEqual(len(calls), 1)
            self.assertEqual(sim.cfg.device, 'cuda:0')
            self.assertEqual(sim.cfg.physx.min_position_iteration_count, cfg.SCENE_MIN_POSITION_ITERATIONS)
            with patch.object(cfg, 'SOLVE_ARTICULATION_CONTACT_LAST', True):
                with self.assertRaisesRegex(RuntimeError, 'unsupported'):
                    create_simulation('cuda:0')
        with patch('importlib.metadata.version', return_value='6.0.1.0'):
            with self.assertRaisesRegex(RuntimeError, 'requires'):
                create_simulation('cuda:0')

    def test_quaternion_roundtrip_and_identity(self):
        native = torch.tensor([[1., 0., 0., 0.], [.5, -.5, .5, -.5]])
        torch.testing.assert_close(xyzw_to_wxyz(wxyz_to_xyzw(native)), native)
        torch.testing.assert_close(wxyz_to_xyzw(native)[0], torch.tensor([0., 0., 0., 1.]))

    def test_ik_converts_both_target_and_measured_orientation(self):
        calls = {}
        backend = SimpleNamespace(
            set_command=lambda value: calls.update(target=value),
            compute=lambda p, q, j, v: calls.update(measured=q) or v,
            reset=lambda ids: calls.update(reset=ids))
        adapter = _WxyzIKBoundary(backend)
        pose = torch.tensor([[.1, .2, .3, 0., 0., 0., 1.]])
        adapter.set_command(pose)
        adapter.compute(pose[:, :3], pose[:, 3:], None, torch.zeros(1, 6))
        torch.testing.assert_close(calls['target'][:, 3:], torch.tensor([[1., 0., 0., 0.]]))
        torch.testing.assert_close(calls['measured'], calls['target'][:, 3:])
        torch.testing.assert_close(pose[:, 3:], torch.tensor([[0., 0., 0., 1.]]))
        adapter.reset([0])
        self.assertEqual(calls['reset'], [0])

    def test_gravity_root_columns_and_invalid_layout(self):
        for fixed, offset in ((True, 0), (False, 0), (False, 6)):
            force = torch.arange(8 + offset).reshape(1, -1)
            art = SimpleNamespace(num_joints=8, is_fixed_base=fixed,
                root_physx_view=SimpleNamespace(get_gravity_compensation_forces=lambda: force))
            torch.testing.assert_close(gravity_compensation(art), force[:, offset:])
        art.is_fixed_base = True
        with self.assertRaises(RuntimeError):
            gravity_compensation(art)

    def test_native_jacobian_com_shift_base_rotation_and_root_columns(self):
        tree = ast.parse((Path(__file__).resolve().parents[1] / 'robot_yam.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'YamRobot')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'get_ee_jacobian_base')
        namespace = dict(torch=torch)
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<Jacobian boundary>', 'exec'), namespace)
        math = ModuleType('isaaclab.utils.math')
        def apply(q, v):
            return torch.as_tensor(Rotation.from_quat(q.numpy()[:, [1, 2, 3, 0]]).apply(v.numpy()), dtype=v.dtype)
        math.quat_apply = apply
        math.quat_conjugate = lambda q: q * torch.tensor([1., -1., -1., -1.])
        rotation = torch.tensor([[np.sqrt(.5), 0., 0., np.sqrt(.5)]], dtype=torch.float32)
        with patch.dict(sys.modules, {'isaaclab.utils.math': math}):
            for fixed, root in ((True, 0), (False, 6)):
                raw = torch.zeros(1, 1 if fixed else 2, 6, 8 + root)
                raw[0, 0 if fixed else 1, 5, root] = 1.
                data = SimpleNamespace(root_quat_w=rotation,
                    body_quat_w=rotation[:, None, :].repeat(1, 2, 1),
                    com_pos_b=torch.tensor([[[0., 0., 0.], [.02, 0., 0.]]]))
                art = SimpleNamespace(data=data, is_fixed_base=fixed, num_joints=8,
                    root_physx_view=SimpleNamespace(get_jacobians=lambda: raw))
                robot = SimpleNamespace(articulation=art, tool_body_index=1,
                    arm_joint_indices=list(range(6)), _cartesian_anchor_checked=True,
                    config=SimpleNamespace(ee_position_in_gripper_m=[.12, 0., 0.]))
                actual = namespace['get_ee_jacobian_base'](robot)
                torch.testing.assert_close(actual[0, :, 0], torch.tensor([0., .1, 0., 0., 0., 1.]), atol=1e-6, rtol=1e-6)
                self.assertEqual(tuple(actual.shape), (1, 6, 6))
