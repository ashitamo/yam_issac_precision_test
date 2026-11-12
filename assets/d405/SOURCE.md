# YAM D405 wrist mount

Mesh and optical extrinsics are from MuJoCo Menagerie, revision
`f054586a8e90465d49ee5be15335c4a0c7f57caf`, file
`i2rt_yam/yam_linear_4310.xml` and asset
`i2rt_yam/assets/linear_4310/d405_wrist_linear_4310.stl`.

https://github.com/google-deepmind/mujoco_menagerie/tree/f054586a8e90465d49ee5be15335c4a0c7f57caf/i2rt_yam

The upstream model derives the mounting geometry from I2RT's linear_4310
station CAD at SDK v1.3.6. See the adjacent MIT LICENSE.

Optical origin in wrist frame is (0, 0.070435, 0.077006) m.
Optical quaternion WXYZ is (0, 0, 0.216440, -0.976296).
The optical frame has +Z forward and +Y down.
Nominal focal length is 1.93 mm and sensor size is 3.896 by 2.140 mm.

The experimental gripper visual transform matches the upstream link_6 mesh
transform to the rounding accuracy of the URDF. We therefore identify that
wrist frame with the local gripper frame. This is a model-based alignment
for the linear_4310 bracket and requires actual calibration for other brackets.

Housing and bracket are visual geometry attached to the gripper. Additional
camera/bracket mass, collision, stereo matching noise and depth failure on
black surfaces are not modeled in this version. Depth is ideal rendered depth.
