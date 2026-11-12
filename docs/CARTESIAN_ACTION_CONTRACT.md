# Cartesian action contract V1

`env_yam.CartesianYamEnv` 包住 `YamPickInsertEnv`，expert 與未來 policy
共用控制路徑；不改原有成功條件與物理設定。

## Action 與座標

每列 `[dx,dy,dz,rx,ry,rz,opening]`：平移為 BASE 軸向 m、rotation vector
為 BASE 軸向 rad（不是 Euler），opening 為兩指內側總開口絕對值 m。
以當前量測 TCP 組合：`p_target=p_measured+dp`、
`R_target=Exp(rotvec) @ R_measured`，旋轉左乘；pose quaternion 為 XYZW。
平移向量長度上限 20 mm，旋轉長度上限 0.2 rad。非法 policy action
直接拒絕，不默默裁切。

## 時序與實作

30 Hz action 只組合一次絕對 TCP 目標；servo 可選 240／120／60 Hz，
分別保持 8／4／2 個 tick，預設仍 240 Hz。物理保持 240 Hz 時，每 tick
分別走 1／2／4 個物理步；任務與 terminal 檢查仍逐物理步執行。
每 tick 用 Isaac Lab DifferentialIKController DLS、當前量測與原生 Jacobian
求解；低層保留 joint slew、限位與重力補償。重力前饋與目標按 servo 頻率
更新並保持，力矩預算不變。480 Hz 物理仍支援原介面回歸，但新 collector
目前只允許 240 Hz 物理。低頻 servo 的原生驗證範圍以收集報告為準。

Jacobian 取原生 link origin，加 TCP offset 並轉 BASE。目前 USD 有 world
anchor 但 PhysX 回報 floating root，因此驗證 anchor 後跳過 6 個 root DOF；
有原生 finite-difference 測試，不假設不同資產布局皆相同。

expert 舊 joint reference 可能超前實際手臂。`get_cartesian_actions()` 重用
既有 expert FK 轉 TCP，再明確限制 teacher reference delta；保存／執行的是
限制後 action，不是未執行的理想目標。執行端未新增自製 IK。

環境不自動 reset。policy window 中間終止時保留第一個 terminal snapshot；
recorder 最後 transition 可短於 1/30 s，不能拿 reset 後狀態替代。

## 驗證範圍

整併後重跑：`outputs/consolidated_cartesian_v1/cartesian_result.json`，原生
Jacobian 檢查通過、中心／遠角 2/2 成功，439 policy ticks／3512 物理步，
最大幾何數值與整併前一致。測試與環境 import 已更新，不保留舊 module wrapper。

- `unittest_cartesian.py`：座標、單次 delta、8 ticks、終止與局部 reset。
- `outputs/cartesian_native_smoke_v2`：小位移追蹤約 0.018 mm／0.006 度。
- `outputs/cartesian_expert_v3`：0.50 mm clearance 中心／遠角完整任務 2/2。
- `test_cartesian_adapter.py --expert` 可重跑原生測試。

這不是全 289 點或 learned policy 成功率。
