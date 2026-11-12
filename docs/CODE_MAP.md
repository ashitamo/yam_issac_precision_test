# 程式用途與整併後結構

## 目前收集路徑

`record_expert_data.py` 已取代舊 test_dataset_pilot.py；各回合存 data/，
四個孔位各兩回合的收集與回播 8/8 通過。支援固定孔位 barrier reset，
另已提供 --all_grid 四公差各 289 點的跨批次 supervisor；正式 split 尚未建立。
Ubuntu／雙卡與中斷資料接手現況見 [HANDOVER](../HANDOVER.md)。
53 個 CPU 測試與四相機 native preflight 通過。效能數據及低迭代候選限制
見 EXPERT_SPEED_VALIDATION.md；不能把 CPU 失敗案例算成可用吞吐。

## 先前整併回歸（歷史）

- 根目錄 Python 由 31 支減為 26 支；6 個舊模組刪除、新增合併的 env_dataset。
- 46 個 CPU 單位測試通過，包括舊 module import 殘留檢查。
- `outputs/consolidated_cartesian_v1/`：中心／遠角原生完整任務 2/2，Jacobian 檢查通過。
- `data/consolidated_rgbd_pilot_v1/`：399 action／400 幀 RGB-D，時間對齊通過；
  保存 action 回播成功，最大 TCP 差 0.214 mm。舊 pilot 未覆寫。
- 僅整併結構，未改控制／物理頻率、PD、摩擦力、成功門檻；不宣稱效能改善。
- `outputs/consolidated_viewer_smoke_v2/`：共用環境 viewer 入口原生 24 步通過，
  已保存並目視檢查 640×360 相機影像。v1 因把 batched regex 當 USD 建立路徑失敗，
  v2 使用實際 env prim 路徑修正；這是 headless smoke，不是 GUI 效能或整回合 joint expert 驗證。

舊報告 metadata 仍保留當時檔名／hash，不修改為新版本。來源 fingerprint 改變，
舊驗證 resume 若拒絕混用版本，應開新輸出目錄，不要強行繞過檢查。

## 日常入口

| 程式 | 用途 |
| --- | --- |
| `run_environment.py` | 開啟共用場景、腕部相機；可選 joint-space expert |
| `record_expert_data.py` | 多環境 Cartesian expert RGB-D 收集、分塊 HDF5、可選保存 action 回播 |

兩個入口都使用同一個 ParallelYamScene／YamPickInsertEnv，不再各寫場景。
它們保留不同控制模式：舊 viewer 使用 joint-space，pilot 使用 Cartesian adapter；
不把兩種模式的結果混為一談。

## 核心模組（自動 import，不需要單獨啟動）

| 程式 | Review 範圍 |
| --- | --- |
| `env_yam.py` | 依序看 ParallelYamScene（場景／物理／reset）、YamPickInsertEnv（joint action／觀測／任務生命週期）、CartesianYamEnv（policy action／原生 DIK） |
| `env_expert.py` | GraspValidation（階段流程）＋ParallelExpert（每列 action 與回合調度） |
| `env_dataset.py` | WristCamera／WristCameraBatch（批次 RGB-D）＋EpisodeRecorder（分塊／對齊／HDF5） |
| `env_workspace_grid.py` | YamKinematics（原有 CPU 篩查）＋grid IDs／manifest |
| `robot_yam.py` | 機器人、碰撞、關節控制、原生 TCP Jacobian |
| `env_config.py` | 集中參數與資產路徑 |
| `env_task.py` | 環境、expert、離線報告共用的成功／幾何判斷 |
| `env_profiling.py` | 效能計時輔助；collector 的 --profile 才額外包裝原生引擎呼叫 |

這次已移除的舊模組（不是 compatibility wrapper）：
env_parallel.py、env_cartesian.py、env_parallel_expert.py、env_camera.py、
env_recorder.py、yam_kinematics.py。所有目前根目錄程式已更新 import；
archive／validation 的歷史 source_snapshot 不改寫。

## 驗證工具

### 共用介面要點

AppLauncher 先啟動，再建立 SimulationContext、ParallelYamScene，呼叫 sim.reset()。
YamPickInsertEnv 提供 reset(env_ids)、step(actions)、get_observations()；
CartesianYamEnv 包裝它提供 policy action，契約見 CARTESIAN_ACTION_CONTRACT.md。
joint action 為 joint1..6 絕對 rad＋總開口 m，Cartesian action 不可混用。

環境不自動 reset。第一次 reset 全部列，之後可局部 reset；全域物理時間不歸零，
指定列 episode_time 歸零、episode_id 遞增。先保存 terminal snapshot，再 reset，
不能把下一回合初始影像當作前一回合結尾。
terminated 表示成功／已回報失敗，truncated 表示 timeout；success 需另讀。
reward 是新成功當步一次 +1，不是完整 RL reward。privileged 是 expert 用真值，
不可直接當成視覺 policy observation。

目前多環境僅以平移分隔、BASE 軸向一致，不支援任意旋轉 base 的任務座標。
不同孔位在初始化時建立固定碰撞體，不在運行中移動靜態工件；每列 robot／peg
reset 不影響其他列。名義 IK 篩查不代表完整掃掠碰撞或實機精度保證。
正式 split 尚未完成；全網格改以新場景分批收集，不在運行中搬靜態工件。
checkerboard candidate 標籤不是訓練分割。

### 工具清單

| 程式 | 何時需要 |
| --- | --- |
| `test_cartesian_adapter.py` | action／IK／servo 改動後 |
| `test_env_interface.py` | step、reset、終止介面改動後 |
| `test_repeated_rollout.py` | 多回合／局部 reset |
| `test_parallel_environments.py` | 多環境碰撞隔離 smoke |
| `test_parallel_expert.py` | joint-space 多環境完整任務 |
| `test_rollout_throughput.py` | 批次效能比較 |
| `test_precision_subset.py` | 少量 0.10 mm／solver 比較 |
| `test_physics.py` | 數值物理驗證與報告 |
| `test_environment_A.py` | 舊循序 A 排程，不是正式 collector |
| `test_assets.py` | 資產／限制／安裝版本 |
| `test_gripper_release.py` | 隔離張爪 |
| `unit_test/unittest_*.py`（5 支） | CPU 單元測試，不開模擬器 |

`test_*.py` 驗證工具保留根目錄；單元測試集中在 `unit_test/`。它們不會被 collector 自動啟動，
也未為了減少檔案而塞成帶大量模式的單一腳本。

## 資料夾

- `data/`：dataset、metadata、試收預覽與對齊摘要。
- `outputs/`：一般 demo、診斷與效能輸出。
- `validation/`：需保留的驗證證據。
- `archive/`：歷史與失敗紀錄，不供 runtime 載入。
- `assets/`：必要模型。
