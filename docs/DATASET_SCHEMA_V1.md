# Dataset schema V1 — pilot

## 目前收集入口

使用 `record_expert_data.py`，舊 `test_dataset_pilot.py` 已由它取代。
支援 num_envs 個固定孔位並行、episodes_per_env 重複、可選 --replay。
各列各回合寫獨立 HDF5；manifest 記錄 grid_id／env_id／batch_id，split 仍為
unassigned，不會自行決定 Train/Validation。預覽在 episodes/episode_<id>/。
固定孔位 worker 整批結束才 reset；另由 --all_grid supervisor 以新場景分批換孔位，
不移動運行中的靜態碰撞體。campaign manifest 用 cases，worker manifest 用 episodes，
詳見 [HANDOVER](../HANDOVER.md)。

影像改用 lossless LZF，16 筆分塊寫入；不改解析度或 depth 數值。
新增 `servo/measured_timestamp_s`，明確記錄每筆 servo 關節量測時間。
60／120 Hz servo 的最後 tick 若在中途結束，使用實際 terminal joint state
與時間，不把終止後的狀態標成終止前資料。物理仍 240 Hz、policy／相機仍 30 Hz。
收集摘要改為 collection_summary.json；--replay 為額外驗證，非每次強制執行。
新的 metadata 另含 seed、robot_config、camera prim_path／sensor_output_index，
以明確相機路徑配對環境列，避免 env_10 等字典序造成影像錯配。
episode ID、回合／全域時鐘以及 servo 有限值均會檢查；不跨 reset 接成一回合。
使用 low_iter_candidate 時 solver 設定完整保存在 config，限制詳見 README。
下方舊 pilot 檔名／數據是歷史紀錄，現況操作以 README 為準。

## 本次實測

`data/cartesian_rgbd_pilot_v2/pilot_summary.json`：試收與保存 action 回播均成功。
399 個 action、400 幀 RGB-D／狀態，T+1 與相機時間對齊檢查通過；最後
transition 為 0.025 s。回播 399 步，最大 TCP 位置差 0.214 mm（門檻 3 mm）。
已目視檢查中途與終止 RGB，白 peg、黑工件與開爪可見。45 個單位測試通過。
這只是單次 pilot，不是資料集成功率或 learned policy 驗證。

v1 因 terminal 子步未滿相機 update period 而抓到重複 frame，已中止並保留
錯誤報告；後續整理時已刪除無效 partial HDF5（報告留在
`archive/output_cleanup_20261111/cartesian_rgbd_pilot_v1/`）。
v2 改由 collector 排程、相機即時更新後通過，沒有放寬檢查。

`env_dataset.py` 的 EpisodeRecorder 實作 `yam_cartesian_rgbd_pilot_v1`；
`record_expert_data.py` 收集每回合資料，--replay 可額外回播保存 action。
split 為 unassigned，未建立正式 train/validation/test 分割。

| HDF5 路徑 | 長度／定義 |
| --- | --- |
| `obs/joint_position`, `joint_velocity` | T+1，6 軸，rad／rad/s |
| `obs/gripper_opening` | T+1，總開口 m |
| `obs/ee_pose_base` | T+1，TCP xyz(m) + XYZW |
| `obs/images/rgb` | T+1，640×360 RGB uint8 |
| `obs/images/depth`, `depth_valid` | T+1，光軸 Z 深度 float32 m／uint8 mask，無效深度 0 |
| `actions/cartesian` | T，實際發出的 7 維 action，見控制契約 |
| `commands/ee_target_base` | T，當次組合的絕對 TCP 目標 |
| `time/observation_s`, `episode_s` | T+1，物理全域／回合時間，float64 s |
| `time/camera_s`, `camera_frame` | T+1，render 邊界物理時間／原生 counter |
| `time/action_start_s`, `action_end_s` | T，transition 邊界 |
| `terminated`, `truncated`, `rewards` | T，結束旗標／稀疏 reward |
| `servo/start_index`, `count` | T，以下扁平 servo 紀錄的索引 |
| `servo/joint_actions`, `command_timestamp_s`, `measured_joint_position` | 240 Hz 命令、發出時間與該 servo 步完成後的關節量測 |
| `privileged/*` | peg pose／速度、hole position，不作 policy 輸入 |
| `diagnostics/*` | 關節 target／torque 等診斷 |

policy observation 白名單：RGB、六軸 joint position、總開口。depth、EE、
velocity 為額外保存的原始資料，不自動納入模型輸入。

通常 T 個 action 對 T+1 個 observation，每筆 1/30 s、8 個 servo 命令。
servo 關節量測時間為同列 command_timestamp_s 加 servo_period_s，不是命令發出前量測。
最後提前終止可較短，保存實際 terminal 影像與狀態。影像／狀態時間必須一致，
觀測時間及相機 counter 必須遞增，否則中止試收。

相機為理想 pinhole D405，沒有硬體噪聲／stereo 失效模型。metadata 保存 K、
gripper→camera pose、TCP offset、設定 snapshot、版本與程式來源 SHA256。
時間為同步強制 render 的物理邊界，不是實體曝光時間。collector 負責 30 Hz
排程，相機 update_period=0，以支援任意 terminal 子步立即取得新 frame。

寫入 `.partial.hdf5`，完整終止後才標 complete 並改名 `.hdf5`；不覆寫現有檔。
manifest 的 validation_passed 還要求保存 action 回播成功且位置差 ≤3 mm。
僅 complete 不代表可訓練；失敗／中斷檔保留診斷，不自動納入 demonstration。

```powershell
python -u record_expert_data.py --headless --num_envs 2 --data_dir data\my_new_collection --replay
```

使用已設定的 Isaac Lab Python 與新輸出目錄。尚無 ACT／DP 轉換器、正式分割、
或 learned policy rollout；全網格資料排程／多環境影像收集已由 record_expert_data.py 提供。
