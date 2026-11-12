# Dataset

所有試收／正式訓練原始資料放這裡，不再放 outputs。

`parallel_expert_4x2_lowiter_v1/`：最新四環境、四孔位各兩回合的 RGB-D 驗證，
8/8 收集與回播通過，3205 幀觀測的事後審核也通過。原始檔在 episodes/，
summary／manifest 記錄成功與時間；目前仍未分 Train/Validation。

`consolidated_rgbd_pilot_v1/` 是程式整併後新跑的回歸 pilot，試收與 action
回播均通過（399 action／400 幀）；舊 pilot 保留，不互相覆寫。

`cartesian_rgbd_pilot_v2/` 是已通過的單回合 pilot：399 個 action、400 幀 RGB-D
與狀態。`episodes/episode_000000.hdf5` 為資料，PNG 是抽樣預覽，
`pilot_summary.json` 為對齊／回播檢查；不是正式 train split。

新收集使用 `python record_expert_data.py --headless --num_envs 4 --data_dir data\new_collection`，
指定新目錄避免覆蓋。可加 --replay，或用 --episodes_per_env 重複固定孔位。
效能對照資料在 perf_reference_*／perf_batch_*，是否通過看各自 collection_summary.json。
