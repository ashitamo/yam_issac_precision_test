# 執行輸出

Dataset 已改放 `../data/`；有效 pilot 在 `../data/cartesian_rgbd_pilot_v2/`。
舊失敗 Cartesian 探測及 parallel_smoke 系列已移到
`../archive/output_cleanup_20261111/`，其失敗試收 partial HDF5 已刪除。
下方 smoke 名稱是歷史索引，現需至 archive 查看。

這裡不放程式或必要資產。失敗的中間探測已移到
`../archive/parallel_trials_20261111/`，有效輸出的原始日誌收在各目錄內。

| 目錄 | 用途 |
| --- | --- |
| latest | run_environment.py 預設輸出，可被下一次執行更新 |
| fast_visual_demo、parallel_demo | 使用者執行的可視化 demo，保留未動 |
| parallel_expert_four_v3、parallel_expert_single_v3 | 多環境入口的完整任務回歸 |
| parallel_smoke、parallel_smoke_1、parallel_smoke_4 | reset／相同場景短測試，不是完整插入驗證 |
| profile_single_v1 | 單環境完整 rollout 的分段計時 |
| speed_fast4_v2 | 候選加速 expert 與引擎細分計時 |
| release_diagnostics | 隔離開爪回歸 |

正式 4／8／16 環境吞吐比較在 `../validation/throughput_v1/benchmark.json`。
歷史報告與原始失敗紀錄仍保留，避免清理後只剩成功案例。
