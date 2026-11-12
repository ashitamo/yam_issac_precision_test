# Expert 等待與速度實驗

## 最新：並行 RGB-D 收集（2026-11-11）

入口為 `record_expert_data.py`，資料在 `data/`。下表包含相機 render／下載、
HDF5 寫檔與檔案審核，但不含 Isaac 啟動、shader 暖機或額外 action 回播。
物理與相機分別固定 240／30 Hz，使用相同 fast expert、PD、摩擦與成功門檻。

| 路徑（data 下） | 裝置／servo／solver | 收集秒數 | 成功 | 額外回播 |
| --- | --- | ---: | --- | --- |
| perf_reference_2x240_v2 | CUDA／240 Hz／64,16，舊逐列讀取 | 139.7 | 2/2 | 未要求 |
| perf_batch_2x240_v1 | CUDA／240 Hz／64,16，批次讀取 | 149.6 | 2/2 | 2/2 |
| perf_batch_2x60_v1 | CUDA／60 Hz／64,16 | 148.6 | 2/2 | 2/2 |
| perf_cpu_2x60_v1 | CPU／60 Hz／64,16 | 110.9 | 0/2 | 失敗，不採用 |
| perf_lowiter_2x60_v1 | CUDA／60 Hz／16,4 | 90.5 | 2/2 | 2/2 |
| parallel_expert_4x2_lowiter_v1 | CUDA／60 Hz／16,4，四孔位各兩回合 | 217.4 | 8/8 | 8/8 |

前五列用中心 r08_c08 與遠角 r16_c16。最後一列另加 r00_c00、r00_c16，
同一 simulator 中並行四列，第二批經 barrier reset 後重跑。
同兩孔位／60 Hz 對照，16/4 收集時間縮短約 39.1%（1.64 倍吞吐）；
不能把此改善歸因於 Python 批次讀取，也不能說單純降 servo 就有效。
這是本機少量單次對照，不是跨硬體統計或全網格證明。

8 回合純收集 217.44 s，含兩批額外回播共 332.20 s。平均每回合攤提
27.18 s（約 132.4 回合／小時），**不是單一 rollout 延遲 27 s**。
以同吞吐估 145 回合約 66 分鐘，未含啟動、失敗重試與尚未實作的新孔位排程。
最大 TCP 回播位置差 1.994 mm，低於原有 3 mm replay gate；這不是插入誤差。
未達單環境超即時，也未測 8／16 台 RGB-D 收集，不能直接線性外推。

16/4 是顯式 low_iter_candidate：scene minimum 與 peg iterations 一起調整，
不是降低 dt 或修改 actuator。仍逐物理步檢查原有 30 mm 深度與 0.050 mm
歷史幾何診斷門檻，沒有為了讓候選通過而放寬。
目前只允許 nominal 0.50 mm；其他 clearance 及跨-clearance 正式實驗必須
另驗證共同數值設定。預設 baseline 仍是 64/16、servo 240 Hz。
CPU 測試在 lift 約 3.3 mm 誤差停滯，不採用為加速方案，失敗檔／原因保留。

原生引擎計時：兩孔位候選的 simulate 約 1.19 s、fetch_results 約 30.91 s；
四孔位兩批約 2.44／63.67 s。fetch 含 GPU 完成等待，不只是資料傳輸。
這些欄位已包含在物理／env-step 計時中，render 也可能包含在 observer，
不能把所有分段相加。--profile 不啟用 cProfile 或額外 CUDA synchronize。

最終 53 個 CPU 測試通過，另完成四相機實際列排序／metadata／單步寫入
preflight（outputs/recorder_preflight_v1.log）。8 回合的 3205 幀 RGB-D
另逐筆審核：row metadata、T/T+1、相機／狀態時間、BASE action 組合、
servo 時間與全 depth validity mask 均通過。

## 以下為較早的無相機 expert 實驗

## 原版與候選版

原版仍是預設。`test_parallel_expert.py --expert_profile fast` 才啟用候選值。
可視化完全未修改；本次 benchmark 不啟用相機、GUI 或 cProfile。

| 設定 | baseline | fast |
| --- | ---: | ---: |
| 接近／下降時的總開口 | 60 mm | 20 mm |
| 關節命令 slew 上限 | 0.5 rad/s | 0.75 rad/s |
| 搬移速度／加速度 | 0.10 m/s、0.20 m/s² | 0.15 m/s、0.30 m/s² |
| 對準前下降速度／加速度 | 0.05 m/s、0.10 m/s² | 0.075 m/s、0.15 m/s² |
| 到點穩定時間 | 0.5 s | 0.15 s |
| 已建立夾持的穩定時間 | 0.15 s | 0.10 s |
| 抬升到點穩定時間 | 0.15 s | 0.10 s |
| 抬升後額外 hold | 0.15 s | 0（下一物理步仍檢查 peg） |
| 開爪後結束 | 張開後再等 3 s，最後檢查落底 | 張開且所有落底條件連續滿足 0.10 s |

不是把夾爪 actuator 加速：關爪前總開口改為 20 mm，對 9 mm peg 留有
11 mm 總間隙，減少兩指必須移動的距離；開爪釋放仍保持 60 mm。
這是已知 peg 尺寸與精確初始位置下的 expert 策略，不代表有定位誤差時仍適用。

fast 不更動 240 Hz、64/16 求解迭代、碰撞幾何、摩擦、peg 質量、10 Nm
總力矩預算、夾爪驅動限制、插入速度、30 mm 深度、幾何／接觸誤差門檻。
穩定確認時間確實縮短，因此需另外驗證，不能稱作完全等價的控制行為。
搬移中原有的掉落／滑移檢查持續有效，不以焊接或附著物體取代物理夾持。

## 開爪提前結束的條件

必須同時滿足：張開到目標、在孔內、深度 >=30 mm、tip 落底誤差 <=0.2 mm、
peg 線速度 <=5 mm/s，且連續穩定 0.10 秒。任一條件失效即重新累計。
不穩定時仍有原本階段 timeout，並記錄失敗，不會無限等待或直接判定成功。

## 分段計時

`outputs/speed_fast4_v2/` 為開啟 profiler 的四孔位候選驗證：4/4 task 與
contact gate 通過，3216 步、整批 95.55 秒。不得與未開 profiler 的時間直接
計算加速比。其中 simulate 呼叫 0.397 秒、fetch_results 41.936 秒，
scene.physics 合計 42.740 秒。

fetch_results 包含等待物理引擎完成，不能把它視為單純資料傳輸成本，也
不能據此再區分 GPU 碰撞、求解與 CPU 排程。以暫時代理原生介面計時，
離開量測後還原，未修改安裝的 Isaac Lab，也未加入額外 GPU synchronize。
本機未找到可用的 Nsight Systems／Compute CLI；尚無可靠的 GPU kernel 細分。

中心點的模擬秒數（四環境中的 row 0，展示各階段，而非嚴格牆鐘比較）：

| 階段 | baseline | fast |
| --- | ---: | ---: |
| approach | 3.533 | 2.267 |
| descend | 1.438 | 0.842 |
| close | 1.621 | 0.413 |
| lift | 1.100 | 0.821 |
| hold | 0.150 | 0.004 |
| transfer | 3.300 | 2.017 |
| align | 3.654 | 2.267 |
| insert | 1.950 | 1.950 |
| release | 4.354 | 1.458 |

## 公平吞吐測試

```powershell
python test_rollout_throughput.py --include_baseline `
  --output_dir validation/my_throughput
```

依序 baseline4、fast4、fast8、fast16；四個孔位固定為 r08_c08、r00_c00、
r00_c16、r16_c16。baseline4／fast4 是完全相同的案例清單。
8／16 環境等比例重複四個位置，屬相同案例分布，但 episode 總數不同。
這是小樣本功能與吞吐測試，不是完整網格成功率、定位泛化或訓練評估。

每例保留實際 PhysX 孔位查詢、完整 task/contact 結果與 trace。
任一回歸失敗就不啟動下一個更大 batch。benchmark.json 分別報告純模擬迴圈
與完整子程序耗時；不採用 profiler 量測來比較加速比。
可在輸出根目錄建立 STOP_REQUESTED 停止此工具所啟動的子程序。

本輪結果存於 `validation/throughput_v1/benchmark.json`。
候選版暫不設為全域預設；未宣稱 0.10 mm 已通過或單環境已超即時。

### 已完成結果（0.50 mm，無 GUI／相機／profiler）

| 設定 | task + contact 通過 | 整批迴圈耗時 | 平均每個成功 rollout |
| --- | ---: | ---: | ---: |
| baseline 4 env | 4/4 | 147.91 s | 36.98 s |
| fast 4 env | 4/4 | 86.66 s | 21.66 s |
| fast 8 env | 8/8 | 133.74 s | 16.72 s |
| fast 16 env | 16/16 | 222.23 s | 13.89 s |

總計 32/32 task/contact 通過，每一組的實際 workpiece 碰撞查詢也通過。
固定四環境、相同案例下，fast 本次耗時減少約 41.4%（吞吐約 1.71 倍）。
fast 16 環境相對 baseline 4 環境的總吞吐約 2.66 倍；其中同時混合了
expert 動作縮短與環境數增加兩種因素，不能全部歸因於平行化。

表中平均值是「產出速率的倒數」，不是單個任務的端到端等待時間；
16 個任務仍需等約 222 秒整批完成。完整程序（含啟動／初始化／關閉）
分別約 157.11、95.11、146.22、232.45 秒，另列於 benchmark.json。

8→16 環境吞吐只再增加約 20%，已出現效益遞減。下一步應改善命令
buffer 的逐 scalar GPU 寫入，以及逐環境狀態讀回／CPU IK，而不是直接
宣稱增加環境就能無限線性加速。尚未證明每個環境可超即時運行。

本輪是每個設定各一次 batch，不是多次重複測試的中位數或信賴區間。
23 個 CPU 回歸測試通過，包含開爪落底條件、profile 介面還原，以及 fast
參數白名單，確認沒有把物理設定或幾何成功容差混入加速覆寫。
