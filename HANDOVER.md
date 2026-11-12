# YAM Pick-and-Insert 交接

更新：2026-11-12。目標機：Ubuntu 22.04、RTX 4090 ×2。
**這是文件交接，不代表 Ubuntu 安裝／雙卡驗證已完成。收集目前按要求停止。**
日常指令保留在 [README](README.md)；本文件說明版本、設計、資料與風險。


## 0. Isaac Lab 2.1 相容分支（本分支優先資訊）

- Branch：`compat/isaaclab-2.1`；main 原版起點 `d23468318ef8c2f5cd17e460b666eb45be0c1a96` 未修改。
- 目標：Ubuntu 22.04.5、550.144.03、RTX 4090；**不保證這個 driver 組合已驗證**，不更動驅動。
- 固定版本：Python 3.10、Isaac Sim 4.5.0、Isaac Lab **v2.1.0**，官方 tag commit
  `21f7136325136ca3f6ca4e0a8125edffe5c24f7e`；Torch 2.5.1/cu121。
- 安裝指令見 README。不要在現有 Python 3.12 環境覆蓋安裝，不用容器繞過 host driver 限制。
- 本機已另建 Windows `env_yam21`；62 個 CPU 測試通過，另完成下述原生測試。Ubuntu 仍未實測。
- 後文原有 104 回合資料、效能數字、0.10 mm 接觸驗證均是 **main 歷史證據**，未移植認證。
- 不自動恢復收集、不修改既有 data/checkpoint。相容版必須用全新 data_dir。

### 實作差異與 review 重點

1. `env_yam.create_simulation` 集中建構舊版 `SimulationCfg(physx=PhysxCfg(...))`，
   並拒絕錯誤 Sim/Lab runtime；不再依賴 isaaclab_physx。
2. Articulation / RigidObject 使用 v2.1 原生 Torch tensor 與不帶 `_index` 的寫入 API。
3. 對外 observation、SciPy、action target、HDF5 繼續 XYZW；只有原生物理、
   spawn（含指尖接觸面）、CameraCfg 與 DifferentialIK 邊界轉 WXYZ。
   身體 pose 來自原生 FK，不新增另一套 runtime URDF FK。
4. Jacobian 使用 `root_physx_view.get_jacobians()`，檢查 fixed/floating 欄位，
   從 COM 平移到 TCP，再轉 BASE；以原生 FK 有限差分測試作為上機門檻。
5. `prepare_legacy_rigid_hierarchy` 在所有環境複製完成後，將巢狀 rigid bodies 的世界位姿
   bake 成 reset Xform stack，保留 joint/body 路徑、幾何局部座標與來源 USD。
   世界固定關節的空 body0 在 4.5 會套用 parent Xform，必須避免 env origin 套兩次；
   reset 測試已檢查 base 與 env origin 的誤差。相容版目前關閉 PhysX replication，
   保留同程序多環境但使用獨立 USD clones，尚未認證大量環境效能。
   同時把來源 `newton:selfCollisionEnabled=False` 明確映射到 PhysX articulation 屬性，
   否則 4.5 預設的 self-collision 會阻擋手指並破壞追蹤。
6. 重力補償使用 `get_gravity_compensation_forces()` 並檢查 0／6 個 root 欄位。
   保留 10 Nm 總力矩預算、夾爪 actuator、質量與摩擦設定；未宣稱動態等效。
6. 這次實驗未用到的 Sim 6 cuMotion／experimental RMPflow 類別從本分支移除，
   不提供同名假相容函式；CartesianYamEnv 繼續使用 Lab 原生 DLS IK。
7. 舊版無法沿用新版兩個求解器開關。分支明確設定
   `TGS_EXTERNAL_FORCES_EVERY_ITERATION=False`、`SOLVE_ARTICULATION_CONTACT_LAST=False`。
   若將不支援的選項設 True，建構時報錯，不靜默略過。所有公差需重新驗證。
8. profiler 改量 `engine.step_inclusive`，不再聲稱可分開量 simulate/fetch_results。
   相機保持每環境 RGB-D、30 Hz 記錄契約；需在 4.5 實測畫面與 row mapping。
9. GUI 不加 `--headless` 即可；2.1 不使用 `--viz kit`。

### 本機實測結果（2026-11-12，Windows 11／RTX 3060／driver 610.88）

原 Sim 6 conda 未修改；獨立安裝位於 `C:/Users/hsun91/.conda/envs/env_yam21`。
Lab source 為 `C:/Users/hsun91/Desktop/IsaacLab-2.1` 的 v2.1.0 tag。
安裝時修正 release tag 與 distribution version 不同的檢查，Pillow 改為 Lab 要求的 11.0.0。

| 測試 | 結果／證據 |
| --- | --- |
| Python 3.10、pip check、CPU 契約 | 62 tests 通過、相依無衝突 |
| 兩環境 reset／物理步進 | `outputs/lab21_windows_reset_004/result.json` 通過；base origin 最大誤差約 1.7 µm，其他 row 未被部分 reset 改動 |
| 原生 TCP Jacobian＋DLS 控制 | `outputs/lab21_windows_cartesian_002/cartesian_result.json` 通過；FK 有限差分差異位置約 0.000060 m/rad，角速度約 0.000211；動態 TCP 位置誤差約 0.0027 mm |
| Warp cu11 重驗 | 改裝 1.8.1+cu11 後，`outputs/lab21_windows_cartesian_cu11_001/cartesian_result.json` 的 Jacobian 與 DLS 測試仍通過，TCP 誤差約 0.0027 mm |
| 環境入口／相機 | Warp cu11 下 headless 120 步完成；另修正 OSQP 載入順序後，GUI 也完成 120 步，`outputs/lab21_gui_fixed_001/rgb.png`，RGB 360×640×3、depth 360×640 |
| 無碰撞診斷 | `outputs/lab21_windows_no_contacts_001` 僅供隔離原因，不可當正式任務成功 |
| 原始速度門檻測試 | `outputs/lab21_windows_full_task_001`／`data/lab21_windows_smoke_002`，close timeout；原生 finger 速度中位數約 1.60 mm/s，但位置差分僅約 0.015 mm/s |
| 修改穩定判斷後的收錄 | `data/lab21_windows_smoke_003`，完成夾取、搬運、插入與開爪落底；HDF5 結構與影像／時間戳稽核通過 |
| 上述回合正式驗收 | **失敗**：`terminal_reason=contact_violation`，最大徑向幾何超界約 0.231 mm > 0.050 mm；沒有調寬門檻或重標成功 |
| Action 回播 | 同一資料夾 summary：成功落底、`replay_passed=true`，TCP 與原紀錄最大差異約 0.251 mm；不能因此改寫原始回合的失敗標籤 |
| 批次／GPU／Linux | 尚未驗證 289 點、雙卡、Ubuntu 550 或多相機原生 row mapping |

`smoke_003/episodes/episode_000000.hdf5` 為 **389 actions／390 observations**，
RGB 為 360×640、含 depth 與 diagnostics；這是診斷用失敗回合，不是已驗收訓練資料。
同層 `episode_000000/rgb_initial.png`、`rgb_middle.png`、`rgb_terminal.png` 可預覽。
`smoke_001` 在 Vulkan 啟動崩潰，沒有 episode；`smoke_002` 保留完成的失敗回合，
已用 STOP_REQUESTED 中止無意義的失敗回播。不要把這些資料混入 main 的 campaign。
這些產物按 .gitignore 留在本機，沒有隨 branch 上傳。

關爪穩定判斷現在使用 **50 ms 位置 envelope 的位移／時間**，再要求穩定滿原定時間。
仍用 1 mm/s 門檻，但這是 settling rate，不等於單一 tick 的瞬時 solver 速度；
保留 aperture、載入力、姿態、lift／slip 及 0.05 mm 接觸門檻。raw velocity 仍完整記錄。
這是 expert 相容性行為變更，不應宣稱物理模型與 main 完全等效。

Windows 執行有相機的指令需加 `--kit_args="--/app/vulkan=false"`；
D3D12 可取得正常影像，原 Vulkan 路徑在 renderer 初始化 native crash。
首次 shader 編譯曾約兩分鐘；不是卡在 expert 等待。
GUI 另曾在 `wheeled_robots -> osqp -> qdldl` 原生匯入時 access violation。
OSQP 單獨匯入正常；在 Kit 前預載後，診斷與正常入口兩次 GUI 啟動均通過。
`run_environment.py` 僅在 Windows 預載 OSQP，沒有改動第三方二進位或停用擴充。
同時在 env_yam21 安裝 requests 2.34.2，解決 asset.browser 缺少依賴。
GUI 的 HDF5 版本警告仍待隔離，不應把此次 viewer 驗證當成 GUI 資料錄製驗證。
還有 **Kit shutdown 不退出** 的未解問題：報告／HDF5 關閉完成後，測試程序曾停在 native 關閉流程，
本次確認各程序屬於測試後才手動停止。已嘗試 stop physics／clear callbacks，但不能宣稱已解決。
因此尚不可承諾 supervisor 可無人值守跑完整 campaign。

### Warp 與 Ubuntu 550 驅動注意事項

舊 Lab 的 warp-lang 沒有上限；2026 的預設安裝拉到 1.19.0／CUDA 13.4，
這不適合直接帶到 550。完整 RGB-D smoke_003 是在該 Windows 套件版本下執行；
之後本機已改為官方 **Warp 1.8.1+cu11**，確認初始化與 CUDA array 配置成功、pip check 通過。
Ubuntu x86_64 對應 wheel 已固定在 README；不能直接複製 Windows wheel。
不要修改 Isaac Sim 隨附的 extension 二進位檔，也不要關閉 driver compatibility check。

官方依據：[Warp 1.19 README 的 driver 要求](https://github.com/NVIDIA/warp/blob/v1.19.0/README.md)、
[Warp 1.8.1 release／CUDA 11 wheel](https://github.com/NVIDIA/warp/releases/tag/v1.8.1)。
Ubuntu 550 上的實際 Warp／Kit／PhysX 啟動仍是待驗 gate。

### Ubuntu 上必須依序通過的 gate

從專案根目錄、啟用 env_yam21 執行；報告與資料一律新目錄。

```bash
python -m pip check
python -B -m unittest discover -s unit_test -p 'unittest_*.py' -q
python -B test_assets.py
python -u test_parallel_environments.py --headless --device cuda:0 --num_envs 2 --steps 24 --output_dir outputs/lab21_reset_001
python -u run_environment.py --device cuda:0 --max_steps 120 --camera_snapshot outputs/lab21_camera_001.png
python -u test_cartesian_adapter.py --headless --device cuda:0 --expert --output_dir outputs/lab21_cartesian_001
python -u record_expert_data.py --headless --device cuda:0 --num_envs 1 --grid_ids r08_c08 --clearance_mm 0.5 --data_dir data/lab21_smoke_001 --replay
```

確認 reports 成功、Jacobian probe 通過、相機非黑圖且時間戳遞增、實際落底／開爪及回播成立；
再測四環境與 0.10 mm 少量點（`test_precision_subset.py`），最後才跑正式全網格。
兩張 GPU 仍非自動分片；GPU 1 要獨立程序、獨立 data_dir。
若 import、資產 reference、shader、driver 或 native probe 失敗，停在該 gate 留下 log；
不能把 CPU tests 通過當成完整移植成功。

參考：[官方 v2.1.0 安裝](https://isaac-sim.github.io/IsaacLab/v2.1.0/source/setup/installation/pip_installation.html)；
[官方 v2.1.0 原始碼](https://github.com/isaac-sim/IsaacLab/tree/v2.1.0)。

## 1. 專案現況與範圍（main 歷史基線）

YAM 從桌上實際夾取白色 peg，lift、transfer、align、insert、release，
以原生物理狀態判斷落底，不把 peg 固定在末端。
工件為 assets/workpiece.stl，黑色靜態通孔碰撞；腕部理想 D405 產生 RGB-D。

已完成 scripted expert、共享環境、Cartesian adapter、並行 RGB-D／HDF5、
四種公差全網格排程、來源檢查與進度保存。55 個 CPU 單元測試曾通過。
未完成正式 Train/Validation/Evaluation manifests、ACT／DP loader／training、
雙卡全網格分片、跨系統續收移植與 Ubuntu 原生驗證。不要把工程測試標為 B 全完成。

## 2. 原機版本：先固定，不要同時升級

下列是 Windows 原機實際查得的 Python distribution 版本，不是 Linux lockfile，
也不保證同名同版 Linux wheel 都可直接下載：

| 元件 | 原機版本 |
| --- | --- |
| Python | 3.12 系列 |
| isaacsim | 6.0.1.0 |
| isaaclab | 6.1.14，editable source 安裝 |
| isaaclab_physx | 1.1.3 |
| torch | 2.10.0+cu128 |
| warp-lang | 1.13.0 |
| numpy／scipy | 2.3.1／1.17.0 |
| h5py／trimesh | 3.16.0／4.11.1 |

原 Isaac Lab checkout commit：

```text
c2e46a0139fc81b016d2eb8bf1d8d6b4c5a1587d
```

原機安裝位置是舊工作區 IsaacLab_isaac6/source/isaaclab，**不是本專案內
附帶了一套 Isaac Lab**。本地 git status 的已見修改在舊 scripts/yam 測試／腳本，
不要把整個 dirty checkout 當成乾淨 release。接手者應核對 core library diff。
以 commit 與實際 API 為準，不要猜目前 package 版本等於官網同名 release tag。

此節僅記錄 main 的原機版本；本分支 Ubuntu 安裝請依第 0 節與 README，不安裝 Sim 6。

## 3. 要搬哪些資料

必要：根目錄程式、assets/、unit_test/、README、HANDOVER、docs/。
驗證／資料追溯：data/、validation/、archive/ 與相關 outputs/，可另存資料碟，
但不要只搬 HDF5 而丟掉 campaign／manifest／metadata／log。
不必搬 __pycache__、Windows conda 環境或舊工作區其他 YAM 示範程式。
本專案資產路徑依 Path(__file__) 定位；Linux 檔名大小寫須一致，
USD 的外部 reference 是否完整，仍需新機資產測試確認。

相容分支已修改程式 API；不修改舊 checkpoint 或搬動已收資料。

## 4. 建議接手驗證順序

所有命令從專案根目錄，以已配置好的 Isaac Lab Python 執行。輸出用新目錄。

```bash
python -B -m unittest discover -s unit_test -p 'unittest_*.py' -q
python -u run_environment.py --headless --device cuda:0 --max_steps 24 \
  --output_dir outputs/ubuntu_smoke_001 --camera_snapshot outputs/ubuntu_smoke_001/rgb.png
python -u test_cartesian_adapter.py --headless --device cuda:0 --expert \
  --output_dir outputs/ubuntu_cartesian_001
python -u record_expert_data.py --headless --device cuda:0 --num_envs 4 \
  --grid_ids r08_c08 r00_c00 r00_c16 r16_c16 --clearance_mm 0.5 \
  --servo_hz 240 --solver_profile baseline --compression gzip \
  --data_dir data/ubuntu_baseline_001 --replay
```

然後用新的 data_dir 分別驗證 1.00／0.25／0.10 mm，不直接在新機重跑 1,156 回合。
看 stdout 是否正常退出、collection_summary.passed、每回合 manifest、PNG、
HDF5 時序與 replay gate。GUI 需桌面環境；SSH headless 不等於遠端 GUI 可直接打開。
先比較 4／8 環境再擴大，不假設 4090 比 3060 的速度等比例提高。

## 5. 雙 RTX 4090 的使用界限

目前固定孔位 collector 可選 --device cuda:0／cuda:1；AppLauncher 目前 source
會把 physics_gpu 與 active_gpu 指向該 device。仍應用 nvidia-smi 查證新機實際
physics／RTX 顯存配置，不假設 CUDA_VISIBLE_DEVICES 重新編號後的渲染選卡一定相同。
兩張卡不自動合併顯存；目前沒有 DDP 訓練或單一 collector 的多 GPU 分片。

兩終端、各一張卡的**待驗證小批配置**如下，位置與輸出目錄不同：

```bash
# 終端 A
python -u record_expert_data.py --headless --device cuda:0 --num_envs 2 \
  --grid_ids r08_c08 r00_c00 --compression gzip --data_dir data/gpu0_smoke_001 --replay
# 終端 B（先單獨測 GPU 1，確認後再同時跑）
python -u record_expert_data.py --headless --device cuda:1 --num_envs 2 \
  --grid_ids r00_c16 r16_c16 --compression gzip --data_dir data/gpu1_smoke_001 --replay
```

本機尚未驗證這兩程序在 Ubuntu 同時運作；需檢查 RTX 是否搶卡、RAM、CPU 與
NVMe 吞吐。不要宣稱此配置已完成雙卡全網格收集。
--all_grid supervisor 會硬編碼子程序 --device cuda:0，且不接受 --device；
不能直接把上面的 GPU 1 用法套到它。兩個 --all_grid 即使不同資料夾也會重複
整份網格，不是分片；同資料夾則有 lock／資料競爭風險，不要這樣啟動。
未實作自動跨機／跨 GPU 排程，之後需以不重疊 case ID 的 manifests 分片。

## 6. 程式閱讀順序

| 檔案 | 責任 |
| --- | --- |
| env_config.py | 物理、相機、網格、expert 與候選 solver 設定 |
| robot_yam.py | 本地 YAM 資產、碰撞、actuator／gravity FF、原生 TCP Jacobian |
| env_yam.py | ParallelYamScene → joint env → Cartesian adapter |
| env_expert.py | 單列 phase controller 與多列 expert；只讀共享 observation 的新路徑 |
| env_task.py | 成功／落底與共用幾何診斷 |
| env_dataset.py | 批次相機列對應、時間同步、分塊 HDF5 與 episode 驗證 |
| env_workspace_grid.py | 17×17 grid 與既有離線 FK／IK 篩查 |
| record_expert_data.py | 固定孔位 worker、audit／可選 replay，以及 --all_grid supervisor |
| env_profiling.py | 可選整體 step 計時，不改已安裝 Isaac Lab，不提供分開的 simulate／fetch 計時 |

run_environment 使用同一套場景，但是 joint-space viewer；不錄製 HDF5。
collector 使用 Cartesian adapter；不要把兩種 action 混用。
原生 runtime IK 是 Isaac Lab DifferentialIKController，expert／離線網格還有
舊 SciPy FK／IK，並非整套 expert 全 GPU 向量化。FrameView／Jacobian layouts
有目前版本特有之處，升級前需重跑 native probe。
test_*.py 為手動模擬驗證；unit_test/unittest_*.py 為 CPU 測試，無需逐支啟動。
舊 test_environment_A.py 含 Windows 特定程序檢查，不是 Ubuntu 主收集入口。

## 7. 控制、座標與物理契約

- SI：m、rad、s；pose quaternion XYZW，action 不是 Euler。
- Cartesian action：[dp_BASE(3), drotvec_BASE(3), 總開口絕對值(1)]。
  p*=p_measured+dp，R*=Exp(dr)@R_measured，目標每 policy tick 只組合一次。
- policy／相機 30 Hz；物理 240 Hz；servo 240／120／60 Hz，預設 240。
  drive 每物理步作用，重力 FF／joint target 按 servo 更新並保持。
- action norm 上限：平移 20 mm，rotvec 0.2 rad。policy 越界拒絕；
  expert 另有顯式 reference governor，記錄的是實際發出的限制後 action。
- TCP 在 gripper local [0,0,0.14472] m，姿態沿用 gripper，不是單指尖。
- arm 10 Nm／3 rad/s、Kp 400／Kd 40。PD 額度預留 gravity FF，不能簡單
  把額外重力 effort 加到已滿的 10 Nm drive 上。
- finger 每指 1.5 N／0.02 m/s、Kp 1000／Kd 17；開口是兩指總內側距離，
  joint7/8 為負值張開，不能把總開口直接當單指 command。
- baseline solver 64/16；low_iter_candidate 為 16/4，只允許 0.50 mm。
  正式跨-clearance 比較需固定共同驗證的數值設定，不混用 profile 當唯一變因。
- CPU backend 曾 lift 停滯，不推薦為加速方案。剛體／摩擦與感測模型不是實機標定。

環境目前以平移分隔、BASE 軸向一致，不是任意旋轉 base 都已驗證。
不自動 reset；先存 terminal snapshot，再明確 reset。終止於 servo tick 中途
時，影像與 measurement timestamp 必須是實際終止子步，不可標成之後的狀態。
成功包含 lift 歷史、開爪、孔內、深度至少 30 mm、接近桌面落底、速度與穩定時間；
歷史最大徑向／底部幾何診斷門檻為 0.050 mm，不等於 PhysX contact_offset
（20 µm）。不要為了 throughput 放寬成功標準。

## 8. Dataset 與全網格排程

公差指單側 radial clearance：(hole diameter - peg diameter)/2。
nominal hole 10 mm，四種 peg diameter 為 8／9／9.5／9.8 mm，長 50 mm、25 g。
STL 孔為多邊形，內切半徑約 4.992667 mm，effective clearance 略小於 nominal。
17×17、200×200 mm、間距 12.5 mm；中心 r08_c08，候選棋盤格標籤不是正式 split。

HDF5 有 T actions／T+1 observations；RGB uint8、axial depth float32 m、
depth_valid mask。policy 白名單是 RGB、q6、總開口；privileged 真值僅給 expert，
不是因為被存檔就可當 policy input。相機為理想 pinhole D405，沒有 stereo／噪聲模型。
raw schema 名稱仍含 pilot_v1，並非已提供 ACT／DP loader。

固定孔位 worker 使用 barrier reset；--all_grid 才分批新建場景換孔位，避免在
運行中移動靜態 collider。四種公差交錯、各 289 案，共 1,156 案。
全網格固定 baseline、240 Hz servo、gzip、每點一回合、不額外 replay。
完整 task failure 照樣歸檔，不反覆重跑到成功；只有未完成的基礎設施錯誤會重試。
連續三次沒有新完成回合會停；來源／資產改動、低空間或 stop marker 也會停。
磁碟保留 20 GB，啟動下一批另要求最壞情況 RGB-D 空間；不會刪舊資料騰空間。

worker 的 manifest 用 episodes；campaign 各 dataset manifest 用 cases，
內含 pending／success／task_failure。不要用同一段 loader 假設兩者格式相同。
保留所有失敗、partial、log，但不要當成成功 demonstration。
--replay 通過門檻為任務成功與最大 TCP 差 ≤3 mm，並非 bitwise deterministic，
也不能把全軌跡最大 TCP 差直接當最終插入誤差。

## 9. 已有資料、停止狀態與續收

2026-11-12 最後核對：

| 公差 mm | campaign 登記完成 | 已登記失敗 |
| --- | ---: | ---: |
| 1.00 | 28 | 0 |
| 0.50 | 28 | 0 |
| 0.25 | 24 | 0 |
| 0.10 | 24 | 0 |

路徑：data/full_grid_4clearance_20261112_01/。合計 104／1,156。
重開機中斷於 c0p25_batch_006，位置 r01_c07～r01_c10；該批另有 2 份完整
HDF5 與 2 份 partial，尚未核對入 canonical manifest，不把它們加進已完成數。
使用者已要求先不收，目前沒有重新啟動。status 的 running／PID／ETA 為舊值。

不要在新機直接 --resume 舊目錄：

1. campaign command 含 Windows 絕對路徑，資產指紋含 mtime，且有舊 RUNNING.lock。
2. 現有程式未實作跨平台 checkpoint 轉換，不可刪掉來源檢查來硬續。
3. 先備份並審核 104 案及中斷批次的 complete／成功／時間／metadata。
4. 對 complete 但尚未登記的案做對帳；partial 只保留，不補名為成功。
5. 由 canonical case IDs 建立 missing-case 清單，在 Ubuntu 新 campaign 收缺項，
   另記平台／driver／runtime provenance；目前「只跑此缺項清單」入口尚需實作。
6. 同機同路徑 resume 也要先確認舊 supervisor／worker 全退出，核對／備份 stale lock。

不要只根據檔案數、過期 running 標記或 manifest 的布林值宣稱整份資料已驗收。
目前所有 split=unassigned，還不能宣稱完成訓練集或把 candidate-seen 當已訓練位置。

## 10. 長時間執行與優先待辦

Ubuntu 長任務建議放在新開的 tmux session，另存 log；SSH 斷線不應當作停止方式。
不要讓同一 data_dir 有兩個 producer；不得在收集中修改程式／assets。
用 campaign 根目錄的 STOP_REQUESTED 停止，查程序與 log 確認退出；
重開機不會自動續收，本專案沒有 systemd／開機服務。
遷移驗證完成前，不要順手啟動全網格或改寫 Windows 原資料。

優先順序：Linux 相同版本 smoke → 四公差小批 → 第二張卡驗證 → 不重疊雙卡
分片與 missing-case 遷移 → 固定 Train/Validation/Evaluation manifests → 正式補收
與資料驗收 → loader／ACT／DP pilot。
現有 3060 測試時間不能直接套用到 4090×2，應重新量測含相機與寫檔的成功吞吐。
詳細舊測試見 docs/EXPERT_SPEED_VALIDATION.md、PRECISION_SUBSET_RESULT.md、
REPEATED_ROLLOUT_VALIDATION.md；版本舊的結果不自動代表新機驗收通過。
