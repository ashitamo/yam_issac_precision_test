# YAM Pick-and-Insert

> 此分支：`compat/isaaclab-2.1`，目標為 **Isaac Lab 2.1.0 + Isaac Sim 4.5.0 + Python 3.10**。
> 原 Isaac Sim 6 版本保留在 `main`（分支起點 `d234683`）。
> 已在 Windows／RTX 3060 原生啟動，驗證 reset、Jacobian、Cartesian IK、RGB-D 收錄與 action 回播。
> **完整成功驗收尚未通過**：中心點原始回合接觸幾何超界約 0.23 mm，大於 0.05 mm；Ubuntu 550 驅動仍未驗證。
> 下方既有收集統計與效能數字屬於 main，不代表本分支已完成原生驗證。

YAM 夾取白色 peg，搬運至黑色工件並插入、開爪，最後確認 peg 落底。
目前可開啟環境、執行 scripted expert，以及**多環境 RGB-D 收集與可選 action 回播**。
已提供四種公差的 17×17 全網格批次收集；正式資料分割與 learned policy 訓練尚未完成。
接手專案、Ubuntu 遷移、版本與中斷資料處理請先讀 [HANDOVER.md](HANDOVER.md)。

## 1. 準備執行環境

### Ubuntu 22.04／RTX 4090 ×2（新機）

請建立獨立環境，**不要在既有 Isaac Sim 6 conda 環境內降級**。
本分支固定 v2.1.0，不要改裝最新版或直接跟著 v2.1.1 的 CUDA 12.8 安裝範例。
參考 [官方 v2.1.0 安裝文件](https://isaac-sim.github.io/IsaacLab/v2.1.0/source/setup/installation/pip_installation.html)。

```bash
conda create -n env_yam21 python=3.10 -y
conda activate env_yam21
python -m pip install --upgrade pip
python -m pip install 'setuptools<81' toml
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com
git clone --branch v2.1.0 --depth 1 https://github.com/isaac-sim/IsaacLab.git ~/IsaacLab-2.1
cd ~/IsaacLab-2.1
./isaaclab.sh --install none
cd /path/to/yam_issac_precision_test
git switch compat/isaaclab-2.1
python -m pip install 'numpy==1.26.4' 'scipy==1.15.3' 'h5py==3.12.1' 'trimesh==4.6.8' 'Pillow==11.0.0'
python -m pip install 'https://github.com/NVIDIA/warp/releases/download/v1.8.1/warp_lang-1.8.1%2Bcu11-py3-none-manylinux2014_x86_64.whl'
python -m pip check
```

v2.1.0 的 `isaaclab` distribution 版本其實是 `0.36.21`，不是 `2.1.0`。
Warp 固定官方 1.8.1/cu11 wheel，避免未鎖版本自動安裝 CUDA 13 wheel 而要求更新 driver。
這是目標安裝配方，不是已在 Ubuntu 實測的完整 lockfile；若 pip check 衝突，先保留錯誤，不要強制忽略。
驅動維持 550 不在此修改；仍須先確認 Isaac Sim 4.5 能啟動，不能以跳過 driver check 代替驗證。
複製專案程式、assets、unit_test 與文件；不要複製 Windows conda 環境。

```bash
conda activate env_yam21                    # 使用新機實際建立的環境名稱
cd /path/to/yam_issac_precision_test      # 換成實際專案位置
nvidia-smi
python -B -m unittest discover -s unit_test -p 'unittest_*.py' -q
```

開啟場景（需要可用的本機桌面／顯示環境）：

```bash
python -u run_environment.py  --device cuda:0
python -u run_environment.py  --device cuda:0 --expert --grid_id r08_c08
```

先依 HANDOVER 的順序完成資產、reset、Jacobian 與單次任務驗證。
**尚未通過之前不要啟動全網格收集。** 小批收集使用獨立新目錄，不能續寫 main 的 campaign。

先用 GPU 0 做四孔位、baseline RGB-D 收集與回播驗證：

```bash
python -u record_expert_data.py --headless --device cuda:0 \
  --num_envs 4 --grid_ids r08_c08 r00_c00 r00_c16 r16_c16 \
  --servo_hz 240 --solver_profile baseline --compression gzip \
  --data_dir data/ubuntu_baseline_001 --replay
```

更換公差可加 `--clearance_mm 1.0`／`0.5`／`0.25`／`0.1`，各用新資料夾。
兩張 4090 不代表目前程式自動用雙卡；固定孔位入口可用 `--device cuda:1`
測第二張卡。全網格 supervisor 目前固定啟動 GPU 0，尚無雙 GPU 分片排程。
不要讓兩個程序寫同一個 data_dir。完整雙卡試跑與背景執行說明見 HANDOVER。

### Windows（本機測試環境）

已另外建立 `env_yam21`，沒有修改原本的 `env_isaaclab6_win`。
目前使用 Python 3.10.20、Sim 4.5.0.0、Lab v2.1.0（distribution 0.36.21）、
Torch 2.5.1/cu121。環境不是 repo 內容；另一台電腦仍須重新安裝。

```powershell
conda activate env_yam21
Set-Location C:\Users\hsun91\Desktop\yam_issac_precision_test
python -B -m unittest discover -s unit_test -p 'unittest_*.py' -q
python -u .\run_environment.py --kit_args="--/app/vulkan=false"
```

**本機 Windows 的有相機／GUI 測試需要 D3D12**：在下方 Windows 指令後加
`--kit_args="--/app/vulkan=false"`。不將這個 Windows 專用參數帶到 Ubuntu。
原 Vulkan 渲染啟動曾發生 native crash；不是以關閉 driver check 解決。

Windows 本次測試還有「報告寫完後 Kit shutdown 不退出」的問題。
已完成報告不等於程序正常結束；這個問題尚待解決，暫不宣稱可無人值守全網格收集。
詳細測試結果與檔案位置見 HANDOVER。

## 2. 啟動環境

### 只看場景與腕部相機，不執行任務

```powershell
python -u .\run_environment.py
```

視窗會保持開啟，關閉視窗結束。相機是理想 D405 RGB／depth，不含實體噪聲模型。
GUI 的 Camera 選單可切換到場景內的 D405 Camera。

### 看 expert 完成夾取與插入

```powershell
python -u .\run_environment.py  --expert --grid_id r08_c08
```

`r08_c08` 是 17×17 網格的中心點。任務完成或失敗後程式會退出。
此入口是 joint-space expert；它**不會錄製訓練用 HDF5**。
診斷報告預設寫入 `outputs/latest/`，同名檔可能被更新。
保留獨立結果時，另加 `--output_dir outputs\my_expert_run`。

選用參數：

| 參數 | 用途 |
| --- | --- |
| `--headless` | 無視窗執行，GUI 不加此參數 |
| `--clearance_mm 0.5` | 單側徑向間隙；可選 1.0／0.5／0.25／0.1，預設 0.5 |
| `--max_steps 240` | 限制物理步數；預設 0，無 expert 時持續顯示 |
| `--camera_snapshot outputs\camera.png` | 暖機後保存一張腕部 RGB 預覽 |

## 3. 並行收集資料

正式收集入口為 `record_expert_data.py`，不再使用 `test_*` 名稱。
同一個 Isaac Sim 中同時跑多個環境、每列獨立相機與 HDF5 episode。

先用四個指定孔位，收集後額外回播驗證：

```powershell
python -u .\record_expert_data.py --headless --num_envs 4 --grid_ids r08_c08 r00_c00 r00_c16 r16_c16 --servo_hz 60 --solver_profile low_iter_candidate --data_dir .\data\expert_4env_001 --replay
```

不需要每回合回播時，移除 `--replay` 即可省下驗證回播時間。
看收集畫面時，移除 `--headless`。

每個孔位重複收兩回合（共 8 回合）：

```powershell
python -u .\record_expert_data.py --headless --num_envs 4 --grid_ids r08_c08 r00_c00 r00_c16 r16_c16 --episodes_per_env 2 --servo_hz 60 --solver_profile low_iter_candidate --data_dir .\data\expert_8episodes_001
```

### 參數與範圍

| 參數 | 定義 |
| --- | --- |
| `--num_envs` | 並行環境數，預設 4；仍受 GPU 記憶體限制 |
| `--grid_ids` | 一個 ID 供所有列使用，或剛好 num_envs 個 ID |
| `--episodes_per_env` | 各列固定孔位重複次數；預設 1 |
| `--clearance_mm` | 單側徑向間隙，預設 0.5 |
| `--servo_hz` | 60／120／240 Hz，預設仍為 240；低頻是比較選項，不代表所有位置均已驗證 |
| `--solver_profile` | 預設 baseline（64/16）；low_iter_candidate（16/4）只允許 0.50 mm |
| `--profile` | 額外量測整體 SimulationContext.step，摘要中的巢狀時間不可直接相加；舊版不分開量 simulate／fetch |
| `--replay` | 額外重播每批保存的 action，檢查成功與 TCP 差異 |
| `--data_dir` | 必須是新的空資料夾，不覆寫或續寫已有 dataset |
| `--max_wall_seconds` | 收集迴圈（含可選回播）的總時間上限，預設 3600 s |
| `--reference_reads` | 效能比較用舊逐列讀取路徑，日常不要加 |

policy 與相機保持 30 Hz、物理保持 240 Hz；使用 fast expert。
相機用單一原生批次 view，所有列共用一次 render；CPU expert 資料按 dtype
打包傳輸，HDF5 使用 LZF 壓縮／分塊寫入。重力補償與 joint 目標按 servo 頻率
更新並保持，PhysX drive 仍每物理步運作，力矩預算與成功門檻未放寬。

上方較快指令已在本機四個孔位各兩回合通過：8/8 收集、8/8 回播。
純收集 217.4 秒（約 3 分 37 秒），含額外回播 332.2 秒，不含啟動／暖機。
平均吞吐約每回合 27.2 秒，不代表單回合延遲只有 27.2 秒。
2 環境同條件對照中，16/4 比 64/16 收集時間短約 39%；單純降 servo 頻率
沒有明顯加速。詳見 [效能與驗證紀錄](docs/EXPERT_SPEED_VALIDATION.md)。
LZF 為無損，但檔案可能大於先前 gzip pilot。CPU backend 在此次 lift 測試
失敗，不建議用於收集；目前通過的是 CUDA 路徑，8／16 環境尚未實測。

目前採固定孔位、整批結束後 barrier reset；不是逐列異步重派新孔位。
固定孔位入口以外，已新增下節的 `--all_grid` 跨批次排程；**正式 split manifests 尚未完成**。
不要把單個 grid ID 的重複回合視為不同位置或獨立資料多樣性。1.0／0.25／0.1 mm 可選，
不代表已完成所有 clearance 的動態驗證。
正式跨 clearance 實驗前應固定共同通過的數值設定，不可直接混用 0.50 mm 快版
與其他 clearance 的 baseline，然後把差異全部歸因於 clearance。

### 結果在哪裡、怎麼判讀

以 `data/expert_4env_001/` 為例：

| 檔案 | 內容 |
| --- | --- |
| `episodes/episode_000000.hdf5` 等 | 每回合獨立 RGB-D、state、action、時間與 metadata |
| `episodes/episode_000000/rgb_*.png` | 該回合 initial／middle／terminal 預覽 |
| `collection_summary.json` | 成功數、收集耗時、分段計時；完成時 state=completed |
| `manifest.json` | 每回合相對路徑、grid／env／batch ID、成功與結構檢查；split=unassigned |
| `environment_config.json` | 設定、版本與來源 hash；相機 K 與初始狀態在各 HDF5 metadata |
| `expert_diagnostics/batch_000/env_000/` | 分批分列的 expert 診斷 |

summary 的 `passed=true` 表示所有回合成功且結構對齊通過；若指定 replay，
也必須全部回播通過。失敗回合不丟棄，manifest 會保留 false 的結果。
`.partial.hdf5` 不是有效 demonstration，complete=true 也不能取代成功／結構檢查。
可在資料目錄建立 `STOP_REQUESTED` 檔案停止，未完成 episode 保留為 partial。

現有資料仍為原始 HDF5，不是 ACT／DP 專用轉換器；正式分割與模型載入尚未實作。
舊單回合試收 `data/consolidated_rgbd_pilot_v1/` 保留作歷史回歸，新結果請看各自 summary。

### 四種公差 × 17×17 全位置

完成新機小批驗證後，才啟動新 campaign。下例為 Ubuntu Bash：

```bash
python -u record_expert_data.py --all_grid --num_envs 4 \
  --data_dir data/full_grid_ubuntu_001
```

共 1,156 回合，四種公差交錯收集，每位置一回合。統一 baseline：物理／servo
240 Hz、policy／相機 30 Hz、64/16 求解器、無損 gzip，不自動加額外回播。
每批建立新 simulator 處理不同靜態孔位，保留成功與失敗，不只保存成功樣本。

- `status.json`：即時完成／失敗／待收數與預估時間；須檢查更新時間，不能只看 running。
- `dataset_c1p00`、`dataset_c0p50`、`dataset_c0p25`、`dataset_c0p10`：各自 manifest 與批次資料。
- `campaign.json`、`source_snapshot/`：排程、重試與來源紀錄；不是正式 train/test split。
- 在 campaign 根目錄建立 `STOP_REQUESTED` 可停止；磁碟不足或來源改動也會停下。

同機、同路徑、來源／資產未變且已確認前次程序退出時，才考慮 `--resume`。
重開機可能留下 `RUNNING.lock`；先核對程序與中斷批次，不要直接刪鎖重跑。
**Windows campaign 不可直接搬到 Ubuntu 加 --resume**：checkpoint 有絕對路徑、
資產時間戳及舊鎖。既有資料先保留，剩餘位置的遷移續收需另做核對。

目前 Windows campaign 已因重開機停止，登記完成 104／1,156，104 成功、0 失敗；
中斷批次另有 2 份完整 HDF5 與 2 份 partial 待核對。按使用者要求暫不續收。
舊 status.json 仍可能寫 running，不能解讀為背景仍在跑。

## 4. 程式與資料放置

| 路徑／程式 | 用途 |
| --- | --- |
| `env_yam.py` | 場景、reset、觀測、joint 與 Cartesian 控制 |
| `env_expert.py` | expert 階段流程與多環境調度 |
| `env_dataset.py` | 腕部相機、對齊檢查、HDF5 寫入 |
| `env_workspace_grid.py` | 網格、ID 與離線運動學篩查 |
| `robot_yam.py` | 手臂、碰撞與底層控制 |
| `env_config.py` | 集中參數；頻率、相機、物理、expert 分組設定 |
| `env_task.py` | 共用成功／幾何條件 |
| `assets/` | 必要模型，不是輸出，不能隨意刪除 |
| `data/` | 試收與正式 dataset |
| `outputs/` | viewer、診斷、效能測試輸出 |
| `validation/`、`archive/` | 驗證證據與歷史紀錄 |

`test_*.py` 是手動驗證工具，`unit_test/unittest_*.py` 是 CPU 單元測試，
不會被 collector 全部自動執行。完整用途見 [CODE_MAP](docs/CODE_MAP.md)。

## 5. 開發驗證（一般操作可跳過）

單位測試：

```powershell
python -B -m unittest discover -s unit_test -p "unittest_*.py" -q
```

無相機的中心／遠角 Cartesian 完整任務回歸：

```powershell
python -u .\test_cartesian_adapter.py --headless --expert --output_dir .\outputs\cartesian_check_001
```

多環境 expert 示範（無相機、不是 dataset collector）：

```powershell
python -u .\test_parallel_expert.py --num_envs 4 --grid_ids r08_c08 r00_c00 r00_c16 r16_c16 --clearance_mm 0.5 --output_dir .\outputs\parallel_check_001
```

現有部分驗證工具會拒絕覆寫，請使用新目錄。來源 fingerprint 改變後不要強制
resume 舊報告。289 點完整動態驗證留待正式資料收集時逐點記錄，不需現在
啟動舊 `test_environment_A.py` 長時間排程。

## 6. 進階文件

### Git 與資料搬遷

Git 納入程式、文件、unit_test 與必要 assets（含第三方 license／來源說明）。
data、outputs、validation、archive 的執行結果不進 Git，僅保留部分目錄 README；
環境、快取與本機編輯器設定也忽略。clone repo 不會取得既有 dataset 或驗證報告，
需要另行複製，不能因為 Git 中沒有它們就視為已刪除或已驗收。
必要 USD／STL 資產目前合計約 14 MB，直接納入一般 Git，未要求 Git LFS。
資產來源授權需保留；本次沒有替專案宣告新的開源 license。

- [程式用途與共用介面](docs/CODE_MAP.md)
- [詳細交接／Ubuntu 遷移](HANDOVER.md)
- [Cartesian action 定義](docs/CARTESIAN_ACTION_CONTRACT.md)
- [HDF5 資料格式](docs/DATASET_SCHEMA_V1.md)
- [原始實驗規劃](docs/Isaac_Lab_Pick_and_Insert_Experiment_Plan_V1.3.md)

其餘 `docs/*VALIDATION.md`、precision／profiling 文件是指定版本的驗證紀錄，
不是日常操作前必讀文件。跨批次孔位排程已實作；尚未完成正式 grid split、ACT／DP、
全網格收集／動態驗收、Ubuntu 實測、雙 GPU 分片及 Cartesian torque impedance control。
