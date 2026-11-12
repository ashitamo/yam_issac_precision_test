# Isaac Lab Pick-and-Insert 實驗規劃 V1.3

> **研究主題（暫定）**：Investigating Precision Limitations, Overfitting, and Impedance Control in Imitation Learning for Peg-in-Hole Manipulation  
> **更新重點**：不同 Radial Clearance 對應**不同的 Demonstration Dataset 與獨立訓練的 Policy**；不做 Multi-Clearance Training。Peg 由 Gripper 實際夾取，而非固定於末端。

## 1. 研究目的與問題

本研究以 NVIDIA Isaac Lab 的 **Pick-and-Insert** 任務，分析模仿學習（Imitation Learning, IL）在高精度操作上的性能，以及訓練過程與低階控制器對結果的影響。

| 編號 | Research Question | 核心比較 | 主要指標 |
|---|---|---|---|
| **RQ1** | **IL 是否存在 Precision Limitation？** | 不同 Radial Clearance 下，各自訓練的 ACT / Diffusion Policy | Task Success、Insertion Success、Pre-Insertion XY Error |
| **RQ2** | **Overfitting 的模型在 Seen Configuration 下是否有更好的精度？** | 同一資料集／訓練 Run 的不同 Checkpoints；Seen vs Unseen | Train/Val Loss、Seen/Unseen XY Error、Success |
| **RQ3** | **Cartesian Impedance Control 是否改善精密操作能力？** | 固定相同 Policy／初始條件，替換 Position 與 Cartesian Impedance Controller | Insertion Success、XY Error；必要時診斷 Contact Force |

**研究界線：**

- 不預設 IL 有固定的精度上限，也不預設 Overfitting 一定改善 Seen 精度。
- **成功插入 ≠ 插入前定位誤差已小於 Clearance**；接觸與順應性可能協助完成插入。
- RQ3 主要比較**接觸插入能力**，不預設 Impedance Control 會提高自由空間絕對定位精度。
- Checkpoint Selection / Validation Loss 分析是研究 RQ2 的工具，不是唯一研究主題。

## 2. 模擬環境與任務

| 項目 | 規劃 |
|---|---|
| Simulator | NVIDIA Isaac Lab / Isaac Sim |
| Robot | I2RT YAM（先確認資產、末端與可用控制介面） |
| Gripper | Parallel-Jaw Gripper（具體型號待確認） |
| Task | **Pick-and-Insert**：Approach Peg → Grasp → Lift → Move/Align → Insert |
| Workpiece | **圓柱體 Ø32 × 32 mm** |
| Hole | 中央 Through-Hole |
| Hole Diameter | **Ø10 mm（主要實驗）**；Ø16 mm（後續可選，不納入第一輪） |
| Peg | 圓柱，長 **50 mm**，直徑依 Clearance 決定 |
| Peg Attachment | **實際夾取**；不得固定焊接於 TCP |
| Peg Initial Pose | 固定、可重現 |
| Peg / Hole Orientation | 固定，不做姿態泛化 |
| Target Insertion Depth | **30 mm（暫定）** |
| Placement Error / Randomization | 第一版不加入 |
| Policy Observation | RGB（候選：Wrist + Fixed Camera）+ Robot State；確切欄位待確認 |
| Policy Action | 兩種 IL 共用一致的 Action Definition（候選：Cartesian Delta Pose + Gripper Command） |
| 第一輪 Low-Level Controller | **Position Control**（具體 IK / Joint Tracking 流程待定） |

**幾何與成功判定的最低要求：**

1. Peg 必須由 Gripper **實際抓起**，而非透過固定約束直接跟隨末端。
2. **Grasp Success**：Peg 已被夾持、抬離桌面，且未掉落。
3. **Insertion Success**：以模擬器 Ground-Truth Peg / Hole Geometry 驗證 Peg 真正進入 Hole 且插入深度 ≥ 30 mm；不能只看 TCP 下降距離。
4. **Task Success = Grasp Success 且 Insertion Success**。
5. **Pre-Insertion XY Error**：在預先定義的接近插入時刻，計算 Peg 軸線與 Hole 軸線在 Hole 入口平面的水平偏差（mm）。固定該測量時刻／條件，避免不同實驗使用不同的取樣點。

Gripper 夾持幾何只需要能穩定抓取、完成 30 mm 插入且不發生非預期干涉；**不把夾持位置或 Peg 外露長度當成主要研究指標**。

## 3. Workspace 與 Grid

| 項目 | 規劃 |
|---|---|
| Workspace | **200 × 200 mm**（Hole Center 的 XY 範圍） |
| Grid | **17 × 17**（289 個位置） |
| Center-to-Center Spacing | **12.5 mm**；含區域兩端點：200/(17−1) |
| Seen / Unseen 基本配置 | **Checkerboard Interleaved** |
| Candidate Seen Positions | 145 |
| Candidate Unseen Positions | 144 |
| XY Perturbation | 不加入；每次使用精確的 Grid Coordinates |
| 前置驗證 | 先進行 YAM Reachability、Grasp / Insertion Expert 成功率檢查 |

對所有 Clearances，**使用同一份 Grid 位置清單及相同 Train/Validation/Evaluation Split**，避免空間配置成為混淆因素。

### 3.1 資料分割原則（待 Pilot 確認數量）

- **Train Positions**：由 145 個候選 Seen Grid Points 中選取，收集 BC Demonstrations。
- **Offline Validation Positions**：從上述候選集合保留**未進入訓練資料**的位置，生成獨立 Expert Demonstrations 供計算 Validation Loss。它們不應被標成已訓練過的精確位置。
- **Seen Rollout**：從真正用於訓練的 Grid Positions 中選取，用於衡量訓練配置下的重現能力。
- **Unseen Rollout（開發評估）**：從完全未進入 Train 的位置中選取，追蹤 Checkpoint 的空間插值表現。
- **Final Test**：另外保留**不參與 Checkpoint 選擇**的 Unseen Positions，最後才使用。

**可用的第一版配置（暫定）：** 120 Train Positions、25 Offline Validation Positions；定期 Rollout 各取 12 個 Seen / 12 個 Unseen Positions。正式數量與各集合 ID 由固定 Manifest 記錄。考慮避免 Offline Validation 和 Unseen Rollout 的評估重疊；若無法完全分離，必須明確記錄兩者共用的位置，且 Final Test 必須獨立。

**定義限制：** Checkerboard 的 Unseen Positions 多數位於訓練位置附近，因此測試的是 **Spatial Interpolation**，不宣稱是 Workspace 外的 Extrapolation。

## 4. RQ1 — 不同 Clearance 使用獨立 Dataset 與獨立訓練

**固定 Ø10 mm Hole**，藉由改變 Peg Diameter 設定 Radial Clearance：

\[
\text{Radial Clearance}=\frac{D_{\mathrm{hole}}-D_{\mathrm{peg}}}{2}
\]

| Radial Clearance | Hole Diameter | Peg Diameter | 獨立 Demonstration Dataset | 每 Seed 的模型 |
|---|---|---|---|---|
| **1.00 mm** | Ø10.00 mm | **Ø8.00 mm** | `dataset_c1p00` | ACT、DP |
| **0.50 mm** | Ø10.00 mm | **Ø9.00 mm** | `dataset_c0p50` | ACT、DP |
| **0.25 mm** | Ø10.00 mm | **Ø9.50 mm** | `dataset_c0p25` | ACT、DP |
| **0.10 mm** | Ø10.00 mm | **Ø9.80 mm** | `dataset_c0p10` | ACT、DP |

### 4.1 明確的訓練原則

1. **四種 Clearance 各自收集 Scripted Expert Demonstrations。**
2. **每一份 Dataset 都獨立訓練 ACT 與 Diffusion Policy。**
3. **不將四種 Clearance 混合成單一 Multi-Clearance Dataset。**
4. **不把只在一種 Peg Diameter 訓練的 Policy，直接測另一種 Peg Diameter，當成 RQ1 的主要結果。**
5. 各 Dataset 對應**同一組 Grid IDs、資料量、收集程序與訓練流程**；Peg Diameter / Clearance 為刻意改變的任務條件。
6. 每一種 Clearance 都先以 Scripted Expert 驗證成功率與接觸求解穩定性；尤其 **0.10 mm** 必須確認物理步長、接觸精度與求解器設定足夠可靠。

因此正式設計為：

\[
4\ \text{Clearances}\times2\ \text{Policies}\times3\ \text{Training Seeds}
=\boxed{24\ \text{Training Runs}}
\]

這是正式實驗的**目標數量**，不是一開始就全部執行。先從 **0.50 mm** 完成 Pilot。

### 4.2 RQ1 的主要評估

- Task Success Rate
- Insertion Success Rate（須清楚標記分母：全部 Rollouts 或僅成功 Grasp 的 Rollouts）
- Pre-Insertion XY Error（mm）

**解讀限制：** 不同 Clearance 同時代表不同 Peg Diameter、各自的資料與獨立訓練的模型。本研究比較的是 **在各自對應訓練條件下，IL 面對更嚴格插入需求的操作表現**，而非聲稱純粹改變 Clearance 的單一因果效果。

## 5. RQ2 — Overfitting 與 Seen Precision

### 5.1 要檢驗的假設

> 隨訓練進展，IL 對已訓練 Grid Positions 的操作精度，可能在 Validation Loss 不再改善甚至上升後繼續提升；這個趨勢不一定發生於 Unseen Positions。

不預設假設成立。**Validation Loss 上升不是 Overfitting 的唯一判準**；須結合 Training Loss、Train/Val Gap、Seen/Unseen Rollout 指標分析。

### 5.2 訓練與 Checkpoint Schedule

| 項目 | 暫定設定 |
|---|---|
| Policies | ACT、Diffusion Policy |
| 每 Clearance 的 Dataset | 獨立一份，兩種 IL 使用相同資料 |
| Training Budget | **100 Epochs**（Pilot 後檢查是否充分） |
| Actual Optimizer Steps | 依 DataLoader / Batch Size / Gradient Accumulation 紀錄 |
| Seeds | **3**（正式實驗；Pilot 各先跑 1） |
| Train / Validation Loss | 每 Epoch 記錄 |
| Checkpoint Save | 每 **5%** Training Progress（5%、10%……100%） |
| Rollout Evaluation | 每 **10%** Training Progress（10%、20%……100%） |
| 定期 Rollout Positions | 暫定 **12 Seen + 12 Unseen**；所有 Checkpoints 固定同一份名單 |

**百分比優先於固定 Step 數：**

\[
S_{\mathrm{total}}=\text{100 Epochs 的實際 Optimizer Steps}
\]

\[
S_{\mathrm{eval},k}\approx \frac{k}{10}S_{\mathrm{total}},\quad k=1,\ldots,10
\]

目前 100 Epochs，因此每 10 Epochs Rollout 一次；實際還需記錄對應 Optimizer Step。不同 Policy 的同一百分比不代表相同收斂程度。

### 5.3 核心評估與分析

| 指標 | 分析目的 |
|---|---|
| Training Loss | 擬合程度 |
| Validation Loss | 離線泛化指標 |
| Held-out Action Error（可選） | 跨檢查點的動作預測誤差；需固定採樣／推論設定 |
| Seen Pre-Insertion XY Error | 已訓練位置的精度是否改善 |
| Unseen Pre-Insertion XY Error | 未訓練位置是否同步改善 |
| Seen / Unseen Task Success | 精度差異是否轉化成任務結果 |

**主要比較方式：** 在**同一 Clearance、同一 Policy、同一 Seed** 中，分析 Checkpoint 變化；之後再彙整各 Seeds。ACT 和 DP 的原始 Training Loss 定義不同，不直接比較其絕對數值。

**必做圖表：**

1. Training / Validation Loss vs Training Progress。
2. Seen / Unseen Pre-Insertion XY Error vs Training Progress。
3. Seen / Unseen Task Success vs Training Progress。
4. 最佳 Validation-Loss Checkpoint、Late Checkpoint、Best Development-Rollout Checkpoint 在**獨立 Final Test** 上的比較。

**評估避免洩漏：** 定期 Unseen Rollout 若被用來選 Checkpoint，就屬於 Development Evaluation，**不可再當最終測試集**。

## 6. RQ3 — Position Control vs Cartesian Impedance Control

在已完成 RQ1/RQ2 的模型上，選擇少數代表性 Checkpoints，比較兩種控制器，不重新展開全部 Checkpoints 的完整矩陣。

| 控制條件 | Position Control | Cartesian Impedance Control |
|---|---|---|
| Policy Weights | 相同 | 相同 |
| Policy Observation / Action | 相同 | 相同 |
| Peg/Hole Geometry | 相同 | 相同 |
| Initial Conditions | 相同 | 相同 |
| Free-Space Tracking | 先調校／驗證 | 先調校／驗證 |
| Contact Handling | Position Tracking | Cartesian Stiffness/Damping / Compliance |

主要觀察 Task Success、Insertion Success 和 Pre-Insertion XY Error。**若出現插入差異但對位誤差相近**，可再補 Contact Force、Tracking Error 作機制診斷。

**必要前置條件：** 確認 YAM 的 Isaac Lab 資產與 Actuator 模型支援所需的 Torque / Effort Command；若無法實作真正的 Cartesian Impedance，不能僅把 Position PD 改名稱當成阻抗控制。

**解讀注意：** Controller 切換可能改變 Closed-Loop State Distribution，不能直接把全部改善歸因於「定位能力提高」。

## 7. Scripted Expert 與資料收集

### 7.1 任務流程

1. Reset Robot / Peg / Workpiece 至指定 Grid Position。
2. Approach Peg。
3. Close Gripper / Grasp。
4. Lift Peg。
5. Move to Hole / Align。
6. Insert 至目標深度。
7. 以 Ground Truth 判定成功；儲存 Observation、Action、Robot State、Episode Metadata。

### 7.2 資料收集要求

- Expert 可以使用 Ground-Truth Peg/Hole Pose 產生動作，但 **Policy Observation 不應包含 Expert 專用的 Ground-Truth Hole Pose**。
- 同一 Grid Position、同一初始狀態、同一 Deterministic Scripted Expert 下，不要重複製作完全相同的示範充數。
- 不同 Clearances **各建獨立 Dataset**，但盡量讓 Train/Validation Position IDs 一致。
- 收集到的 Demonstrations 需標記 `clearance_mm`、`peg_diameter_mm`、`grid_id`、`episode_success`、`seed`、`initial_state`。
- **暫定每 Training Position 1 筆**，但約 120 筆示範是否足以訓練影像式 ACT / DP 尚未證實；若 Pilot 失敗，先檢查示範有效性、Action/Observation 設定及資料多樣性，再決定是否需要增加資料。

## 8. 實驗執行順序與預算

### Phase 0 — Environment & Expert Validation

- 完成 YAM / Gripper / Ø32 Workpiece / Peg-Hole 幾何。
- 實作 Pick-and-Insert Scripted Expert。
- 驗證 Reachability、成功判定與 0.50 mm Clearance 的接觸穩定性。

### Phase 1 — 0.50 mm Pilot

- 只建立 `dataset_c0p50`。
- ACT、DP **各訓練 1 Seed**，驗證 100 Epochs、Checkpoint Schedule 與 Rollout Pipeline。
- 確認能在 Seen / Unseen Positions 取得有意義、非全零或全滿分的評估訊號。

### Phase 2 — RQ1 + RQ2 正式實驗

- 建立四個獨立 Dataset；各自訓練 ACT、DP，各 3 Seeds。
- 每 5% Save，每 10% Rollout。
- 依 Clearance 分析精度限制及訓練過程表現。

**若所有 24 個 Runs 都採用 10 次 Checkpoint Rollout、每次 24 Positions：**

\[
24\times10\times24=5{,}760\ \text{Rollouts}
\]

此數字僅為**定期評估**，不含 Expert、Pilot、Final Test、RQ3。可利用較少的 Checkpoint Development 評估先篩選，再對重要 Checkpoints 完整評估，以控制成本；不要把所有評估條件一次展開。

### Phase 3 — RQ3 Controller Comparison

- 選擇少數代表性 Clearance 與 Checkpoints。
- 固定 Policy 和 Episode Initial States，比較 Position 與 Cartesian Impedance。
- 視結果決定是否需要診斷 Contact Force / Tracking Error。

### Phase 4 — 結果分析

- 梳理 RQ1：各自訓練下的 Clearance–Success / Error 關係。
- 梳理 RQ2：Train/Val Loss、Seen/Unseen Precision 與 Checkpoint 關係。
- 梳理 RQ3：Controller 對插入成功率與定位誤差的不同影響。
- 報告限制：Sim-to-Real、圓柱對稱性、單一任務、Dataset Size、物理接觸精度。

## 9. TODO Checklist

### A. Isaac Lab 環境

- [ ] 確認 YAM Robot USD、Joint Limits、Actuator 與 Gripper 控制能力。
- [ ] 建立 Ø32 × 32 mm 圓柱 Workpiece、Ø10 mm 中央通孔。
- [ ] 建立四種 Peg Diameter（Ø8.0 / Ø9.0 / Ø9.5 / Ø9.8 mm），Peg Length 50 mm。
- [ ] 定義 Peg Tip、Hole Top Plane、Insertion Depth 及 Success 判定。
- [ ] 實作 Grasp / Lift / Insert，確認不會因 Gripper / Workpiece 非預期干涉而失敗。
- [ ] 建立 200 × 200 mm、17 × 17 Grid，檢查所有候選位置的 Reachability。
- [ ] 驗證 Contact Solver／Physics Time Step，優先完成 0.50 mm，再驗證 0.10 mm。

### B. Expert & Dataset

- [ ] 完成 Pick-and-Insert Scripted Expert。
- [ ] 確認 Expert 在 0.50 mm 下穩定成功。
- [ ] 決定 Camera、Observation Fields、Policy Action Definition（包含 Gripper Command）。
- [ ] 固定 Train / Offline Validation / Development Rollout / Final Test Grid Manifests。
- [ ] 收集 `dataset_c0p50`，輸出 HDF5（或模型所需格式）。
- [ ] 檢查影像、Robot State、Action Timestamp 與成功標記。
- [ ] Pilot 通過後，建立 `dataset_c1p00`、`dataset_c0p25`、`dataset_c0p10`。

### C. Training & Evaluation

- [ ] ACT Training Pipeline：1-Seed Pilot。
- [ ] DP Training Pipeline：1-Seed Pilot。
- [ ] 確認 100 Epochs / Dataset Size 足以產生有意義的 Learning Curve。
- [ ] 每 5% Save Checkpoint、每 10% Rollout 的排程與 Logging。
- [ ] 計算 Pre-Insertion XY Error、Grasp / Insertion / Task Success。
- [ ] 固定 12 Seen + 12 Unseen Development Rollout Initial States。
- [ ] 確立獨立 Final Test，不能拿 Development 評估結果當最終測試。
- [ ] 正式執行 4 Clearances × 2 Policies × 3 Seeds。

### D. Analysis & Controller

- [ ] RQ1：比較各 Clearance 的 Task Success、Insertion Success、XY Error。
- [ ] RQ2：繪製 Train/Val Loss 與 Seen/Unseen Precision 的 Checkpoint Curves。
- [ ] RQ2：檢查是否真正出現可支持 Overfitting 的 Train/Val 分歧。
- [ ] RQ2：分析 Best Val Loss / Late / Best Development Rollout 的獨立 Final Test。
- [ ] 確認 Cartesian Impedance Controller 的實際 Torque/Effort 支援與調校。
- [ ] RQ3：固定 Policy Checkpoints 比較 Position / Cartesian Impedance。
- [ ] 必要時補充 Contact Force、Tracking Error、Failure Type 診斷。
- [ ] 整理研究限制、主要結論與後續改善方法。

## 10. 已確定 vs 尚待驗證

| 類型 | 項目 |
|---|---|
| **已決定** | 三個 RQs；Pick-and-Insert；圓柱 Workpiece Ø32 × 32 mm；Ø10 Hole 為主；四個 Clearance 各自建立 Dataset 並獨立訓練；200 × 200 mm、17 × 17 Grid；不加入 Placement Error；100 Epochs 作為初始預算；每 5% Save、每 10% Rollout |
| **初步暫定** | Peg Initial Pose；Target Insertion Depth 30 mm；Camera / Observation / Action；Position Controller 實作細節；Train 120 / Validation 25；12 Seen + 12 Unseen 定期 Rollout；3 Seeds |
| **需先驗證** | YAM Gripper / Torque Control 可行性；Contact Solver 在 0.10 mm Clearance 的可靠性；約 120 筆示範是否足以訓練；100 Epochs 是否足夠；Peg 真正能被夾持並完成插入 |
| **目前不做** | Multi-Clearance Training；額外 Placement Error；Orientation Generalization；Cross-Embodiment；將 Ø16 Hole 納入第一輪 |

---

**執行原則：** 先做 **Ø10 Hole、0.50 mm Radial Clearance、Position Control 的完整 Pilot**，再擴充不同 Clearances，並使用同一批訓練所得 Checkpoints 同時回答 RQ1 與 RQ2；最後挑選少數代表性模型進行 RQ3 Controller Comparison。
