# 共用觀測與連續回合驗證

## 本輪結果

使用 0.50 mm、fast expert、既有 240 Hz/64/16 與使用者接受的 0.050 mm
幾何診斷容差，4 個固定孔位，各 3 回合，**12/12 環境與 expert 結果一致成功**。
沒有相機、policy、HDF5 recorder 或 289 點排程。

| 環境 | 孔位 | 三回合成功 | 每回合模擬秒數 |
| --- | --- | --- | --- |
| 0 | r08_c08 | 3/3 | 12.129、12.125、12.138 |
| 1 | r00_c00 | 3/3 | 12.358、11.971、11.946 |
| 2 | r00_c16 | 3/3 | 11.738、11.471、11.425 |
| 3 | r16_c16 | 3/3 | 13.375、13.338、13.513 |

共 9654 物理步，global physics time 40.225 秒；執行迴圈約 307.39 秒，
不含最初 AppLauncher／場景建立。這是新增觀測與診斷路徑的功能回歸，
不是效能提升證據，不宣稱單環境超即時或全網格可靠度。

## 已檢查

- 8 次局部 reset（每列兩次），每次只重設已完成的列；實際觀察到其他列
  仍在執行時即接續下一回合，並非全 batch 同步重開。
- reset 前後其他列的關節位置／速度、末端、開口、peg pose/velocity、
  previous_action、episode ID 與時鐘逐值相同；控制器物件也未替換。
- reset 不推進物理時間；指定列 episode time 歸零，episode ID 依序 1、2、3。
- 同一環境每個物理步時間戳向前；重設不把 global physics timestamp 歸零。
- 每回合獨立 task 結果、trace 與 environment_terminal.json，12 份 terminal
  snapshot 在 reset 前保存；已有檔案時拒絕覆寫。
- 12 份 expert 結果的 observation_source 均為 shared_env。

結果根目錄：`validation/shared_repeat_4x3_v1/`。
摘要：`repeat_summary.json`；每回合目錄為 env_000（第一回合）、
env_000/episode_001、env_000/episode_002，其餘列同理。

## Expert 觀測路徑

新路徑必須明確提供 snapshot。末端、開口、joint telemetry、peg 與孔位
皆從同一份環境觀測讀取；不再由 expert 直接存取 simulator state。
缺少 snapshot 或 episode／時間不符即報錯，不靜默 fallback。
環境仍負責控制寫入、重力 FF、物理步進、task termination 與 reset。
expert 只產生 actions 並維護 phase／IK／診斷，靜態模型參數仍可讀取。

`diagnostics` 含 implicit-PD 力矩估計，並非實際 contact-force sensor。
`privileged` 含精確 peg／孔位；兩者不自動等同於最終 policy 的可用輸入。
保留的舊驗證入口仍可明確選用 legacy simulator-read 路徑作回歸參考。

## 負向與時間邊界驗證

`outputs/env_interface_lifecycle_v2/interface_result.json` 原生短測試通過：
局部 reset 後，刻意注入第 0 列 controller failure、第 1 列 episode timeout，
驗證 terminated／truncated 區分、兩份 final_observation、零成功 reward，
下一步不重複發出 newly_done，也不自動 reset。這是刻意注入的狀態測試，
不是兩個機器人操作失敗案例。

35 個 CPU 單位測試通過，包括禁止 simulator read/write 的 expert 物件、
錯誤 episode／倒退時間、每物理子步 snapshot、一回合一次 reward、接觸
超界不可被後續落底掩蓋，以及原有控制／幾何回歸。

## 重跑

```powershell
python -B -m unittest discover -s unit_test -p "unittest_*.py" -v
python test_repeated_rollout.py --episodes_per_env 3 --output_dir validation/repeat_new
python test_env_interface.py --output_dir outputs/lifecycle_new
```

所有測試需使用空的輸出目錄；重跑不覆寫舊結果。可在連續測試輸出目錄
建立 STOP_REQUESTED 結束本次模擬。環境本身仍為手動 reset，執行器保存
終止快照後按列呼叫 reset，不會把下一回合初始畫面當成上一回合最後一幀。

## 下一步仍屬 B 前／B 的工作

固定最終 policy Observation/Action 與 policy／servo 頻率；接相機同步與
正式 episode recorder，再做少量資料品質檢查。尚未開始訓練資料收集。
