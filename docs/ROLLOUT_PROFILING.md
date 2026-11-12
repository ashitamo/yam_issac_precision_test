# Rollout 分段計時（2026-11-11）

## 使用

```powershell
python test_parallel_expert.py --num_envs 1 --profile `
  --output_dir outputs/profile_demo
```

不指定 --profile 時不啟用 cProfile；物理、IK、控制頻率、任務判定不變。
沒有額外 CUDA synchronize，因此是呼叫端牆鐘時間歸因，不是隔離的 GPU
kernel 耗時。非同步 GPU 工作可能計入稍後等待它的區段。

輸出：

- `profile.json`：分段總時間、次數、平均耗時與函式明細。
- `profile.txt`：按自身／累計耗時排序的 Python 呼叫列表。
- `rollout.pstats`：可用 pstats／視覺化 profiler 開啟的原始資料。
- 原有 summary、task 結果與 trace 仍會產生；例外離開迴圈時也保存 profile。

頂層 command、scene、observe 互不重疊。scene.write、scene.physics、
scene.update 是 scene 內的子區段，不可和 scene 再加總。
cProfile cumulative 是含子呼叫的時間，也不可逐項相加。

## 實測

中心 r08_c08、0.50 mm、1 環境、無視窗／無相機，5179 步。
抓取／插入／開爪落底與 contact gate 全部通過。
模擬時間 21.579 秒，開啟 profiling 的迴圈牆鐘時間 103.838 秒。

| 區段（不重複加總） | 秒 | 比例 |
| --- | ---: | ---: |
| 模擬步進 sim.step | 66.402 | 63.95% |
| 控制計算與命令提交（含部分 IK） | 13.260 | 12.77% |
| 觀測、資料讀回、判定、階段切換（含部分 IK） | 12.295 | 11.84% |
| 將命令緩衝寫入模擬 | 9.961 | 9.59% |
| 更新 articulation／peg 狀態緩衝 | 1.499 | 1.44% |
| 迴圈與計時等其餘開銷 | 約 0.421 | 約 0.41% |

結果：`outputs/profile_single_v1/`。先前未開啟 profiling 的同一中心案例
約 93.59 秒，顯示 profiling 本身有成本；不是效能退步或加速成果的證據。
此次沒有改物理參數，亦沒有重跑 289 點。

函式計時中 solve_ik 共 236 次、累計約 5.06 秒（分布於 command 與
observe 階段切換）；力矩限制更新共 5179 次、約 3.74 秒。
CPU tensor 讀回共 46659 次、約 1.47 秒；檔案 open 約 0.09 秒、print 約
0.001 秒。這些是 cProfile 歸因，不能和區段時間再次加總，也不能由 CPU
讀回的數字推斷所有 GPU 同步成本。

重要限制：cProfile 自身只歸因約 36.6 秒，未覆蓋全部原生引擎工作，不能
只看其排名就認定 IK 或 Python 是主要瓶頸。區段 perf_counter 才是本次
總時間占比的依據。sim.step 的 66.4 秒也不等於純 GPU solver 時間。

## 結論與下一步

主要瓶頸在「引擎步進與其等待」，不是已證實由 SciPy IK 主導。
本機 PhysXManager.step 依序呼叫 simulate 與 fetch_results；下一輪應
用原生引擎／GPU 計時再區分碰撞、求解、CPU 工作與同步等待。

其次才是減少命令寫入與重複資料讀取。每步動態力矩限制確實會回傳 CPU
寫入 PhysX，但僅移除此步既不安全（會破壞 10 Nm 總力矩預算），也不能
解釋目前大部分時間。

暫不降低 240 Hz、64/16 求解迭代或碰撞精度。若之後測精度／效能折衷，
必須另立案例，通過完整插入與接觸誤差回歸後才可採用。

新增 profiler 的例外保存測試後，CPU 回歸共 20 個，全部通過。
