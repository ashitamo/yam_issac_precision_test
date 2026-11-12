# URDF torque 與中心／四角驗證

## 限制設定

- 依目前 URDF，joint1..joint6 的 motor effort 上限都是 10 Nm，取代前版 200 Nm。
- 速度上限保留 3 rad/s，低於 URDF 的 10 rad/s。Joint reference 速率仍限制為 0.5 rad/s。
- PhysX implicit PD 的 maxForce 與額外 feedforward effort 分開，因此先將 FF 限制在 ±10 Nm，再將 PD drive 上限設為 `10 - abs(FF)`。
- 這是保守的總 motor torque 預算，不代表已識別實機馬達模型。Gripper 仍為原每指 1.5 N，摩擦與增益未增加。

報告中的 `arm_peak_requested_effort_nm` 是限制前的 PD+FF 需求，起始 transient 可超過 10。
`arm_peak_estimated_effort_nm` 是 implicit actuator 模型的限幅估計，不是實際力矩感測器。
`arm_saturation_time_s` 記錄限制前需求達到 nominal motor cap 的時間；保守 PD 預留額度也可能更早限幅。
被動碰撞／關節反作用力與 motor torque 是不同量，不能直接混用。

## 修正的邊界問題

最初的五點測試在最遠角落 `r16_c16` 失敗。單獨追蹤終點 joint targets，會使搬運中間的末端高度
從目標約 140 mm 下沉到 trace 記錄的約 77 mm，並有約 9.5° 姿態偏差。
只修正搬運後，下探對位的相同問題仍會觸發滑移判定，因此兩段都改為平滑 Cartesian reference。
上述中間探測保留於 `archive/validation/urdf10nm_trials/`，沒有刪除失敗案例。

- 搬運保持高度與姿態，參考速度上限 100 mm/s、加速度上限 200 mm/s²。
- 下探保持 XY 與姿態，參考速度上限 50 mm/s、加速度上限 100 mm/s²。
- Quintic blend 的起終點速度、加速度為零，IK 更新 30 Hz。
- 待下探參考走完且手臂跟上後，才啟用精密對位回授。
- 沒有提高 torque／夾持力／摩擦，也沒有放寬 2 mm 滑移或插入成功判定。

## 五個固定位置

| Grid ID | Hole X (m) | Hole Y (m) |
| --- | --- | --- |
| r08_c08 | 0.265 | 0.065 |
| r00_c00 | 0.165 | -0.035 |
| r00_c16 | 0.365 | -0.035 |
| r16_c00 | 0.165 | 0.165 |
| r16_c16 | 0.365 | 0.165 |

同一版本、0.50 mm nominal radial clearance、240 Hz physics/control、64/16 solver iterations。
結果以 `validation/urdf10nm_verified/physics_summary.json` 及各 case 的完整 JSON/trace/log 為準。

最終同版本重跑結果為 **5/5 任務成功且接觸診斷通過**。

| Grid ID | 最終深度 (mm) | 插入前 XY 誤差 (mm) | 最大夾持相對滑移 (mm) |
| --- | --- | --- | --- |
| r08_c08 | 約 32.0 | 0.001146 | 0.003097 |
| r00_c00 | 約 32.0 | 0.005750 | 0.002828 |
| r00_c16 | 約 32.0 | 0.002719 | 0.004876 |
| r16_c00 | 約 32.0 | 0.000135 | 0.003221 |
| r16_c16 | 約 32.0 | 0.003131 | 0.005483 |

14 項單元測試通過。每案的 implicit torque 模型估計峰值皆不超過 10 Nm，
並已驗證來源 fingerprint 相符時 `--resume` 正確重用五案。
上述誤差是此模擬器、固定初始條件的量測，不代表實機精度或統計成功率。

```powershell
python -u .\test_physics.py --profiles baseline --clearances 0.5 --grid_ids r08_c08 r00_c00 r00_c16 r16_c00 r16_c16 --output_dir .\validation\urdf10nm_new_run
```

這是單一 baseline profile 的五點任務檢查，不是跨 profile 收斂測試。
因此 summary 的 `all_profile_comparisons_pass` 會是 false（未執行完整三 profile），
任務與接觸診斷應看 `all_requested_cases_pass` 及每案狀態。

本次未完成全 289 點動態測試，也未在新 10 Nm／新動作參考下重做 0.10 mm 的步長收斂測試。
先前 `validation/physics/` 六組 200 Nm 結果保留為歷史資料，不作為新設定的通過證據。
