# 歷史紀錄

此處全部從原專案搬移，沒有刪除或覆寫。日常執行不依賴這些資料。
目前正式驗證摘要在 [validation/physics](../validation/physics/physics_summary.json)。

| 現在位置 | 原本位置／用途 |
| --- | --- |
| `standalone_previous/` | 根目錄舊的兩個結果 JSON、trace、camera_check.png、grasp_validation_images/ |
| `release_diagnostics/` | 根目錄舊的隔離張爪測試輸出 |
| `refactor_before_merge/collision_geometry.py` | 整合進 robot_yam.py 前的舊模組副本，執行不再依賴它 |
| `validation/original_center/` | 舊網格中心的 IK 篩查 |
| `validation/physics_before_alignment_fix/` | 修改對位回授前的測試 |
| `validation/physics_distal_faces/` | 夾指接觸模型與手動壓入量探測，不是受控比較資料集 |
| `validation/physics_final/` | 名稱雖有 final，實際是中間網格的舊測試 |
| `validation/urdf10nm_trials/` | 對齊 10 Nm 時的初次五點失敗，以及 Cartesian 搬運／下探的中間探測；目前一致版本看專案 validation/urdf10nm_verified/ |
| `parallel_trials_20261111/` | 5 組失敗或不採信的多環境探測，附原始日誌與搬移清單；不是正式驗證結果 |

原始研究計畫與兩份說明文件已移到 `../docs/`，不是歸檔淘汰。
核心程式與正式驗證結果不依賴此歸檔目錄。工件現位於 `../assets/workpiece.stl`；目前程式名稱請看根目錄 README。
