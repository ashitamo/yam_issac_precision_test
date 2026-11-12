"""Run the current-controller A validation campaign, with durable progress.

Only validation is performed. No training split or IL dataset is created.
"""
import argparse
import ctypes
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import env_config as cfg
from test_physics import source_signature, write_json, read_json

DEFAULT_OUTPUT = cfg.PROJECT_ROOT / 'validation' / 'environment_A_10nm'


def pid_running(pid):
    """Windows read-only query. Never use os.kill(pid, 0) on Windows."""
    if os.name != 'nt':
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
    kernel = ctypes.windll.kernel32
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return kernel.GetLastError() == 5  # Access denied: conservatively assume active.
    try:
        code = ctypes.c_ulong()
        return not kernel.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value == 259
    finally:
        kernel.CloseHandle(handle)


def update_review(root, state, signature, active_stage=None, error=None):
    reports = {name: read_json(root / name / 'physics_summary.json') for name in ['center_c0p10', 'grid_c0p50']}
    center, grid = reports['center_c0p10'], reports['grid_c0p50']
    cp, gp = center.get('progress', {}), grid.get('progress', {})
    center_complete = cp.get('finished') == cp.get('total') == 3
    grid_complete = gp.get('finished') == gp.get('total') == 289
    center_pass = bool(center_complete and center.get('all_requested_cases_pass')
                       and center.get('all_profile_comparisons_pass'))
    grid_pass = bool(grid_complete and grid.get('all_requested_cases_pass'))
    status = dict(state=state, pid=os.getpid(), active_stage=active_stage, source_signature=signature,
                  updated_unix_s=time.time(), error=error,
                  center_c0p10=cp, grid_c0p50=gp,
                  center_convergence_pass=center_pass, grid_all_cases_pass=grid_pass,
                  A_requested_validation_pass=center_pass and grid_pass)
    write_json(root / 'status.json', status)
    lines = ['# A 環境驗證進度', '', f'狀態 `{state}`；目前階段 `{active_stage}`。',
             '', '| 驗證 | 已結束／總數 | 任務成功 | 任務＋接觸檢查成功 |', '| --- | --- | --- | --- |',
             f'| 0.10 mm 中心三種 profile | {cp.get("finished",0)}/3 | {cp.get("task_passed",0)} | {cp.get("fully_passed",0)} |',
             f'| 0.50 mm 全網格 baseline | {gp.get("finished",0)}/289 | {gp.get("task_passed",0)} | {gp.get("fully_passed",0)} |',
             '', '## A 的正式驗證範圍', '',
             '- A.1–A.4 資產／位置限制／10 Nm motor 上限、四種 peg 生成、幾何成功判定已實作。資產檢查見 assets.log。',
             f'- A.5/A.6 全網格動態覆蓋：{"289 案已結束" if grid_complete else "進行中／尚未開始"}。全部任務與接觸檢查通過：{grid_pass}。',
             f'- A.7 新 10 Nm／新 Cartesian reference 的 0.10 mm 中心比較通過：{center_pass}。',
             '- 0.50 mm 全網格只跑 baseline；不是每個位置都做三種 profile 比較。',
             '- 每個位置、每個條件只跑一案；不是多 seed 重複試驗的統計成功率。',
             '- 不是實機驗證，也不另行認證全部 self-collision／掃掠體積。',
             '- 1.00/0.25 mm 的動態可靠度尚未驗證。', '',
             '## 失敗位置與診斷', '',
             '- `grid_c0p50/physics_summary.md`、`task_grid.svg`：全網格失敗位置、階段及程式回報原因。',
             '- `center_c0p10/physics_summary.json`：三種 profile 的深度／XY spread 及接觸診斷。',
             '- 分類是觸發的檢查，不能直接當作確定的碰撞對象或物理根因；每案保留完整 log／trace。', '',
             '## B 的進入條件', '',
             '只有在所有要求案例已結束、沒有基礎設施錯誤且上述兩組通過，才可將本次 A 驗證範圍視為通過。',
             '若有失敗，先依位置與階段診斷，不刪除失敗點來湊通過率、不默默修改固定網格。',
             'B 尚未開始。之後才固定 train/validation/dev/final-test manifests、Observation/Action 與 recorder。', '',
             '## 操作', '',
             '`python test_environment_A.py --stop` 請求停止；`--resume` 在相同來源下續跑。',
             '保持電腦開啟、不休眠。批次期間不要改程式／資產，來源改變會停止以避免混用版本。']
    if error:
        lines += ['', '## 執行錯誤', '', str(error)]
    (root / 'A_REVIEW.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop', action='store_true')
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    stop_file, lock_file = root / 'STOP_REQUESTED', root / 'run.lock'
    if args.stop:
        stop_file.write_text('User requested campaign stop\n', encoding='utf-8')
        print('Stop requested. The active case and its own child process will be stopped.', flush=True)
        return
    if lock_file.exists():
        lock = read_json(lock_file)
        if lock.get('pid') and pid_running(int(lock['pid'])):
            raise SystemExit(f'Campaign already running, PID {lock["pid"]}')
        if not args.resume:
            raise SystemExit('Stale run.lock exists. Use --resume after checking the previous process.')
        lock_file.unlink()
    if stop_file.exists():
        if not args.resume:
            raise SystemExit('Stop request exists; use --resume to explicitly resume.')
        stop_file.unlink()
    if (root / 'campaign.json').exists() and not args.resume:
        raise SystemExit('Output exists. Use --resume or a new --output_dir; old results are retained.')
    # Exclusive lock prevents duplicate launches using this campaign folder.
    with lock_file.open('x', encoding='utf-8') as stream:
        json.dump(dict(pid=os.getpid(), started_unix_s=time.time()), stream)
    signature = source_signature()
    child = None
    try:
        old_campaign = read_json(root / 'campaign.json')
        if old_campaign and old_campaign.get('source_signature') != signature:
            raise RuntimeError('Sources changed. Use a new campaign output directory to retain coherent results.')
        versions = {}
        for name in ['isaacsim', 'isaaclab', 'torch', 'numpy', 'scipy', 'trimesh']:
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        manifest = dict(source_signature=signature, python=sys.version, executable=sys.executable,
                        platform=platform.platform(), machine=platform.node(), versions=versions,
                        grid_expected=289, center_expected=3,
                        order=['center_c0p10', 'grid_c0p50'], created_unix_s=time.time())
        if old_campaign:
            for key in ['versions', 'machine', 'python']:
                if old_campaign.get(key) != manifest[key]:
                    raise RuntimeError('Runtime changed; use a new output directory instead of resume.')
        else:
            write_json(root / 'campaign.json', manifest)
            snapshot = root / 'source_snapshot'
            snapshot.mkdir()
            for file in cfg.PROJECT_ROOT.glob('*.py'):
                shutil.copy2(file, snapshot / file.name)
        update_review(root, 'running', signature, 'asset_preflight')
        with (root / 'assets.log').open('w', encoding='utf-8') as log:
            check = subprocess.run([sys.executable, '-u', str(cfg.PROJECT_ROOT / 'test_assets.py')],
                                   cwd=cfg.PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=120)
        if check.returncode:
            raise RuntimeError('Asset preflight failed; inspect assets.log')
        shutil.copy2(cfg.PROJECT_ROOT / 'validation/asset_audit.json', root / 'asset_audit.json')
        commands = [
            ('center_c0p10', ['--clearances', '0.1', '--grid_ids', 'r08_c08', '--profiles', 'baseline', 'half_dt', 'strong_solver']),
            ('grid_c0p50', ['--clearances', '0.5', '--profiles', 'baseline', '--all_grid']),
        ]
        for name, options in commands:
            if stop_file.exists():
                update_review(root, 'stopped', signature, name)
                return
            folder = root / name
            folder.mkdir(exist_ok=True)
            cmd = [sys.executable, '-u', str(cfg.PROJECT_ROOT / 'test_physics.py'), *options,
                   '--output_dir', str(folder), '--expected_signature', signature,
                   '--stop_file', str(stop_file), '--resume']
            print('[CAMPAIGN START]', name, flush=True)
            with (root / (name + '.log')).open('a', encoding='utf-8') as log:
                child = subprocess.Popen(cmd, cwd=cfg.PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT)
                while child.poll() is None:
                    update_review(root, 'running', signature, name)
                    time.sleep(5)
            code = child.returncode
            child = None
            update_review(root, 'running', signature, name)
            if stop_file.exists() or code == 130:
                update_review(root, 'stopped', signature, name)
                return
            if code:
                raise RuntimeError(f'{name} runner failed with code {code}; inspect its campaign log')
            print('[CAMPAIGN END]', name, flush=True)
        status = update_review(root, 'completed', signature)
        print('[CAMPAIGN COMPLETED]', status, flush=True)
    except BaseException as error:
        stop_file.write_text('Campaign interrupted\n', encoding='utf-8')
        if child is not None and child.poll() is None:
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                # Only our own active process tree is targeted.
                if os.name == 'nt':
                    subprocess.run(['taskkill', '/PID', str(child.pid), '/T', '/F'], capture_output=True)
                else:
                    child.kill()
                child.wait(timeout=10)
        update_review(root, 'error', signature, error=str(error))
        raise
    finally:
        lock_file.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
