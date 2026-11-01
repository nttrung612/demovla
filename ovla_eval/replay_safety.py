"""Phát lại các action đã lưu trong rollout, vừa chạy simulator vừa ghi tiếp xúc / lực / dịch chuyển vật thể.

Không cần GPU hay model: chỉ cần simulator. Dùng để trả lời "rollout đã chạy có tình huống mất an toàn nào không"
trước khi quyết định metric an toàn chính. Simulator phải xác định (deterministic): `replay_matches` cho biết
kết quả phát lại có khớp với rollout gốc (cùng success, cùng số bước) hay không.

Ví dụ:
    python -m ovla_eval.replay_safety --suite libero_spatial --rollouts outputs/rollouts/libero_spatial
    python -m ovla_eval.replay_safety --suite libero_spatial --rollouts outputs/rollouts/libero_spatial --tasks 1 8
"""

import argparse
import json
from pathlib import Path

from . import libero_utils as lu  # import trước để set biến môi trường MUJOCO_GL, ...
import numpy as np
import tqdm

from .constants import SUITES
from .safety_log import DISPLACEMENT_THRESHOLD, SafetyLogger


def replay_episode(env, init_state, actions: np.ndarray) -> dict:
    """Giống mở đầu của rollout_libero.run_episode: reset, nạp init state, chờ NUM_STEPS_WAIT bước rồi mới chạy action."""
    env.reset()
    env.set_init_state(init_state)
    for _ in range(lu.NUM_STEPS_WAIT):
        env.step(lu.DUMMY_ACTION)
    logger = SafetyLogger(env)
    done, n_steps = False, 0
    for action in actions:
        _, _, done, _ = env.step(action.tolist())
        logger.record()
        n_steps += 1
        if done:
            break
    return {"replay_success": bool(done), "replay_steps": n_steps, **logger.summary()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", choices=SUITES, required=True)
    ap.add_argument("--rollouts", required=True, help="thư mục chứa task_XX/ep_YYY.npz (vd outputs/rollouts/libero_spatial)")
    ap.add_argument("--tasks", type=int, nargs="*", help="chỉ phát lại các task id này (mặc định: tất cả có trong thư mục)")
    ap.add_argument("--out", help="mặc định: <rollouts>/safety_replay.json")
    args = ap.parse_args()

    root = Path(args.rollouts)
    suite = lu.get_task_suite(args.suite)
    task_dirs = sorted(root.glob("task_*"))
    if args.tasks:
        task_dirs = [d for d in task_dirs if int(d.name.split("_")[1]) in args.tasks]

    results = []
    for task_dir in tqdm.tqdm(task_dirs, desc="tasks"):
        task_id = int(task_dir.name.split("_")[1])
        task = suite.get_task(task_id)
        init_states = suite.get_task_init_states(task_id)
        env = lu.make_env(task)
        for npz in sorted(task_dir.glob("ep_*.npz")):
            with np.load(npz) as d:
                episode, saved_success, actions = int(d["episode"]), bool(d["success"]), d["pred_env"]
            res = replay_episode(env, init_states[episode], actions)
            res.update(
                task_id=task_id,
                episode=episode,
                instruction=task.language,
                saved_success=saved_success,
                saved_steps=int(len(actions)),
                replay_matches=bool(res["replay_success"] == saved_success and res["replay_steps"] == len(actions)),
            )
            results.append(res)
        env.close()

    out = Path(args.out) if args.out else root / "safety_replay.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print_report(results)
    print(f"Đã lưu vào {out}")


def print_report(results: list[dict]) -> None:
    n = len(results)
    if not n:
        print("Không có episode nào để phát lại.")
        return
    print(f"\n=== {n} episode ===")
    print(f"phát lại khớp rollout gốc: {sum(r['replay_matches'] for r in results)}/{n}")
    for name, group in (("thành công", [r for r in results if r["saved_success"]]), ("thất bại", [r for r in results if not r["saved_success"]])):
        if not group:
            continue
        moved = sum(bool(r["moved_nontarget"]) for r in group)
        touched = sum(r["unintended_steps"] > 0 for r in group)
        forces = [r["unintended_peak_force"] for r in group if r["unintended_steps"] > 0]
        print(
            f"{name:>10} ({len(group):3d} ep): vật ngoài mục tiêu bị đẩy > {DISPLACEMENT_THRESHOLD * 100:.0f} cm = {moved}, "
            f"robot chạm vật ngoài ý muốn = {touched}"
            + (f", lực đỉnh trung vị = {np.median(forces):.1f} N, lớn nhất = {max(forces):.1f} N" if forces else "")
        )
    world: dict[str, list] = {}
    for r in results:
        for name, w in r["world_contacts"].items():
            agg = world.setdefault(name, [0, 0, 0.0])
            agg[0] += 1
            agg[1] += w["steps"]
            agg[2] = max(agg[2], w["peak_force"])
    if world:
        print("\nRobot chạm vào \"world\" (body không thuộc vật/fixture nào), theo body/geom:")
        for name, (eps, steps, peak) in sorted(world.items(), key=lambda kv: -kv[1][2]):
            print(f"  {name:50} {eps:3d} episode, {steps:5d} bước, lực đỉnh {peak:.1f} N")
    print("\nVật ngoài mục tiêu bị đẩy nhiều nhất:")
    for r in sorted(results, key=lambda r: -max(r["moved_nontarget"].values(), default=0.0))[:5]:
        if r["moved_nontarget"]:
            moved = ", ".join(f"{k} {v * 100:.1f} cm" for k, v in r["moved_nontarget"].items())
            print(f"  task {r['task_id']:02d} ep {r['episode']:03d} ({'OK' if r['saved_success'] else 'FAIL'}): {moved}")


if __name__ == "__main__":
    main()
