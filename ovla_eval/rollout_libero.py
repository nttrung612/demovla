"""Chạy closed-loop rollout OpenVLA trong simulator LIBERO.

Logic giống run_libero_eval.py của OpenVLA (cùng tiền xử lý, max steps, init states, seed) nhưng lưu lại
toàn bộ trajectory: action từng bước, độ tự tin của model, trạng thái end-effector, video.

Ví dụ:
    python -m ovla_eval.rollout_libero --suite libero_spatial --trials 5
    python -m ovla_eval.rollout_libero --suite libero_goal --tasks 0 3 --trials 50
    python -m ovla_eval.rollout_libero --suite libero_spatial --response mask --no-video \\
        --thresholds outputs/rollouts/libero_spatial/monitor.json --out-dir outputs/responses/mask
"""

import argparse
import json
import time
import traceback
from dataclasses import replace
from pathlib import Path

from . import libero_utils as lu  # import trước để set biến môi trường MUJOCO_GL, ...
import imageio
import numpy as np
import tqdm

from .constants import PAPER_SUCCESS_RATE, SUITE_CHECKPOINTS, SUITE_MAX_STEPS, SUITES
from .monitor import LIFT_DZ, LIFT_STEPS, ResponseGuard
from .policy import OpenVLAPolicy, set_seed
from .preprocess import resize_image


DEFAULT_OUT_DIR = "outputs/rollouts"


class PolicyError(Exception):
    """Lỗi của model hoặc decode. Không được tính là lỗi simulator: phải dừng cả lần chạy thay vì ghi episode thất bại."""


def run_episode(env, policy: OpenVLAPolicy, instruction: str, init_state, max_steps: int, guard: ResponseGuard | None = None) -> dict:
    env.reset()
    obs = env.set_init_state(init_state)
    frames, records, states = [], [], []
    success, error = False, None
    t = 0
    try:
        while t < max_steps + lu.NUM_STEPS_WAIT:
            if t < lu.NUM_STEPS_WAIT:
                obs, _, _, _ = env.step(lu.DUMMY_ACTION)
                t += 1
                continue
            img = resize_image(lu.agentview_image(obs))
            frames.append(img)
            states.append(lu.eef_state(obs))
            try:
                pred = policy.predict(img, instruction, dz_min=guard.dz_min if guard else None)
            except Exception as e:
                raise PolicyError(f"policy.predict lỗi ở bước {t}") from e
            if guard:  # pred_env lưu action thực sự gửi vào env, vì replay_safety phát lại đúng mảng này
                pred = replace(pred, env=guard.step(pred, states[-1]))
            records.append(pred)
            obs, _, done, _ = env.step(pred.env.tolist())
            if done:
                success = True
                break
            t += 1
    except PolicyError:
        raise
    except Exception:  # giống OpenVLA: lỗi simulator thì tính là thất bại, chạy tiếp episode khác
        error = traceback.format_exc()

    def stack(field):
        return np.stack([getattr(r, field) for r in records]) if records else np.zeros((0, policy.action_dim))

    return {
        "success": success,
        "error": error,
        "frames": frames,
        "arrays": {
            "pred_raw": stack("raw"),
            "pred_env": stack("env"),
            "token_ids": stack("token_ids"),
            "token_prob": stack("token_prob"),
            "bin_entropy": stack("bin_entropy"),
            "expected_raw": stack("expected_raw"),
            "eef_state": np.stack(states) if states else np.zeros((0, 8)),
            **({"alarm_steps": np.array(guard.alarm_steps, dtype=np.int64)} if guard else {}),
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", choices=SUITES, required=True)
    ap.add_argument("--checkpoint", help="mặc định: checkpoint fine-tune tương ứng với suite")
    ap.add_argument("--unnorm-key", help="mặc định: tên suite (tự thêm hậu tố _no_noops nếu cần)")
    ap.add_argument("--tasks", type=int, nargs="*", help="chỉ chạy các task id này (mặc định: tất cả)")
    ap.add_argument("--trials", type=int, default=50, help="số episode mỗi task (tối đa 50 init states)")
    ap.add_argument("--no-center-crop", action="store_true")
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument(
        "--fresh-env-at",
        type=int,
        nargs="*",
        default=[],
        help="tạo lại env trước các episode này, để khớp baseline từng resume bằng process mới (xem replay_safety)",
    )
    ap.add_argument("--response", default="none", choices=["none", "hold", "clip", "mask"], help="phản ứng sau báo động")
    ap.add_argument("--thresholds", help="monitor.json của ovla_eval.monitor, bắt buộc khi có --response")
    ap.add_argument("--lift-dz", type=float, default=LIFT_DZ, help="dz tối thiểu khi nhấc tay (clip, mask)")
    ap.add_argument("--lift-steps", type=int, default=LIFT_STEPS, help="số bước nhấc tay sau mỗi báo động (clip, mask)")
    args = ap.parse_args()
    if args.response != "none":
        if not args.thresholds:
            ap.error("--response cần --thresholds")
        if args.out_dir == DEFAULT_OUT_DIR:
            ap.error("--response cần --out-dir riêng để không trộn với kết quả baseline")

    set_seed(args.seed)
    monitor_config = json.loads(Path(args.thresholds).read_text()) if args.thresholds else None
    if monitor_config and monitor_config["suite"] != args.suite:
        print(f"Ngưỡng được hiệu chỉnh trên {monitor_config['suite']}: dùng bộ ngưỡng chung cho {args.suite}")
    checkpoint = args.checkpoint or SUITE_CHECKPOINTS[args.suite]
    policy = OpenVLAPolicy(
        checkpoint,
        unnorm_key=args.unnorm_key or args.suite,
        device=args.device,
        attn_implementation=args.attn,
        use_center_crop=not args.no_center_crop,
    )
    print(f"checkpoint={checkpoint}  unnorm_key={policy.unnorm_key}  center_crop={policy.use_center_crop}")

    suite = lu.get_task_suite(args.suite)
    task_ids = args.tasks if args.tasks else list(range(suite.n_tasks))
    max_steps = SUITE_MAX_STEPS[args.suite]
    out_root = Path(args.out_dir) / args.suite
    out_root.mkdir(parents=True, exist_ok=True)

    def make_guard(task_id: int) -> ResponseGuard | None:
        if args.response == "none":
            return None
        same_suite = monitor_config["suite"] == args.suite
        row = monitor_config["thresholds"][str(task_id) if same_suite else "all"]
        thresholds = {s: (float("inf") if v is None else v) for s, v in row.items()}
        return ResponseGuard(thresholds, args.response, args.lift_dz, args.lift_steps)

    results = {}
    for task_id in tqdm.tqdm(task_ids, desc="tasks"):
        task = suite.get_task(task_id)
        init_states = suite.get_task_init_states(task_id)
        n_trials = min(args.trials, len(init_states))
        task_dir = out_root / f"task_{task_id:02d}"
        task_dir.mkdir(exist_ok=True)
        env = None
        episodes = []
        for ep in range(n_trials):
            npz_path = task_dir / f"ep_{ep:03d}.npz"
            if npz_path.exists():  # resume: episode đã chạy rồi
                with np.load(npz_path) as d:
                    episodes.append({"episode": ep, "success": bool(d["success"]), "steps": int(len(d["pred_env"]))})
                continue
            if env is not None and ep in args.fresh_env_at:
                env.close()
                env = None
            if env is None:
                env = lu.make_env(task)
            start = time.time()
            res = run_episode(env, policy, task.language, init_states[ep], max_steps, make_guard(task_id))
            n_steps = len(res["arrays"]["pred_env"])
            np.savez_compressed(
                npz_path,
                success=res["success"],
                instruction=task.language,
                suite=args.suite,
                task_id=task_id,
                episode=ep,
                checkpoint=checkpoint,
                unnorm_key=policy.unnorm_key,
                **res["arrays"],
            )
            if not args.no_video and res["frames"]:
                tag = "success" if res["success"] else "fail"
                imageio.mimwrite(task_dir / f"ep_{ep:03d}_{tag}.mp4", res["frames"], fps=30)
            if res["error"]:
                (task_dir / f"ep_{ep:03d}_error.txt").write_text(res["error"])
            episodes.append({"episode": ep, "success": res["success"], "steps": n_steps})
            done = sum(e["success"] for e in episodes)
            tqdm.tqdm.write(
                f"[task {task_id:02d}] ep {ep:03d}: {'OK  ' if res['success'] else 'FAIL'} "
                f"steps={n_steps:3d} ({n_steps / (time.time() - start):.1f} it/s)  task SR={done}/{len(episodes)}"
                + ("  (có lỗi, xem _error.txt)" if res["error"] else "")
            )
        if env is not None:
            env.close()
        n_ok = sum(e["success"] for e in episodes)
        results[task_id] = {
            "instruction": task.language,
            "success_rate": n_ok / max(len(episodes), 1),
            "n_success": n_ok,
            "n_trials": len(episodes),
            "episodes": episodes,
        }

    total_ok = sum(r["n_success"] for r in results.values())
    total = sum(r["n_trials"] for r in results.values())
    summary = {
        "suite": args.suite,
        "checkpoint": checkpoint,
        "unnorm_key": policy.unnorm_key,
        "center_crop": policy.use_center_crop,
        "seed": args.seed,
        "response": args.response,
        "success_rate": total_ok / max(total, 1),
        "paper_success_rate": PAPER_SUCCESS_RATE.get(args.suite, float("nan")) / 100,
        "tasks": results,
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n=== Kết quả ===")
    for task_id, r in results.items():
        print(f"task {task_id:02d}  {r['success_rate'] * 100:5.1f}%  ({r['n_success']}/{r['n_trials']})  {r['instruction']}")
    print(
        f"TỔNG: {summary['success_rate'] * 100:.1f}% ({total_ok}/{total})   "
        f"paper: {PAPER_SUCCESS_RATE.get(args.suite, float('nan')):.1f}%"
    )
    print(f"Đã lưu vào {out_root}")


if __name__ == "__main__":
    main()
