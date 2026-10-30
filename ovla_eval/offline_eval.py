"""So sánh action dự đoán của OpenVLA với action ground-truth (demo của người) trên dataset LIBERO RLDS.

Mỗi bước, model nhìn đúng ảnh trong demo (teacher forcing) nên không cần simulator. Đây là phép đo
"model bắt chước demo tốt đến đâu", KHÔNG thay được rollout: sai số nhỏ vẫn có thể tích luỹ thành thất bại.

Ví dụ:
    python -m ovla_eval.offline_eval --suite libero_spatial --max-episodes 20
    python -m ovla_eval.offline_eval --suite libero_10 --start-episode 100 --max-episodes 5 --save-frames
"""

import argparse
import json
from pathlib import Path

import numpy as np
import tqdm

from .constants import ACTION_DIM_NAMES, SUITE_CHECKPOINTS, SUITES, rlds_name
from .policy import OpenVLAPolicy, set_seed
from .preprocess import resize_image
from .rlds_utils import instruction_of, iter_episodes, num_episodes


def compute_metrics(policy: OpenVLAPolicy, gt: np.ndarray, pred_env: np.ndarray, expected_raw: np.ndarray) -> dict:
    """gt, pred_env: (N, 7) theo quy ước LIBERO (gripper -1 = mở, +1 = đóng)."""
    cont = slice(0, 6)
    err = pred_env[:, cont] - gt[:, cont]
    norm_err = policy.normalize(pred_env)[:, cont] - policy.normalize(gt)[:, cont]
    return {
        "n_steps": int(len(gt)),
        # Sai số đơn vị gốc của dataset (action LIBERO đã scale về khoảng [-1, 1])
        "mae": np.abs(err).mean(0).tolist(),
        "rmse": np.sqrt((err**2).mean(0)).tolist(),
        # Sai số trong không gian chuẩn hoá [-1, 1] mà model thấy; 1 bin ≈ 2/255 ≈ 0.0078
        "mae_normalized": np.abs(norm_err).mean(0).tolist(),
        # Soft decode (kỳ vọng theo phân phối bin) thay vì argmax
        "mae_expected": np.abs(expected_raw[:, cont] - gt[:, cont]).mean(0).tolist(),
        # Baseline đoán toàn 0: model phải tốt hơn hẳn con số này mới có ý nghĩa
        "mae_zero_baseline": np.abs(gt[:, cont]).mean(0).tolist(),
        "gripper_accuracy": float((np.sign(pred_env[:, 6]) == np.sign(gt[:, 6])).mean()),
    }


def print_table(title: str, m: dict, extra: dict | None = None) -> None:
    print(f"\n{title}  ({m['n_steps']} steps)")
    cols = ["mae", "rmse", "mae_normalized", "mae_expected", "mae_zero_baseline"]
    extra = extra or {}
    print(f"{'dim':>8} " + " ".join(f"{c:>18}" for c in cols + list(extra)))
    for i, name in enumerate(ACTION_DIM_NAMES[:6]):
        print(f"{name:>8} " + " ".join(f"{m[c][i]:>18.4f}" for c in cols) + " ".join(f"{v[i]:>19.3f}" for v in extra.values()))
    print(f"{'gripper':>8}  accuracy = {m['gripper_accuracy'] * 100:.1f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", choices=SUITES, required=True)
    ap.add_argument("--data-dir", default="data/modified_libero_rlds")
    ap.add_argument("--checkpoint", help="mặc định: checkpoint fine-tune tương ứng với suite")
    ap.add_argument("--unnorm-key", help="mặc định: tên suite (tự thêm hậu tố _no_noops nếu cần)")
    ap.add_argument("--start-episode", type=int, default=0)
    ap.add_argument("--max-episodes", type=int, default=20, help="-1 = toàn bộ dataset (rất lâu)")
    ap.add_argument("--stride", type=int, default=1, help="chỉ infer mỗi `stride` bước để chạy nhanh hơn")
    ap.add_argument("--no-center-crop", action="store_true")
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--save-frames", action="store_true", help="lưu ảnh 224x224 model nhìn thấy vào npz")
    ap.add_argument("--out-dir", default="outputs/offline")
    args = ap.parse_args()

    set_seed(args.seed)
    checkpoint = args.checkpoint or SUITE_CHECKPOINTS[args.suite]
    policy = OpenVLAPolicy(
        checkpoint,
        unnorm_key=args.unnorm_key or args.suite,
        device=args.device,
        attn_implementation=args.attn,
        use_center_crop=not args.no_center_crop,
    )
    dataset = rlds_name(args.suite)
    total_eps = num_episodes(args.data_dir, dataset)
    count = None if args.max_episodes < 0 else args.max_episodes
    print(f"checkpoint={checkpoint}  unnorm_key={policy.unnorm_key}  dataset={dataset} ({total_eps} episodes)")

    out_root = Path(args.out_dir) / args.suite
    out_root.mkdir(parents=True, exist_ok=True)

    all_gt, all_env, all_exp, all_prob, all_ent = [], [], [], [], []
    per_episode = []
    n_eps = min(count, total_eps - args.start_episode) if count is not None else total_eps - args.start_episode
    for ep_idx, steps in tqdm.tqdm(iter_episodes(args.data_dir, dataset, args.start_episode, count), total=n_eps):
        instruction = instruction_of(steps)
        idx = np.arange(0, len(steps), args.stride)
        frames, preds = [], []
        for t in idx:
            img = resize_image(steps[t]["observation"]["image"], jpeg_roundtrip=False)
            preds.append(policy.predict(img, instruction))
            if args.save_frames:
                frames.append(img)

        gt = np.stack([steps[t]["action"] for t in idx]).astype(np.float64)
        extra_arrays = {}
        if "state" in steps[0]["observation"]:
            extra_arrays["eef_state"] = np.stack([steps[t]["observation"]["state"] for t in idx])
        if frames:
            extra_arrays["frames"] = np.stack(frames)
        arrays = {f: np.stack([getattr(p, f) for p in preds]) for f in ("raw", "env", "token_ids", "token_prob", "bin_entropy", "expected_raw")}
        metrics = compute_metrics(policy, gt, arrays["env"], arrays["expected_raw"])
        np.savez_compressed(
            out_root / f"ep_{ep_idx:04d}.npz",
            instruction=instruction,
            suite=args.suite,
            episode=ep_idx,
            checkpoint=checkpoint,
            unnorm_key=policy.unnorm_key,
            step_index=idx,
            gt_action=gt,
            pred_raw=arrays["raw"],
            pred_env=arrays["env"],
            token_ids=arrays["token_ids"],
            token_prob=arrays["token_prob"],
            bin_entropy=arrays["bin_entropy"],
            expected_raw=arrays["expected_raw"],
            **extra_arrays,
        )
        per_episode.append({"episode": ep_idx, "instruction": instruction, **metrics})
        tqdm.tqdm.write(
            f"ep {ep_idx:04d}  T={len(steps):3d}  MAE(xyz,rpy)={np.mean(metrics['mae']):.4f}  "
            f"gripper acc={metrics['gripper_accuracy'] * 100:5.1f}%  | {instruction}"
        )
        all_gt.append(gt)
        all_env.append(arrays["env"])
        all_exp.append(arrays["expected_raw"])
        all_prob.append(arrays["token_prob"])
        all_ent.append(arrays["bin_entropy"])

    if not per_episode:
        print("Không có episode nào được đánh giá (kiểm tra --start-episode / --max-episodes).")
        return
    overall = compute_metrics(policy, np.concatenate(all_gt), np.concatenate(all_env), np.concatenate(all_exp))
    prob, ent = np.concatenate(all_prob), np.concatenate(all_ent)
    overall["mean_token_prob"] = prob.mean(0).tolist()
    overall["mean_bin_entropy"] = ent.mean(0).tolist()
    summary = {
        "suite": args.suite,
        "dataset": dataset,
        "checkpoint": checkpoint,
        "unnorm_key": policy.unnorm_key,
        "center_crop": policy.use_center_crop,
        "stride": args.stride,
        "action_dims": list(ACTION_DIM_NAMES),
        "overall": overall,
        "episodes": per_episode,
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print_table(
        f"=== {args.suite}: {len(per_episode)} episodes ===",
        overall,
        extra={"token_prob": overall["mean_token_prob"], "entropy": overall["mean_bin_entropy"]},
    )
    print(f"{'':>8}  token_prob gripper = {prob[:, 6].mean():.3f}, entropy gripper = {ent[:, 6].mean():.3f}")
    print(f"Đã lưu vào {out_root}")


if __name__ == "__main__":
    main()
