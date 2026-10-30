"""Infer 1 bước để kiểm tra checkpoint chạy được và xem chi tiết đầu ra của model.

Ví dụ:
    # ảnh tuỳ ý + câu lệnh (base model, dùng norm stats của BridgeData)
    python -m ovla_eval.infer_single --checkpoint openvla/openvla-7b --unnorm-key bridge_orig \
        --image frame.png --instruction "pick up the black bowl"

    # lấy 1 frame trong dataset LIBERO RLDS, so với ground truth + kiểm tra với vla.predict_action()
    python -m ovla_eval.infer_single --rlds-suite libero_spatial --episode 0 --step 30 --verify --save-input in.png
"""

import argparse

import numpy as np
from PIL import Image

from .constants import ACTION_DIM_NAMES, SUITE_CHECKPOINTS, SUITES, rlds_name
from .policy import OpenVLAPolicy, gripper_to_env, set_seed
from .preprocess import resize_image


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="ảnh RGB (sẽ resize về 224 như lúc train)")
    src.add_argument("--rlds-suite", choices=SUITES, help="lấy ảnh + câu lệnh + action GT từ dataset RLDS")
    ap.add_argument("--instruction", help="bắt buộc khi dùng --image")
    ap.add_argument("--data-dir", default="data/modified_libero_rlds")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--step", type=int, default=0)
    ap.add_argument("--checkpoint", help="mặc định: checkpoint fine-tune của --rlds-suite")
    ap.add_argument("--unnorm-key")
    ap.add_argument("--no-center-crop", action="store_true")
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--verify", action="store_true", help="so với kết quả của vla.predict_action()")
    ap.add_argument("--save-input", help="lưu ảnh model thực sự nhìn thấy (sau resize + crop) để kiểm tra")
    args = ap.parse_args()
    set_seed(7)

    gt = None
    if args.image:
        if not args.instruction:
            ap.error("--image cần đi kèm --instruction")
        if not args.checkpoint:
            ap.error("--image cần chỉ rõ --checkpoint")
        img = resize_image(np.asarray(Image.open(args.image).convert("RGB")))
        instruction = args.instruction
        checkpoint, unnorm_key = args.checkpoint, args.unnorm_key
    else:
        from .rlds_utils import instruction_of, iter_episodes

        _, steps = next(iter_episodes(args.data_dir, rlds_name(args.rlds_suite), args.episode, 1))
        step = steps[min(args.step, len(steps) - 1)]
        img = resize_image(step["observation"]["image"], jpeg_roundtrip=False)
        instruction = instruction_of(steps)
        gt = np.asarray(step["action"], dtype=np.float64)
        checkpoint = args.checkpoint or SUITE_CHECKPOINTS[args.rlds_suite]
        unnorm_key = args.unnorm_key or args.rlds_suite
        print(f"episode {args.episode}, step {args.step}/{len(steps) - 1}")

    policy = OpenVLAPolicy(checkpoint, unnorm_key, args.device, args.attn, use_center_crop=not args.no_center_crop)
    print(f"checkpoint   : {checkpoint}")
    print(f"unnorm_key   : {policy.unnorm_key}   (checkpoint có {len(policy.norm_stat_keys)} bộ norm stats)")
    print(f"instruction  : {instruction}")

    if args.save_input:
        from .preprocess import center_crop

        Image.fromarray(center_crop(img) if policy.use_center_crop else img).save(args.save_input)
        print(f"đã lưu ảnh đầu vào: {args.save_input}")

    pred = policy.predict(img, instruction)
    header = f"{'dim':>8} {'raw':>9} {'env':>9} {'token':>7} {'bin':>4} {'p(token)':>9} {'entropy':>8} {'E[a]':>9}"
    print("\n" + header + (f" {'GT':>9} {'|err|':>8}" if gt is not None else ""))
    for i, name in enumerate(ACTION_DIM_NAMES[: policy.action_dim]):
        row = (
            f"{name:>8} {pred.raw[i]:>9.4f} {pred.env[i]:>9.4f} {pred.token_ids[i]:>7d} {pred.bins[i]:>4d} "
            f"{pred.token_prob[i]:>9.3f} {pred.bin_entropy[i]:>8.3f} {pred.expected_raw[i]:>9.4f}"
        )
        if gt is not None:
            row += f" {gt[i]:>9.4f} {abs(pred.env[i] - gt[i]):>8.4f}"
        print(row)
    print(
        "\nraw = action sau un-normalize (gripper: 1 = mở, 0 = đóng);  env = action gửi vào LIBERO (gripper: -1 = mở, +1 = đóng)"
        "\np(token) = xác suất của token được chọn;  entropy trên 256 bin (max ≈ 5.55);  E[a] = kỳ vọng action theo phân phối bin"
    )

    if args.verify:
        ref = policy.predict_reference(img, instruction)
        diff = np.abs(ref - pred.raw).max()
        print(f"\nverify: vla.predict_action() = {np.round(ref, 4)}")
        print(f"        max |diff| = {diff:.2e}  -> {'KHỚP' if diff < 1e-6 else 'KHÔNG KHỚP, kiểm tra lại decode'}")
        print(f"        env action từ predict_action() = {np.round(gripper_to_env(ref), 4)}")


if __name__ == "__main__":
    main()
