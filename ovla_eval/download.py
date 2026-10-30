"""Tải trước checkpoint và dataset RLDS từ HuggingFace.

Ví dụ:
    python -m ovla_eval.download --suites libero_spatial                # checkpoint + dataset của 1 suite
    python -m ovla_eval.download --suites all --checkpoints --base      # tất cả checkpoint, kể cả openvla-7b gốc
"""

import argparse

from huggingface_hub import snapshot_download

from .constants import BASE_CHECKPOINT, RLDS_DATASET_REPO, SUITE_CHECKPOINTS, SUITES, rlds_name


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suites", nargs="+", default=["libero_spatial"], help=f"{list(SUITES)} hoặc 'all'")
    ap.add_argument("--checkpoints", action="store_true", help="chỉ tải checkpoint")
    ap.add_argument("--dataset", action="store_true", help="chỉ tải dataset")
    ap.add_argument("--base", action="store_true", help="tải thêm openvla/openvla-7b gốc")
    ap.add_argument("--data-dir", default="data/modified_libero_rlds")
    args = ap.parse_args()

    suites = list(SUITES) if args.suites == ["all"] else args.suites
    unknown = set(suites) - set(SUITES)
    if unknown:
        ap.error(f"suite không hợp lệ: {sorted(unknown)}")
    want_ckpt = args.checkpoints or not args.dataset
    want_data = args.dataset or not args.checkpoints

    if want_ckpt:
        repos = [SUITE_CHECKPOINTS[s] for s in suites] + ([BASE_CHECKPOINT] if args.base else [])
        for repo in repos:
            print(f"==> checkpoint {repo} (~15 GB, vào HF cache)")
            snapshot_download(repo)
    if want_data:
        patterns = [f"{rlds_name(s)}/*" for s in suites]
        print(f"==> dataset {RLDS_DATASET_REPO}: {patterns} -> {args.data_dir}")
        snapshot_download(RLDS_DATASET_REPO, repo_type="dataset", allow_patterns=patterns, local_dir=args.data_dir)
    print("Xong.")


if __name__ == "__main__":
    main()
