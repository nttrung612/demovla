"""Hằng số dùng chung, không import thư viện nặng (để script download chạy được độc lập)."""

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")

BASE_CHECKPOINT = "openvla/openvla-7b"  # train trên Open-X, KHÔNG có norm stats của LIBERO
SUITE_CHECKPOINTS = {s: f"openvla/openvla-7b-finetuned-{s.replace('_', '-')}" for s in SUITES}

RLDS_DATASET_REPO = "openvla/modified_libero_rlds"


def rlds_name(suite: str) -> str:
    return f"{suite}_no_noops"


# Số bước tối đa mỗi episode, lấy từ run_libero_eval.py của OpenVLA (≈ episode demo dài nhất của suite)
SUITE_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}

# Success rate (%) công bố trong paper OpenVLA cho các checkpoint fine-tune, để đối chiếu
PAPER_SUCCESS_RATE = {
    "libero_spatial": 84.7,
    "libero_object": 88.4,
    "libero_goal": 79.2,
    "libero_10": 53.7,
}

ACTION_DIM_NAMES = ("dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper")
