"""Tạo môi trường LIBERO và lấy observation, giống experiments/robot/libero/libero_utils.py của OpenVLA."""

import math
import os

# Phải set trước khi import robosuite/mujoco: render headless bằng EGL trên server
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
# torch>=2.6 mặc định torch.load(weights_only=True), làm LIBERO không load được file init state (pickle numpy)
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import numpy as np  # noqa: E402
from libero.libero import benchmark, get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402

ENV_RESOLUTION = 256  # độ phân giải dùng khi tạo dataset RLDS
NUM_STEPS_WAIT = 10  # bỏ qua vài bước đầu vì simulator thả rơi vật thể lúc khởi tạo
DUMMY_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


def get_task_suite(name: str):
    return benchmark.get_benchmark_dict()[name]()


def make_env(task, resolution: int = ENV_RESOLUTION, seed: int = 0):
    bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl_file, camera_heights=resolution, camera_widths=resolution)
    env.seed(seed)
    return env


def agentview_image(obs) -> np.ndarray:
    """Camera của LIBERO bị lộn ngược; xoay 180° để khớp với ảnh trong dataset train."""
    return np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.array(quat, dtype=np.float64, copy=True)
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def eef_state(obs) -> np.ndarray:
    """(8,) = vị trí end-effector (3) + hướng dạng axis-angle (3) + vị trí 2 ngón gripper (2)."""
    return np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]])
