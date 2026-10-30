"""Kiểm tra nhanh môi trường: GPU/torch, version thư viện, LIBERO render headless được không."""

import importlib
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

ok = True


def check(name, fn):
    global ok
    try:
        print(f"[ OK ] {name}: {fn()}")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")


def torch_info():
    import torch

    assert torch.cuda.is_available(), "torch không thấy GPU"
    cap = torch.cuda.get_device_capability(0)
    arch = f"sm_{cap[0]}{cap[1]}"
    assert arch in torch.cuda.get_arch_list(), f"bản torch này không build cho {arch}: {torch.cuda.get_arch_list()}"
    x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
    (x @ x).sum().item()
    return f"torch {torch.__version__}, CUDA {torch.version.cuda}, {torch.cuda.get_device_name(0)} ({arch})"


def versions():
    want = {"transformers": "4.40.1", "tokenizers": "0.19.1", "timm": "0.9.10"}
    got = {k: importlib.import_module(k).__version__ for k in want}
    bad = {k: v for k, v in got.items() if v != want[k]}
    assert not bad, f"sai version {bad}, cần {want}"
    return ", ".join(f"{k} {v}" for k, v in got.items())


def tf_cpu():
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    import tensorflow_datasets as tfds

    return f"tensorflow {tf.__version__}, tfds {tfds.__version__}"


def libero_render():
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(0)
    suite.get_task_init_states(0)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
    env.seed(0)
    env.reset()
    obs, *_ = env.step([0.0] * 6 + [-1.0])
    shape = obs["agentview_image"].shape
    env.close()
    return f"render OK {shape}, task 0 = '{task.language}'"


check("GPU / torch", torch_info)
check("thư viện model", versions)
check("tensorflow (CPU)", tf_cpu)
check("LIBERO + MuJoCo EGL", libero_render)
sys.exit(0 if ok else 1)
