"""Vẽ biểu đồ từ file .npz do rollout_libero / offline_eval lưu ra.

Mỗi episode -> 1 PNG: 7 chiều action theo thời gian (dự đoán, và ground truth nếu có)
+ heatmap xác suất token (model tự tin đến đâu ở từng chiều, từng bước).
Thư mục offline có summary.json -> thêm biểu đồ MAE từng chiều so với baseline đoán toàn 0.

Ví dụ:
    python -m ovla_eval.plot outputs/offline/libero_spatial
    python -m ovla_eval.plot outputs/rollouts/libero_spatial/task_03 --limit 5
    python -m ovla_eval.plot outputs/rollouts/libero_spatial/task_03/ep_000.npz
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from .constants import ACTION_DIM_NAMES  # noqa: E402

PRED_COLOR = "#2a78d6"
GT_COLOR = "#eb6834"
BASELINE_COLOR = "#c3c2b7"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
CONFIDENCE_CMAP = LinearSegmentedColormap.from_list(
    "confidence", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_SECONDARY,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "text.color": INK,
        "font.size": 9,
        "legend.frameon": False,
    }
)


def _str(x) -> str:
    return str(x.item() if isinstance(x, np.ndarray) else x)


def plot_episode(npz_path: Path, out_path: Path) -> None:
    d = np.load(npz_path)
    pred = d["pred_env"]
    gt = d["gt_action"] if "gt_action" in d else None
    steps = d["step_index"] if "step_index" in d else np.arange(len(pred))
    n_dims = pred.shape[1]

    fig, axes = plt.subplots(
        n_dims + 1, 1, figsize=(11, 1.35 * n_dims + 2.2), sharex=True,
        gridspec_kw={"height_ratios": [1] * n_dims + [1.6]},
    )
    for i in range(n_dims):
        ax = axes[i]
        draw = ax.step if i == n_dims - 1 else ax.plot
        kw = {"where": "post"} if i == n_dims - 1 else {}
        if gt is not None:
            draw(steps, gt[:, i], color=GT_COLOR, lw=1.5, label="ground truth (demo)", **kw)
        draw(steps, pred[:, i], color=PRED_COLOR, lw=1.5, label="OpenVLA", **kw)
        ax.set_ylabel(ACTION_DIM_NAMES[i], rotation=0, ha="right", va="center")
        ax.axhline(0, color=AXIS, lw=0.8, zorder=0)
    axes[n_dims - 1].set_yticks([-1, 1], ["mở -1", "đóng +1"])
    if gt is not None:
        axes[0].legend(loc="upper right", ncol=2)

    hm = axes[n_dims]
    hm.grid(False)
    extent = (steps[0] - 0.5, steps[-1] + 0.5, n_dims - 0.5, -0.5) if len(steps) else None
    im = hm.imshow(d["token_prob"].T, aspect="auto", cmap=CONFIDENCE_CMAP, vmin=0, vmax=1, extent=extent, interpolation="nearest")
    hm.set_yticks(range(n_dims), ACTION_DIM_NAMES[:n_dims])
    hm.set_xlabel("bước (timestep)")
    hm.set_title("p(token) — xác suất model gán cho token action đã chọn", loc="left", fontsize=9, color=INK_SECONDARY)
    cbar = fig.colorbar(im, ax=axes.tolist(), location="right", fraction=0.025, pad=0.01)
    cbar.outline.set_visible(False)

    title = _str(d["instruction"])
    if "success" in d:
        title += "   —   " + ("THÀNH CÔNG" if bool(d["success"]) else "THẤT BẠI")
    if gt is not None:
        title += f"   —   MAE(xyz,rpy) = {np.abs(pred[:, :6] - gt[:, :6]).mean():.4f}"
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_offline_summary(summary_path: Path, out_path: Path) -> None:
    s = json.loads(summary_path.read_text())
    o = s["overall"]
    names = ACTION_DIM_NAMES[:6]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.grid(axis="x", visible=False)
    ax.bar(x - 0.2, o["mae_zero_baseline"], width=0.38, color=BASELINE_COLOR, label="baseline: đoán toàn 0")
    bars = ax.bar(x + 0.2, o["mae"], width=0.38, color=PRED_COLOR, label="OpenVLA")
    ax.bar_label(bars, fmt="%.3f", fontsize=8, color=INK_SECONDARY, padding=2)
    ax.set_xticks(x, names)
    ax.set_ylabel("MAE")
    ax.legend(loc="upper right")
    ax.set_title(
        f"{s['suite']}: {len(s['episodes'])} episodes, {o['n_steps']} steps — gripper accuracy {o['gripper_accuracy'] * 100:.1f}%",
        loc="left",
        fontsize=10,
    )
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="file .npz hoặc thư mục chứa .npz")
    ap.add_argument("--limit", type=int, default=20, help="vẽ tối đa bao nhiêu episode")
    ap.add_argument("--out-dir", help="mặc định: <thư mục>/plots")
    args = ap.parse_args()

    path = Path(args.path)
    files = [path] if path.is_file() else sorted(path.rglob("*.npz"))[: args.limit]
    root = path.parent if path.is_file() else path
    out_dir = Path(args.out_dir) if args.out_dir else root / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    for f in files:
        with np.load(f) as d:
            if len(d["pred_env"]) == 0:
                print(f"bỏ qua {f}: episode không có bước nào (xem file _error.txt)")
                continue
        out = out_dir / (f.relative_to(root).with_suffix(".png").as_posix().replace("/", "__"))
        plot_episode(f, out)
        print(f"-> {out}")
    summary = root / "summary.json"
    if summary.exists() and "overall" in json.loads(summary.read_text()):
        out = out_dir / "summary_mae.png"
        plot_offline_summary(summary, out)
        print(f"-> {out}")


if __name__ == "__main__":
    main()
