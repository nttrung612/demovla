"""Đọc dataset LIBERO dạng RLDS (openvla/modified_libero_rlds) bằng tensorflow_datasets.

Ảnh trong dataset này đã được xoay đúng chiều và đã lọc các action no-op, đúng với dữ liệu fine-tune OpenVLA.
"""

from pathlib import Path

from . import preprocess  # noqa: F401  (ẩn GPU khỏi TF trước khi tfds khởi tạo TF)
import tensorflow_datasets as tfds


def find_builder_dir(data_dir: str | Path, name: str) -> Path:
    root = Path(data_dir) / name
    candidates = sorted(p.parent for p in root.rglob("dataset_info.json"))
    if not candidates:
        raise FileNotFoundError(
            f"Không thấy dataset '{name}' trong {root}. Chạy: python -m ovla_eval.download --dataset --suites ..."
        )
    return candidates[-1]  # version mới nhất


def iter_episodes(data_dir: str | Path, name: str, start: int = 0, count: int | None = None):
    """Yield (episode_index, list các step dạng numpy dict)."""
    builder = tfds.builder_from_directory(str(find_builder_dir(data_dir, name)))
    ds = builder.as_dataset(split="train", shuffle_files=False).skip(start)
    if count is not None:
        ds = ds.take(count)
    for offset, episode in enumerate(ds):
        yield start + offset, list(episode["steps"].as_numpy_iterator())


def num_episodes(data_dir: str | Path, name: str) -> int:
    builder = tfds.builder_from_directory(str(find_builder_dir(data_dir, name)))
    return builder.info.splits["train"].num_examples


def instruction_of(steps) -> str:
    instr = steps[0]["language_instruction"]
    return instr.decode() if isinstance(instr, bytes) else str(instr)
