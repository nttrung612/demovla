#!/usr/bin/env bash
# Cài môi trường OpenVLA-7B + LIBERO trên server Linux có GPU RTX 5090 (Blackwell, sm_120).
#
# Khác với hướng dẫn gốc của OpenVLA (torch 2.2.0 + flash-attn 2.5.5, không chạy được trên sm_120):
#   - torch bản CUDA 12.8 (>= 2.7)
#   - không cài package `openvla`; model load thẳng từ HuggingFace với attention SDPA
#
# Dùng:  bash scripts/install_server.sh          (ENV_NAME=openvla mặc định)
set -euo pipefail

ENV_NAME="${ENV_NAME:-openvla}"
TORCH_VERSION="${TORCH_VERSION:-2.7.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.22.1}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY="$REPO_ROOT/third_party"

if ! command -v conda >/dev/null 2>&1; then
  echo "Cần conda (Miniconda/Miniforge) trên server." >&2
  exit 1
fi
eval "$(conda shell.bash hook)"
if ! conda env list | awk '{print $1}' | grep -x "$ENV_NAME" >/dev/null; then
  conda create -y -n "$ENV_NAME" python=3.10
fi
set +u  # script activate của conda dùng biến chưa khai báo
conda activate "$ENV_NAME"
set -u
# Không lấy package từ ~/.local của Python cùng phiên bản khi cài hoặc kiểm tra.
export PYTHONNOUSERSITE=1
python -m pip install --upgrade pip

echo "==> torch ${TORCH_VERSION} (cu128)"
pip install "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" --index-url https://download.pytorch.org/whl/cu128

echo "==> thư viện Python"
pip install -r "$REPO_ROOT/requirements.txt"

echo "==> LIBERO"
mkdir -p "$THIRD_PARTY"
if [ ! -d "$THIRD_PARTY/LIBERO" ]; then
  git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$THIRD_PARTY/LIBERO"
fi
# --no-deps: requirements gốc của LIBERO pin torch cũ, sẽ phá torch cu128
# Thư mục namespace libero/ thiếu __init__.py, khiến setuptools không tìm thấy package.
touch "$THIRD_PARTY/LIBERO/libero/__init__.py"
pip install -e "$THIRD_PARTY/LIBERO" --no-deps

# LIBERO hỏi đường dẫn bằng input() ở lần import đầu tiên -> tạo sẵn config để chạy non-interactive
LIBERO_PKG="$THIRD_PARTY/LIBERO/libero/libero"
LIBERO_CFG_DIR="${LIBERO_CONFIG_PATH:-$HOME/.libero}"
mkdir -p "$LIBERO_CFG_DIR"
if [ ! -f "$LIBERO_CFG_DIR/config.yaml" ]; then
  cat > "$LIBERO_CFG_DIR/config.yaml" <<EOF
benchmark_root: $LIBERO_PKG
bddl_files: $LIBERO_PKG/bddl_files
init_states: $LIBERO_PKG/init_files
datasets: $LIBERO_PKG/../datasets
assets: $LIBERO_PKG/assets
EOF
  echo "đã tạo $LIBERO_CFG_DIR/config.yaml"
fi

echo "==> kiểm tra"
cd "$REPO_ROOT"
python scripts/check_env.py
