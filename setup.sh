#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(cd "${REPO_ROOT}/.." && pwd)"
VENV_DIR="${REPO_ROOT}/scenic_venv"

clone_or_update_repo() {
  local repo_url="$1"
  local target_dir="$2"

  if [ -d "${target_dir}/.git" ]; then
    git -C "${target_dir}" pull --ff-only
  elif [ -d "${target_dir}" ]; then
    echo "Directory ${target_dir} already exists but is not a git repo; leaving it unchanged."
  else
    git clone "${repo_url}" "${target_dir}"
  fi
}

cd "${REPO_ROOT}"

sudo apt update -y
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update -y
sudo apt install -y python3.10 python3.10-venv

python3.10 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel

clone_or_update_repo "https://github.com/google-research/scenic.git" "${REPO_ROOT}/scenic"
python -m pip install "${REPO_ROOT}/scenic"
python -m pip uninstall -y jax jaxlib || true
python -m pip install "jax[tpu]" -f "https://storage.googleapis.com/jax-releases/libtpu_releases.html"
python -m pip install grain wandb ipdb scikit-learn tensorflow_text
rm -rf "${REPO_ROOT}/scenic"

clone_or_update_repo "https://github.com/google-research/big_vision.git" "${PARENT_DIR}/big_vision"
python -m pip install -r "${PARENT_DIR}/big_vision/requirements.txt"

clone_or_update_repo "https://github.com/google-deepmind/tips.git" "${PARENT_DIR}/tips"

echo
echo "Setup complete."
echo "Activate the environment with:"
echo "  source ${VENV_DIR}/bin/activate"
echo "Expected sibling layout:"
echo "  ${PARENT_DIR}/infusing"
echo "  ${PARENT_DIR}/big_vision"
echo "  ${PARENT_DIR}/tips"
