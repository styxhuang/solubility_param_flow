#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/root/software/uni-elf"
ENV_PREFIX="${PROJECT_ROOT}/.conda-env"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
HTTP_PROXY_VALUE="${HTTP_PROXY:-http://ga.dp.tech:8118}"
HTTPS_PROXY_VALUE="${HTTPS_PROXY:-http://ga.dp.tech:8118}"

export HTTP_PROXY="${HTTP_PROXY_VALUE}"
export HTTPS_PROXY="${HTTPS_PROXY_VALUE}"

source "$(conda info --base)/etc/profile.d/conda.sh"

if [[ ! -d "${ENV_PREFIX}" ]]; then
  conda create -y -p "${ENV_PREFIX}" "python=${PYTHON_VERSION}"
fi

conda activate "${ENV_PREFIX}"

python -m pip install --upgrade pip
python -m pip install --upgrade setuptools wheel
python -m pip install torch joblib rdkit pyyaml addict tqdm matplotlib huggingface_hub seaborn
python -m pip install numpy==1.22.4 pandas==1.4.0 scikit-learn==1.5.0
python -m pip install unimol_tools

python -m pip install -U --no-build-isolation "${PROJECT_ROOT}/lib/unicore"
python -m pip install -U --no-build-isolation --no-deps "${PROJECT_ROOT}"

echo "uni-elf environment ready at ${ENV_PREFIX}"
