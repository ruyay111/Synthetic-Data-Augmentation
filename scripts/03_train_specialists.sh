#!/usr/bin/env bash
# Stage 3: train one unconditional UniTST_MP diffusion specialist per volatility regime.
#
# Derived from train_specialist_diffusions.sh in the ruya tree. Uses the ten-asset regime windows from
# stage 2 (enc_in=10, full Corr loss). Absolute paths are required because run.py chdirs to its own
# directory. --skip_test because pools are sampled separately by scripts/04_generate_pools.py.
#
# Environment overrides:
#   PYTHON        interpreter to use                (default: python)
#   DEVICE        cuda | mps | cpu                  (default: cuda)
#   EPOCHS        training epochs per specialist    (default: from configs/default.yaml)
#   REGIMES       space-separated regimes to train  (default: all)
#   LOSS          loss spec                         (default: from configs/default.yaml)
#   FORCE         1 to retrain regimes that already have a checkpoint
#
# Smoke test one regime before committing to the full run:
#   REGIMES=0 EPOCHS=2 DEVICE=cuda ./scripts/03_train_specialists.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda}"
FORCE="${FORCE:-0}"

if ! command -v "${PYTHON}" >/dev/null 2>&1; then
  echo "[ERROR] python interpreter '${PYTHON}' not found. Set PYTHON=/path/to/python." >&2
  exit 1
fi

# Pull the shared constants out of the config so this script and the notebook cannot drift apart.
eval "$(
  "${PYTHON}" - "${REPO_ROOT}" <<'PY'
import pathlib, sys, yaml
cfg = yaml.safe_load((pathlib.Path(sys.argv[1]) / "configs" / "default.yaml").read_text())
d, r = cfg["diffusion"], cfg["regimes"]
print(f'CFG_N_REGIMES={r["n_regimes"]}')
print(f'CFG_SEQ_LEN={d["seq_len"]}')
print(f'CFG_ENC_IN={d["enc_in"]}')
print(f'CFG_BATCH_SIZE={d["batch_size"]}')
print(f'CFG_SAMPLE_MULTIPLIER={d["sample_multiplier"]}')
print(f'CFG_EPOCHS={d["train_epochs"]}')
print(f'CFG_LR={d["learning_rate"]}')
print(f'CFG_LR_DECAY={d["lr_decay_rounds"]}')
print(f'CFG_LOSS="{d["loss"]}"')
print(f'CFG_SCALE={d["scale"]}')
print(f'CFG_WINDOWS={cfg["paths"]["regime_windows"]}')
print(f'CFG_CHECKPOINTS={cfg["paths"]["checkpoints"]}')
print(f'CFG_TEST_RESULTS={cfg["paths"]["test_results"]}')
PY
)"

EPOCHS="${EPOCHS:-${CFG_EPOCHS}}"
LOSS="${LOSS:-${CFG_LOSS}}"
WINDOWS_DIR="${REPO_ROOT}/${CFG_WINDOWS}"
CHECKPOINTS_DIR="${REPO_ROOT}/${CFG_CHECKPOINTS}"
TEST_RESULTS_DIR="${REPO_ROOT}/${CFG_TEST_RESULTS}"
RUN_PY="${REPO_ROOT}/third_party/diffusion/run.py"

if [[ -n "${REGIMES:-}" ]]; then
  read -r -a REGIME_LIST <<< "${REGIMES}"
else
  read -r -a REGIME_LIST <<< "$(seq 0 $((CFG_N_REGIMES - 1)) | tr '\n' ' ')"
fi

mkdir -p "${CHECKPOINTS_DIR}" "${TEST_RESULTS_DIR}"

echo "[INFO] python=${PYTHON} device=${DEVICE} epochs=${EPOCHS}"
echo "[INFO] regimes=${REGIME_LIST[*]} loss=${LOSS}"
echo "[INFO] windows=${WINDOWS_DIR}"

for k in "${REGIME_LIST[@]}"; do
  data_path="${WINDOWS_DIR}/regime_${k}.npy"
  if [[ ! -f "${data_path}" ]]; then
    echo "[SKIP] regime ${k}: ${data_path} missing; run scripts/02_build_diffusion_dataset.py"
    continue
  fi

  n_windows="$("${PYTHON}" -c "import numpy as np; print(int(np.load(r'''${data_path}''').shape[0]))")"
  if [[ "${n_windows}" -lt 8 ]]; then
    echo "[SKIP] regime ${k}: only ${n_windows} windows"
    continue
  fi

  if [[ "${FORCE}" != "1" ]] && compgen -G "${CHECKPOINTS_DIR}/*_specialist_regime_${k}/checkpoint.pth" >/dev/null; then
    echo "[SKIP] regime ${k}: checkpoint exists (FORCE=1 to retrain)"
    continue
  fi

  echo "[TRAIN] regime ${k}: ${n_windows} windows"
  "${PYTHON}" "${RUN_PY}" \
    --task_name diffusion_denoised_x \
    --model UniTST_MP \
    --data RegimeWindows \
    --data_path "${data_path}" \
    --checkpoints "${CHECKPOINTS_DIR}/" \
    --test_results "${TEST_RESULTS_DIR}/" \
    --seq_len "${CFG_SEQ_LEN}" \
    --enc_in "${CFG_ENC_IN}" \
    --scale "${CFG_SCALE}" \
    --batch_size "${CFG_BATCH_SIZE}" \
    --sample_multiplier "${CFG_SAMPLE_MULTIPLIER}" \
    --train_epochs "${EPOCHS}" \
    --causal_mask \
    --ind_proj \
    --RoPE \
    --channel_embed \
    --learning_rate "${CFG_LR}" \
    --lr_decay_rounds "${CFG_LR_DECAY}" \
    --loss "${LOSS}" \
    --device "${DEVICE}" \
    --gpu 0 \
    --skip_test \
    --description "specialist_regime_${k}"
done

echo "[OK] done. Next: scripts/04_generate_pools.py"
