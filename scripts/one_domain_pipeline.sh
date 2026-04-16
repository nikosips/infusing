#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PARENT_DIR="$(cd "${ROOT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PARENT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

wandb_entity="${1:-}"
finetuning_dataset="${2:?missing finetuning dataset}"
generic_dataset="${3:?missing generic dataset}"
pretraining="${4:?missing pretraining name}"
epochs="${5:?missing epoch count}"
testing_datasets="${6:?missing testing datasets}"
sampling_strategy="${7:?missing sampling strategy}"
log_eval_steps="${8:?missing log_eval_steps}"
pretrained_embedd_weight="${9:?missing pretrained embedding loss weight}"
pretrained_weight_loss_weight="${10:?missing pretrained weight loss weight}"

descriptor_datasets="${finetuning_dataset},${generic_dataset}"
experiment_name="classif_backbone_out_pretrained_embedd_loss_on_${generic_dataset}_weight_${pretrained_embedd_weight}_pretrained_weights_loss_weight_${pretrained_weight_loss_weight}"

echo "Step 1/3: extracting pretrained train descriptors for ${descriptor_datasets}"
bash "${ROOT_DIR}/scripts/extract_pretrained_feats.sh" \
  "${wandb_entity}" \
  "${pretraining}" \
  "${descriptor_datasets}"

echo "Step 2/3: training full method for ${finetuning_dataset} with generic dataset ${generic_dataset}"
bash "${ROOT_DIR}/scripts/full_method.sh" \
  "${wandb_entity}" \
  "${finetuning_dataset}" \
  "${generic_dataset}" \
  "${pretraining}" \
  "${epochs}" \
  "${testing_datasets}" \
  "${sampling_strategy}" \
  "${log_eval_steps}" \
  "${pretrained_embedd_weight}" \
  "${pretrained_weight_loss_weight}"

echo "Step 3/3: running test-set kNN evaluation for ${experiment_name}"
bash "${ROOT_DIR}/scripts/evaluation.sh" \
  "${wandb_entity}" \
  "${finetuning_dataset}" \
  "${generic_dataset}" \
  "${pretraining}" \
  "${experiment_name}" \
  "${testing_datasets}"
