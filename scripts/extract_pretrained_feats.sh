#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PARENT_DIR="$(cd "${ROOT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PARENT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

if [ ! -d "${PARENT_DIR}/big_vision" ]; then
  echo "Missing dependency checkout: ${PARENT_DIR}/big_vision" >&2
  echo "Clone it next to this repo or run setup.sh on this machine." >&2
  exit 1
fi

if [ ! -d "${PARENT_DIR}/tips" ]; then
  echo "Missing dependency checkout: ${PARENT_DIR}/tips" >&2
  echo "Clone it next to this repo or run setup.sh on this machine." >&2
  exit 1
fi

wandb_entity="${1:-}"
pretraining="${2:?missing pretraining name}"
descriptor_datasets="${3:?missing descriptor datasets}"

model='B'



experiment_name=${pretraining}_vit${model}_pretrained_embeddings
group_name=features
project_name=off-the-shelf

mount_dir=data
save_dir=data/exps
workdir="${save_dir}/experiments/${project_name}/${group_name}/${experiment_name}"

wandb_args=()
if [ -n "${wandb_entity}" ]; then
  wandb_args=(
    --wandb_entity "${wandb_entity}"
    --wandb_project "${project_name}"
    --wandb_group "${group_name}"
    --wandb_name "${experiment_name}"
    --use_wandb
  )
fi

echo "Extracting pretrained descriptors for ${descriptor_datasets}"
echo "workdir is ${workdir}"

python -m universal_embedding.knn_main \
--config="universal_embedding/configs/config_knn_vit_no_finetune.py" \
--workdir="${workdir}" \
"${wandb_args[@]}" \
--config.use_grain_dataloader=True \
--config.grain_worker_count=6 \
--config.worker_buffer_size=4 \
--config.train_dir="" \
--config.pretrained_ckpt_dir="${mount_dir}/models/" \
--config.eval_dataset_dir="${mount_dir}/array_records/" \
--config.train_dataset_dir="${mount_dir}/array_records/" \
--config.info_files_dir="${mount_dir}/info_files" \
--config.model_class="${pretraining}_vit_with_embedding" \
--config.model_type="${model}/16" \
--config.knn_eval_names="${descriptor_datasets}" \
--config.extract_only_descrs=True \
--config.test_pretrained_features=True \
--config.save_descriptors=True \
--config.preextracted=False \
--config.descr_save_path="${workdir}" \
--config.disabled_separate_knns='test_knn' \
--config.disabled_merged_knns='train_knn,val_knn,test_knn' \
--config.embedd_to_eval="backbone_out_embedd" \
--config.universal_embedding_is="backbone_out_embedd" \
--config.do_image_eval=True \
--config.do_text_eval=False \
--config.top_k=20
