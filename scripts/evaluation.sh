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
finetuning_dataset="${2:?missing finetuning dataset}"
generic_dataset="${3:?missing generic dataset}"
pretraining="${4:?missing pretraining name}"
experiment_name="${5:?missing experiment name}"
testing_datasets="${6:?missing testing datasets}"

dataset="${finetuning_dataset},${generic_dataset}"
model='B'

experiment_name="${experiment_name}"
project_name=infusing_public_code
group_name="ft:${finetuning_dataset}__pretr:${pretraining}_ViT${model}_eval_topk20"

mount_dir=data
save_dir=data/exps

echo "Running experiment ${experiment_name} in project ${project_name} and group ${group_name}"
echo "Mount dir is ${mount_dir}"
echo "workdir is ${save_dir}/experiments/${project_name}/${group_name}/${experiment_name}_knn"

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

python -m universal_embedding.knn_main \
--config="universal_embedding/configs/config_knn_vit.py" \
--workdir="${save_dir}/experiments/${project_name}/${group_name}/${experiment_name}_knn" \
"${wandb_args[@]}" \
--config.use_grain_dataloader=True \
--config.grain_worker_count=6 \
--config.worker_buffer_size=4 \
--config.train_dir="${save_dir}/experiments/${project_name}/${group_name}/${experiment_name}" \
--config.eval_dataset_dir="${mount_dir}/array_records/" \
--config.train_dataset_dir="${mount_dir}/array_records/" \
--config.info_files_dir="${mount_dir}/info_files" \
--config.pretrained_ckpt_dir="${mount_dir}/models/" \
--config.text_datasets="${mount_dir}/text_image/flickr30k,${mount_dir}/text_image/mscoco" \
--config.model_class="${pretraining}_vit_with_embedding" \
--config.knn_eval_names="${testing_datasets}" \
--config.test_pretrained_features=False \
--config.save_descriptors=False \
--config.preextracted=False \
--config.extract_only_descrs=False \
--config.disabled_separate_knns='train_knn,val_knn' \
--config.disabled_merged_knns='train_knn,val_knn,test_knn' \
--config.embedd_to_eval="backbone_out_embedd" \
--config.universal_embedding_is="backbone_out_embedd" \
--config.do_image_eval=True \
--config.do_text_eval=True \
--config.top_k=20 


sleep 20
