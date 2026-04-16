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
epochs="${5:?missing epoch count}"
testing_datasets="${6:?missing testing datasets}"
sampling_strategy="${7:?missing sampling strategy}"
log_eval_steps="${8:?missing log_eval_steps}"

dataset="${finetuning_dataset},${generic_dataset}"
model='B'

experiment_name=classif_backbone_out
# experiment_name=classif_backbone_out_debug
project_name=infusing_public_code
group_name="ft:${finetuning_dataset}__pretr:${pretraining}_ViT${model}_eval_topk20"

mount_dir=data
save_dir=data/exps

echo "Running experiment ${experiment_name} in project ${project_name} and group ${group_name}"
echo "Mount dir is ${mount_dir}"
echo "workdir is ${save_dir}/experiments/${project_name}/${group_name}/${experiment_name}"

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

python -m universal_embedding.main \
--config=universal_embedding/configs/config_train_vit.py \
--workdir="${save_dir}/experiments/${project_name}/${group_name}/${experiment_name}" \
"${wandb_args[@]}" \
--config.use_grain_dataloader=True \
--config.grain_worker_count=6 \
--config.worker_buffer_size=4 \
--config.dataset_type="one_sample" \
--config.sampling_strategy="${sampling_strategy}" \
--config.update_sampler=False \
--config.update_sampler_mode="train_loss" \
--config.update_sampler_logit_type="backbone_out" \
--config.do_ema_on_sampler=False \
--config.pretrained_train_descriptors_dir="${mount_dir}/exps/experiments/off-the-shelf/${pretraining}_vitb/pretrained_embeddings_knn/descriptors/0/backbone_out_embedd" \
--config.pretrained_ckpt_dir="${mount_dir}/models/" \
--config.eval_dataset_dir="${mount_dir}/array_records/" \
--config.train_dataset_dir="${mount_dir}/array_records/" \
--config.info_files_dir="${mount_dir}/info_files" \
--config.text_datasets="${mount_dir}/text_image/flickr30k,${mount_dir}/text_image/mscoco" \
--config.model_class="${pretraining}_vit_with_embedding" \
--config.model_type="${model}/16" \
--config.classifier="separate" \
--config.num_training_epochs="${epochs}" \
--config.frozen_epochs=0 \
--config.log_eval_steps_frequency=1 \
--config.log_eval_steps=${log_eval_steps} \
--config.log_summary_steps_frequency=50 \
--config.dataset_name=${dataset} \
--config.knn_eval_names=${dataset} \
--config.disabled_separate_knns='train_knn,test_knn' \
--config.disabled_merged_knns='train_knn,val_knn,test_knn' \
--config.embedd_to_eval="backbone_out_embedd" \
--config.universal_embedding_is="backbone_out_embedd" \
--config.loss.classif_losses_on_string="{'0':'backbone_out','1':'','2':'','3':'','4':'','5':'','6':'','7':'','8':''}" \
--config.loss.classif_losses_weights_string="{'0':'1.0','1':'','2':'1.0,1.0','3':'1.0,1.0','4':'1.0,1.0','5':'1.0,1.0','6':'1.0,1.0','7':'1.0,1.0','8':'1.0,1.0'}" \
--config.loss.classif_losses_types_string="{'0':'normface','1':'','2':'normface,normface','3':'normface,normface','4':'normface,normface','5':'normface,normface','6':'normface,normface','7':'normface,normface','8':'normface,normface'}" \
--config.loss.classif_losses_margins_string="{'0':'0.0','1':'','2':'0.0,0.0','3':'0.0,0.0','4':'0.0,0.0','5':'0.0,0.0','6':'0.0,0.0','7':'0.0,0.0','8':'0.0,0.0'}" \
--config.loss.scale=16 \
--config.loss.trainable_scale=False \
--config.loss.aggregation_type="weighted_sum" \
--config.batch_size=128 \
--config.model.encoder_mlp_dim="(64,)" \
--config.loss.pretrained_embedding_distill_loss_on="" \
--config.loss.pretrained_embedding_distill_loss_weight=1000.0 \
--config.loss.pretrained_weights_loss=False \
--config.loss.pretrained_weights_loss_weight=100000 \
--config.checkpoint=True \
--config.top_k=20 \
--config.domain_agnostic_knn=True \
--config.do_knn=True \
--config.do_knn_at_start=True \
--config.do_text_eval_during_validation=True \
--config.do_text_eval=True \
--config.best_val_knn_on="in-domain" \
--config.knn_eval_names_final="${testing_datasets}"


sleep 20
