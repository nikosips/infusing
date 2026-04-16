import ast
import json
import os

import jax
import ml_collections
import numpy as np
from flax.training import checkpoints
from tensorflow.io import gfile

from universal_embedding import info_utils


def save_descriptors(descr_save_path, all_descriptors_dict):
    """Saves descriptor data as JSON."""
    parent_dir = os.path.dirname(descr_save_path)
    if parent_dir:
        gfile.makedirs(parent_dir)

    with gfile.GFile(descr_save_path, "w") as f:
        json.dump(all_descriptors_dict, f, cls=NumpyEncoder)

    print(f"Descriptors file complete: {descr_save_path}")


def _get_model_configs(config):
    """Returns the model config table for the selected model family."""
    model_class = config.model_class.lower()

    if "siglip" in model_class:
        return info_utils.SigLIP_ViT_configs
    if "tips" in model_class:
        return info_utils.TIPS_ViT_configs

    raise ValueError(
        f"Unsupported model_class '{config.model_class}'. "
        "Expected a model class containing 'siglip' or 'tips'."
    )


def _safe_step_interval(steps_per_epoch, frequency, field_name):
    """Converts a per-epoch frequency to a positive step interval."""
    if frequency <= 0:
        raise ValueError(f"{field_name} must be > 0, got {frequency}.")
    return max(1, steps_per_epoch // frequency)


def _parse_literal_config_field(config_section, field_name):
    """Parses a string field in a config section using ast.literal_eval."""
    string_field = f"{field_name}_string"
    setattr(
        config_section,
        field_name,
        ast.literal_eval(getattr(config_section, string_field)),
    )


def calc_train_dependent_config_values(config, knn=False):
    """Fills config values that depend on model choice and training setup."""
    model_configs = _get_model_configs(config)

    if config.model_type not in model_configs:
        raise ValueError(
            f"Unknown model_type '{config.model_type}'. "
            f"Available types: {list(model_configs.keys())}"
        )

    model_cfg = model_configs[config.model_type]

    # Model architecture parameters.
    config.model.hidden_size = model_cfg["hidden_size"]
    config.model.patches = ml_collections.ConfigDict()
    config.model.patches.size = model_cfg["patches_size"]
    config.model.num_heads = model_cfg["num_heads"]
    config.model.mlp_dim = model_cfg["mlp_dim"]
    config.model.num_layers = model_cfg["num_layers"]

    # Pretrained checkpoint path.
    config.pretrained_ckpt = os.path.join(
        config.pretrained_ckpt_dir,
        model_cfg["checkpoint"],
    )

    if knn:
        return

    # Dataset/training schedule values.
    config.steps_per_epoch = (
        info_utils.get_aggregated_size(config.dataset_name) // config.batch_size
    )

    total_steps = config.num_training_epochs * config.steps_per_epoch
    config.lr_configs.steps_per_cycle = total_steps

    if config.log_eval_steps == -1:
        config.log_eval_steps = _safe_step_interval(
            config.steps_per_epoch,
            config.log_eval_steps_frequency,
            "log_eval_steps_frequency",
        )

    config.log_summary_steps = _safe_step_interval(
        config.steps_per_epoch,
        config.log_summary_steps_frequency,
        "log_summary_steps_frequency",
    )

    config.checkpoint_steps = _safe_step_interval(
        config.steps_per_epoch,
        config.checkpoint_steps_frequency,
        "checkpoint_steps_frequency",
    )

    # Optimizer / freezing schedule.
    if config.frozen_epochs == -1:
        config.lr_configs.backbone.frozen_steps = total_steps
    else:
        config.lr_configs.backbone.frozen_steps = (
            config.frozen_epochs * config.steps_per_epoch
        )

    config.lr_configs.backbone.base_learning_rate = (
        config.lr_configs.base_learning_rate
        * config.backbone_learning_rate_multiplier
    )

    # Parse classifier-loss config for embedding-head model variants.
    if "vit_with_embedding" in config.model_class.lower():
        _parse_literal_config_field(config.loss, "classif_losses_on")
        _parse_literal_config_field(config.loss, "classif_losses_weights")
        _parse_literal_config_field(config.loss, "classif_losses_types")
        _parse_literal_config_field(config.loss, "classif_losses_margins")
        _parse_literal_config_field(config.loss, "stopgrad_on_classifier_on")


def save_best_checkpoint(workdir, train_state):
    """Saves a checkpoint at step -1, overwriting any existing best checkpoint."""
    if jax.process_index() != 0:
        return

    checkpoint_state = jax.device_get(train_state)
    checkpoints.save_checkpoint(
        workdir,
        checkpoint_state,
        step=-1,
        overwrite=True,
    )


def read_config(path):
    """Reads a JSON config file into an ml_collections.ConfigDict."""
    with gfile.GFile(path, "r") as f:
        config_dict = json.load(f)
    return ml_collections.ConfigDict(config_dict)


def normalize(a, axis=-1, order=2):
    """L2-normalizes an array along the given axis by default."""
    a = np.asarray(a)
    norms = np.linalg.norm(a, ord=order, axis=axis, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return a / norms


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that converts numpy types to Python-native types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
