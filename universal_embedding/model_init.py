import logging
from collections import OrderedDict, defaultdict
from typing import Iterable, Optional

import flax
from flax import jax_utils
import jax.numpy as jnp
import numpy as np

from scenic.train_lib import pretrain_utils
from scenic.train_lib import train_utils


def initialize_universal_model(
    dataset_dict,
    config,
    model,
    init_rng,
    knn: bool = False,
):
    """Initializes the multi-domain model."""
    dataset_names = [name.strip() for name in dataset_dict.meta_data["dataset_name"].split(",")]
    input_spec = OrderedDict()

    for i, _ in enumerate(dataset_names):
        input_spec_key = (
            ("domain", i),
            ("init", True),
            ("return_feats", knn),
            ("domain_agnostic", knn),
        )
        input_spec[input_spec_key] = [
            (
                dataset_dict.meta_data["input_shape"],
                dataset_dict.meta_data.get("input_dtype", jnp.float32),
            )
        ]

    params, model_state, num_trainable_params, gflops = (
        train_utils.initialize_multitask_model(
            model_def=model.flax_model,
            input_spec=input_spec,
            config=config,
            rngs=init_rng,
        )
    )

    return params, model_state, num_trainable_params, gflops


def initialize_universal_model_for_extraction(
    dataset_dict,
    config,
    model,
    init_rng,
):
    """Initializes the model in descriptor-extraction mode."""
    params, model_state, num_trainable_params, gflops = train_utils.initialize_model(
        model_def=model.flax_model,
        input_spec=[
            (
                dataset_dict.meta_data["input_shape"],
                dataset_dict.meta_data.get("input_dtype", jnp.float32),
            )
        ],
        config=config,
        rngs=init_rng,
        init=True,
        return_feats=True,
        domain=-1,
        domain_agnostic=True,
    )

    return params, model_state, num_trainable_params, gflops


def _filter_top_level_params(params, keys_to_keep: Optional[Iterable[str]]):
    """Keeps only selected top-level parameter collections."""
    if keys_to_keep is None:
        return params

    filtered = {
        key: params[key]
        for key in params
        if key in set(keys_to_keep)
    }
    return flax.core.freeze(filtered)


def _merge_params_by_shape(current_tree, restored_tree):
    """Recursively keeps restored params only when shapes match."""
    if isinstance(current_tree, (dict, flax.core.FrozenDict)):
        merged = {}
        for key in current_tree:
            if key in restored_tree:
                merged[key] = _merge_params_by_shape(current_tree[key], restored_tree[key])
            else:
                merged[key] = current_tree[key]
        return type(current_tree)(merged) if isinstance(current_tree, dict) else flax.core.freeze(merged)

    current_arr = np.asarray(current_tree)
    restored_arr = np.asarray(restored_tree)

    if current_arr.shape == restored_arr.shape:
        return restored_tree
    return current_tree


def load_init_checkpoint(
    config,
    train_state,
    model,
):
    """Loads either an init checkpoint or a pretrained backbone checkpoint."""
    if config.init_ckpt:
        logging.info("Initializing from ckpt %s.", config.init_ckpt)

        ckpt_dir, ckpt_name = config.init_ckpt.rsplit("/", 1)
        ckpt_num = ckpt_name.split("_")[-1]

        restored_train_state = pretrain_utils.restore_pretrained_checkpoint(
            ckpt_dir,
            train_state,
            assert_exist=True,
            step=ckpt_num,
        )

        keys_to_keep = None
        if config.restore_only_backbone and config.keys_to_restore is None:
            keys_to_keep = ["Transformer", "cls", "embedding"]
        elif config.keys_to_restore is not None:
            keys_to_keep = config.keys_to_restore

        restored_params = _filter_top_level_params(
            restored_train_state.params,
            keys_to_keep,
        )

        merged_params = _merge_params_by_shape(train_state.params, restored_params)
        restored_train_state = restored_train_state.replace(params=merged_params)

        train_state = pretrain_utils.init_from_pretrain_state(
            train_state,
            restored_train_state,
        )
        return train_state

    if config.pretrained_ckpt and not config.start_from_scratch:
        logging.info("Initializing from ckpt %s.", config.pretrained_ckpt)

        if "siglip_vit_with_embedding" in config.model_class:
            train_state = model.load_siglip_params(
                train_state,
                config.pretrained_ckpt,
                config.model,
                config.params_early_train,
            )
        elif "tips_vit_with_embedding" in config.model_class:
            train_state = model.load_tips_params(
                train_state,
                config.pretrained_ckpt,
            )

    return train_state


def _safe_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm