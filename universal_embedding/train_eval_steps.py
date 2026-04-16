from typing import Any, Callable, Dict, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax
from jax.example_libraries.optimizers import clip_grads

from scenic.train_lib import train_utils


# Aliases for custom types.
Batch = Dict[str, jnp.ndarray]
Metrics = Dict[str, Tuple[float, int]]
TrainingLogs = Dict[str, Any]

MetricFn = Callable[..., Metrics]
LossFn = Callable[..., jnp.ndarray]
LrFn = Callable[[jnp.ndarray], jnp.ndarray]


def train_step(
    train_state: train_utils.TrainState,
    batch: Batch,
    batch_domain_idx: int,  # Always clean batches.
    *,
    flax_model: nn.Module,
    loss_fn: LossFn,
    classifier_lr_fn: LrFn,
    backbone_lr_fn: LrFn,
    metrics_fn: MetricFn,
    config: ml_collections.ConfigDict,
    debug: bool = False,
) -> Tuple[train_utils.TrainState, Metrics, TrainingLogs]:
    """Runs a single training step.

    Given the current training state and a batch, computes the forward pass,
    loss, gradients, optimizer update, metrics, and logging values.

    Args:
        train_state: Current training state.
        batch: A single training batch.
        batch_domain_idx: Domain index for the current batch.
        flax_model: Flax model to train.
        loss_fn: Loss function.
        classifier_lr_fn: Learning-rate schedule for classifier parameters.
        backbone_lr_fn: Learning-rate schedule for backbone parameters.
        metrics_fn: Metrics function.
        config: Experiment configuration.
        debug: Whether to enable model-specific debug behavior.

    Returns:
        Updated training state, computed metrics, and training logs.
    """
    training_logs = {}

    new_rng, rng = jax.random.split(train_state.rng)

    # Bind RNG to the current host/device.
    dropout_rng = train_utils.bind_rng_to_host_device(
        rng,
        axis_name="batch",
        bind_to="device",
    )

    def training_loss_fn(params):
        variables = {"params": params, **train_state.model_state}

        outputs, new_model_state = flax_model.apply(
            variables,
            batch["inputs"],
            domain=batch_domain_idx,
            mutable=["batch_stats"],
            train=True,
            rngs={"dropout": dropout_rng},
            debug=debug,
        )

        loss = loss_fn(
            outputs,
            batch,
            batch_domain_idx,
            model_params=variables["params"],
            pretrained_params=train_state.pretrained_params,
        )

        return loss, (new_model_state, outputs)

    compute_gradient_fn = jax.value_and_grad(training_loss_fn, has_aux=True)
    (loss, (new_model_state, outputs)), grad = compute_gradient_fn(train_state.params)

    # Average gradients across devices. This axis name must match the surrounding pmap.
    grad = jax.lax.pmean(grad, axis_name="batch")

    if config.get("max_grad_norm") is not None:
        grad = clip_grads(grad, config.max_grad_norm)

    updates, new_opt_state = train_state.tx.update(
        grad,
        train_state.opt_state,
        train_state.params,
    )
    new_params = optax.apply_updates(train_state.params, updates)

    training_logs["loss"] = loss
    training_logs["l2_grads"] = jnp.sqrt(
        sum(jnp.vdot(g, g) for g in jax.tree_util.tree_leaves(grad))
    )
    training_logs["l2_params"] = jnp.sqrt(
        sum(jnp.vdot(p, p) for p in jax.tree_util.tree_leaves(new_params))
    )
    training_logs["l2_updates"] = jnp.sqrt(
        sum(jnp.vdot(u, u) for u in jax.tree_util.tree_leaves(updates))
    )

    training_logs["classifier_lr"] = classifier_lr_fn(train_state.global_step)
    training_logs["backbone_lr"] = backbone_lr_fn(train_state.global_step)

    # Metrics are computed for the same forward pass that produced `outputs`,
    # so using the pre-update params here is consistent.
    metrics = metrics_fn(
        outputs,
        batch,
        batch_domain_idx,
        model_params=train_state.params,
        pretrained_params=train_state.pretrained_params,
    )

    new_train_state = train_state.replace(
        global_step=train_state.global_step + 1,
        opt_state=new_opt_state,
        params=new_params,
        model_state=new_model_state,
        rng=new_rng,
    )

    return new_train_state, metrics, training_logs


def representation_fn_eval(
    train_state: train_utils.TrainState,
    batch: Batch,
    domain_idx: int,
    *,
    flax_model: nn.Module,
    project_feats: bool = True,
    gather_to_host: bool = True,
    domain_agnostic: bool = False,
    config: ml_collections.ConfigDict,
) -> Tuple[Dict[str, jnp.ndarray], Batch]:
    """Feeds inputs to the model and returns their embeddings and batch data.

    Args:
        train_state: Current training state.
        batch: A single evaluation batch.
        domain_idx: Domain index for this batch.
        flax_model: Flax model.
        project_feats: Whether to return projected features.
        gather_to_host: Whether to gather outputs across devices/hosts.
        domain_agnostic: Whether to use domain-agnostic KNN.
        config: Experiment configuration.

    Returns:
        A dictionary of embeddings and the corresponding batch.
    """
    variables = {"params": train_state.params, **train_state.model_state}
    domain_agnostic_knn = config.domain_agnostic_knn

    outputs = flax_model.apply(
        variables,
        batch["inputs"],
        domain=domain_idx if not domain_agnostic_knn else -1,
        train=False,
        return_feats=True,
        debug=False,
        project_feats=project_feats,
        domain_agnostic=domain_agnostic_knn,
    )

    embeddings_dict = outputs["embeddings"]

    if gather_to_host:
        for embed_type in embeddings_dict:
            embeddings_dict[embed_type] = jax.lax.all_gather(
                embeddings_dict[embed_type],
                "batch",
            )

        batch = jax.lax.all_gather(batch, "batch")

    return embeddings_dict, batch