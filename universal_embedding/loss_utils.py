from typing import Iterable, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax import traverse_util


def kl_divergence(
    log_predictions,
    targets,
):
    """Computes KL divergence D(targets || predictions).

    Args:
        log_predictions: Log-probabilities of the predicted distribution with
            shape [..., dim].
        targets: Probabilities of the target distribution with shape [..., dim].

    Returns:
        KL divergence with shape [...].
    """
    loss = targets * (
        jnp.where(targets == 0, 0.0, jnp.log(targets)) - log_predictions
    )
    return jnp.sum(loss, axis=-1)


def _transform_logits(
    logits,
    one_hot,
    loss_type,
    loss_margin,
    loss_config,
    model_params,
    domain,
    classif_loss_on,
):
    """Applies ArcFace / CosFace / NormFace-style transformations to logits."""
    use_trainable_scale = (
        loss_config.trainable_scale and classif_loss_on != "domain_classifier"
    )

    if use_trainable_scale:
        learnable_scale = model_params["Embedding_Head"][
            f"{classif_loss_on}_logit_scale_{domain}"
        ]
        total_scale = loss_config.scale * learnable_scale
        base_logits = logits / total_scale
    else:
        total_scale = loss_config.scale
        base_logits = logits

    if loss_type == "arcface":
        clipped_logits = jnp.clip(base_logits, -1.0 + 1e-7, 1.0 - 1e-7)
        theta_yi = jnp.arccos(clipped_logits)
        transformed_target = jnp.cos(theta_yi + loss_margin)
        transformed_logits = transformed_target * one_hot + base_logits * (1 - one_hot)

    elif loss_type == "cosface":
        transformed_logits = (base_logits - loss_margin) * one_hot + base_logits * (1 - one_hot)

    elif loss_type == "normface":
        transformed_logits = base_logits

    else:
        raise ValueError(
            f"Unsupported loss_type '{loss_type}'. "
            "Expected one of: 'arcface', 'cosface', 'normface'."
        )

    return transformed_logits * total_scale


def logits_distillation_loss(
    teacher_logits,
    student_logits,
    config,
):
    """KL-based distillation loss between teacher and student logits."""
    if config.loss.logits_distill_stopgrad_teacher:
        teacher_logits = jax.lax.stop_gradient(teacher_logits)

    teacher_logits = teacher_logits / config.loss.logits_distill_temperature["teacher"]
    student_logits = student_logits / config.loss.logits_distill_temperature["student"]

    teacher_probs = jax.nn.softmax(teacher_logits, axis=-1)
    student_log_probs = jax.nn.log_softmax(student_logits, axis=-1)

    loss = jnp.mean(
        kl_divergence(
            student_log_probs,
            teacher_probs,
        )
    )
    return loss


def embedding_similarity_distillation_loss(
    teacher_embeddings,
    universal_student_embeddings,
    stopgrad_teacher: bool = False,
):
    """Matches pairwise similarity structure between teacher and student embeddings."""
    if stopgrad_teacher:
        teacher_embeddings = jax.lax.stop_gradient(teacher_embeddings)

    batch_simils_student = jnp.dot(
        universal_student_embeddings,
        universal_student_embeddings.T,
    )
    batch_simils_teacher = jnp.dot(
        teacher_embeddings,
        teacher_embeddings.T,
    )

    return jnp.mean((batch_simils_student - batch_simils_teacher) ** 2)


def embedding_distillation_loss(
    teacher_embeddings,
    universal_student_embeddings,
    stopgrad_teacher: bool = False,
):
    """Per-sample embedding MSE along the feature dimension."""
    if stopgrad_teacher:
        teacher_embeddings = jax.lax.stop_gradient(teacher_embeddings)

    return jnp.mean(
        (universal_student_embeddings - teacher_embeddings) ** 2,
        axis=-1,
    )


def remove_key_from_pytree(pytree, keys_to_remove: Iterable[str]):
    """Removes subtrees whose flattened path starts with one of the given prefixes."""
    flat_dict = traverse_util.flatten_dict(pytree, sep="/")
    filtered_flat_dict = {
        key: val
        for key, val in flat_dict.items()
        if not any(key.startswith(prefix) for prefix in keys_to_remove)
    }
    return traverse_util.unflatten_dict(filtered_flat_dict, sep="/")


def pretrained_weights_loss_fn(
    model_params,
    pretrained_params,
    keys_to_ignore: Optional[Iterable[str]] = None,
):
    """Computes parameter-space MSE regularization against pretrained weights."""
    if keys_to_ignore is None:
        keys_to_ignore = ("Embedding_Head",)

    model_params_filtered = remove_key_from_pytree(model_params, keys_to_ignore)
    pretrained_params_filtered = remove_key_from_pytree(pretrained_params, keys_to_ignore)

    assert (
        jax.tree_util.tree_structure(model_params_filtered)
        == jax.tree_util.tree_structure(pretrained_params_filtered)
    ), "Parameter structures do not match after filtering."

    squared_diffs_tree = jax.tree_util.tree_map(
        lambda p, pp: jnp.mean((p - pp) ** 2),
        model_params_filtered,
        pretrained_params_filtered,
    )

    return sum(jax.tree_util.tree_leaves(squared_diffs_tree))