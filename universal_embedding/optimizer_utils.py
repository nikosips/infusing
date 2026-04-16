import flax
import jax
import jax.numpy as jnp
import optax

from scenic.train_lib import optimizers


def backbone_lr(config, full_config=None):
    """Returns the learning-rate function for backbone parameters.

    The backbone learning rate is zero while the backbone is frozen, then
    switches to `base_learning_rate` once `step >= frozen_steps`.

    Args:
        config: Backbone LR config. Must contain:
            - base_learning_rate
            - frozen_steps
        full_config: Unused, kept for interface compatibility.

    Returns:
        A function mapping global step -> learning rate.
    """
    del full_config

    def lr_fn(step):
        learning_rate = config["base_learning_rate"]
        is_unfrozen = jnp.where(step < config.frozen_steps, 0.0, 1.0)
        return learning_rate * is_unfrozen

    return lr_fn


def get_multioptimizer(
    optimizer_config,
    classifier_lr_fn,
    backbone_lr_fn,
    params,
    config,
):
    """Builds an Optax optimizer with separate masks for classifier/backbone params.

    Parameter groups:
    - classifier params: matched by `config.params_early_train`
    - backbone params: everything not in `params_early_train`

    Args:
        optimizer_config: Scenic optimizer config.
        classifier_lr_fn: LR schedule for classifier / early-train params.
        backbone_lr_fn: LR schedule for backbone / late-train params.
        params: Parameter pytree.
        config: Experiment config containing model_class and param group lists.

    Returns:
        An Optax gradient transformation combining masked optimizers.
    """
    classifier_optim = optimizers.get_optimizer(
        optimizer_config,
        classifier_lr_fn,
        params,
    )
    backbone_optim = optimizers.get_optimizer(
        optimizer_config,
        backbone_lr_fn,
        params,
    )

    false_tree = jax.tree_util.tree_map(lambda _: False, params)

    classifier_traversal = flax.traverse_util.ModelParamTraversal(
        lambda path, _: any(name in path for name in config.params_early_train)
    )

    backbone_traversal = flax.traverse_util.ModelParamTraversal(
        lambda path, _: all(name not in path for name in config.params_early_train)
    )
    unchanged_optim = []

    classifier_mask = classifier_traversal.update(lambda _: True, false_tree)
    backbone_mask = backbone_traversal.update(lambda _: True, false_tree)

    return optax.chain(
        optax.masked(backbone_optim, backbone_mask),
        optax.masked(classifier_optim, classifier_mask),
        *unchanged_optim,
    )