"""Script for KNN evaluation."""

import functools
import logging

from absl import logging as absl_logging
from clu import metric_writers
from flax import jax_utils
import jax
import jax.numpy as jnp
import ml_collections

from scenic.train_lib import train_utils

from universal_embedding import app
from universal_embedding import grain_datasets
from universal_embedding import knn_utils
from universal_embedding import model_init
from universal_embedding import models
from universal_embedding import text_eval_utils
from universal_embedding import train_eval_steps
from universal_embedding import univ_train_state


def _restore_eval_checkpoint(train_state, train_dir: str, step: int):
    """Restores a checkpoint for evaluation and re-replicates the state."""
    restored_state, _ = train_utils.restore_checkpoint(
        train_dir,
        train_state,
        assert_exist=True,
        step=step,
    )
    restored_state = restored_state.replace(metadata={})
    return jax_utils.replicate(restored_state)


def _run_eval_for_step(
    *,
    step: int,
    train_state,
    knn_evaluator,
    config,
    train_dir: str,
    workdir: str,
    writer,
):
    """Runs image/text evaluation for a single checkpoint step."""
    absl_logging.info("Evaluating step %s.", step)

    if config.do_image_eval:
        knn_utils.knn_step(
            knn_evaluator,
            train_state,
            config,
            train_dir,
            step,
            writer,
            config.preextracted,
        )

    if config.get("do_text_eval"):
        text_eval_utils.get_text_results(
            knn_evaluator,
            train_state,
            config,
            workdir,
            step,
            writer,
        )


def knn_evaluate(
    rng: jnp.ndarray,
    config: ml_collections.ConfigDict,
    workdir: str,
    writer: metric_writers.MetricWriter,
) -> None:
    """Runs standalone KNN evaluation."""
    dataset_dict = grain_datasets.get_knn_eval_datasets(
        config,
        config.eval_dataset_dir,
        config.knn_eval_names,
        config.get("eval_batch_size", config.batch_size),
    )

    model_cls = models.MODELS[config.model_class]
    model = model_cls(config, dataset_dict.meta_data)

    rng, init_rng = jax.random.split(rng)

    if config.no_finetune:
        params, model_state, num_trainable_params, gflops = (
            model_init.initialize_universal_model_for_extraction(
                dataset_dict,
                config,
                model,
                init_rng,
            )
        )
    else:
            
        params, model_state, num_trainable_params, gflops = (
            model_init.initialize_universal_model(
                dataset_dict,
                config,
                model,
                init_rng,
                knn=True,
            )
        )
    del num_trainable_params, gflops

    train_state = univ_train_state.TrainState(
        params=params,
        model_state=model_state,
        pretrained_params=None,
    )

    # Important: this should handle both init_ckpt and pretrained_ckpt.
    train_state = model_init.load_init_checkpoint(
        config,
        train_state,
        model,
    )

    train_state = train_state.replace(metadata={})
    train_state = jax_utils.replicate(train_state)

    del params

    representation_fn_knn = functools.partial(
        train_eval_steps.representation_fn_eval,
        flax_model=model.flax_model,
        project_feats=config.project_feats_knn,
        config=config,
        domain_agnostic=config.no_finetune,
    )

    knn_eval_batch_size = config.get("knn_eval_batch_size") or config.batch_size

    knn_evaluator = knn_utils.KNNEvaluator(
        config,
        representation_fn_knn,
        knn_eval_batch_size,
        config.get("extract_only_descrs", False),
    )

    train_dir = config.get("train_dir")

    # Evaluate directly on pretrained/current state before checkpoint iteration.
    if config.test_pretrained_features:
        _run_eval_for_step(
            step=0,
            train_state=train_state,
            knn_evaluator=knn_evaluator,
            config=config,
            train_dir=train_dir,
            workdir=workdir,
            writer=writer,
        )

    if config.only_best_knn:
        steps_to_eval = [-1]
    else:
        steps_to_eval = [
            epoch * config.steps_per_epoch
            for epoch in range(config.knn_start_epoch, config.knn_end_epoch + 1)
        ]

    for step in steps_to_eval:
        if config.preextracted:
            eval_state = None
        else:
            eval_state = _restore_eval_checkpoint(train_state, train_dir, step)

        _run_eval_for_step(
            step=step,
            train_state=eval_state,
            knn_evaluator=knn_evaluator,
            config=config,
            train_dir=train_dir,
            workdir=workdir,
            writer=writer,
        )

    train_utils.barrier_across_hosts()


if __name__ == "__main__":
    app.run(main=knn_evaluate, knn=True)