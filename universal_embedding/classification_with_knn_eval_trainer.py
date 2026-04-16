"""Training script and k-NN evaluation."""

import collections
import functools
from typing import Any, Dict, Tuple

from absl import logging
from clu import metric_writers
from clu import periodic_actions
from clu import platform
from flax import jax_utils
import jax
import jax.numpy as jnp
import jax.profiler
import ml_collections

from scenic.train_lib import lr_schedules
from scenic.train_lib import optimizers
from scenic.train_lib import train_utils

from universal_embedding import grain_datasets
from universal_embedding import knn_utils
from universal_embedding import logging_utils
from universal_embedding import model_init
from universal_embedding import optimizer_utils
from universal_embedding import sampling_utils
from universal_embedding import text_eval_utils
from universal_embedding import train_eval_steps
from universal_embedding import univ_train_state
from universal_embedding import utils

Batch = Dict[str, jnp.ndarray]


def _write_note(note: str, lead_host: bool) -> None:
    if lead_host:
        platform.work_unit().set_notes(note)


def _save_checkpoint(
    train_state,
    workdir: str,
    chrono,
    lead_host: bool,
    *,
    max_to_keep: int = 10,
) -> None:
    """Saves a standard training checkpoint."""
    train_state = train_utils.sync_model_state_across_replicas(train_state)

    if lead_host:
        unrep_train_state = jax_utils.unreplicate(train_state)
        metadata = dict(unrep_train_state.metadata)
        metadata["chrono"] = chrono.save()
        unrep_train_state = unrep_train_state.replace(metadata=metadata)
        train_utils.save_checkpoint(
            workdir,
            unrep_train_state,
            max_to_keep=max_to_keep,
            overwrite=True,
        )


def _save_best_checkpoint(
    train_state,
    workdir: str,
    chrono,
    lead_host: bool,
) -> None:
    """Saves the current state as the best checkpoint."""
    train_state = train_utils.sync_model_state_across_replicas(train_state)

    if lead_host:
        unrep_train_state = jax_utils.unreplicate(train_state)
        metadata = dict(unrep_train_state.metadata)
        metadata["chrono"] = chrono.save()
        unrep_train_state = unrep_train_state.replace(metadata=metadata)
        utils.save_best_checkpoint(workdir, unrep_train_state)


def _get_best_val_metric(results, config) -> float:
    """Returns the validation metric used for best-checkpoint selection."""
    measure_on = config.best_val_knn_on
    dataset_names = [name.strip() for name in config.dataset_name.split(",")]

    if measure_on == "in-domain":
        key = f"{dataset_names[0]}:separate:val_knn:map_20"
    elif measure_on == "out-of-domain":
        if len(dataset_names) < 2:
            raise ValueError(
                "best_val_knn_on='out-of-domain' requires at least two dataset names."
            )
        key = f"{dataset_names[1]}:separate:val_knn:map_20"
    elif measure_on == "all":
        key = "average:separate:val_knn:map_20"
    else:
        raise ValueError(
            f"Unsupported config.best_val_knn_on='{measure_on}'. "
            "Expected one of: 'in-domain', 'out-of-domain', 'all'."
        )

    return results["map_results"]["universal"][key]


def train(
    *,
    rng: jnp.ndarray,
    config: ml_collections.ConfigDict,
    model_cls: Any,
    dataset_dict: Dict,
    workdir: str,
    writer: metric_writers.MetricWriter,
) -> Tuple[train_utils.TrainState, Dict[str, Any], Dict[str, Any]]:
    """Runs the full training loop and periodic k-NN evaluation.

    Args:
        rng: JAX RNG key.
        config: Experiment configuration.
        model_cls: Model class with flax_module, loss function, and metrics.
        dataset_dict: Dataset container with metadata and iterators.
        workdir: Directory for checkpointing and outputs.
        writer: CLU metrics writer.

    Returns:
        Final train state, latest train summary, and latest eval summary.
    """
    lead_host = jax.process_index() == 0

    model = model_cls(config, dataset_dict.meta_data)

    # Initialize model parameters and state.
    rng, init_rng = jax.random.split(rng)
    params, model_state, num_trainable_params, gflops = (
        model_init.initialize_universal_model(
            dataset_dict,
            config,
            model,
            init_rng,
        )
    )

    # Create optimizer and schedules.
    classifier_lr_fn = lr_schedules.get_learning_rate_fn(config)
    backbone_lr_fn = optimizer_utils.backbone_lr(config.lr_configs.backbone, config)
    optimizer_config = optimizers.get_optax_optimizer_config(config)

    tx = optimizer_utils.get_multioptimizer(
        optimizer_config,
        classifier_lr_fn,
        backbone_lr_fn,
        params=params,
        config=config,
    )

    # Initialize optimizer state on CPU.
    opt_state = jax.jit(tx.init, backend="cpu")(params)

    rng, train_rng = jax.random.split(rng)
    chrono = train_utils.Chrono(warmup=1)

    train_state = univ_train_state.TrainState(
        global_step=0,
        opt_state=opt_state,
        tx=tx,
        params=params,
        pretrained_params=None,
        model_state=model_state,
        rng=train_rng,
        metadata={"chrono": chrono.save()},
    )

    start_step = train_state.global_step

    train_state = model_init.load_init_checkpoint(
        config,
        train_state,
        model,
    )

    train_state = train_state.replace(pretrained_params=train_state.params)

    chrono.load(train_state.metadata["chrono"])
    train_state = train_state.replace(metadata={})

    # Replicate state across devices.
    train_state = jax_utils.replicate(train_state)
    del params

    total_steps, steps_per_epoch = train_utils.get_num_training_steps(
        config,
        dataset_dict.meta_data,
    )

    # Initialize sampling scheme.
    rng, sampler_rng = jax.random.split(rng)
    sampler = sampling_utils.Sampler(
        config,
        dataset_dict,
        total_steps,
        sampler_rng,
    )

    assert len(sampler.dataset_indices_per_step) == total_steps

    sample_weights_log = {
        f"sampling_weights/{dataset_name}": weight
        for dataset_name, weight in sampler.sampling_weights.items()
    }
    writer.write_scalars(0, sample_weights_log)

    train_step_pmapped = jax.pmap(
        functools.partial(
            train_eval_steps.udon_train_step
            if "udon" in config.model_class
            else train_eval_steps.train_step,
            flax_model=model.flax_model,
            loss_fn=model.loss_function,
            classifier_lr_fn=classifier_lr_fn,
            backbone_lr_fn=backbone_lr_fn,
            metrics_fn=model.get_metrics_fn("train"),
            config=config,
            debug=config.debug_train,
        ),
        axis_name="batch",
        donate_argnums=(0, 1),
        static_broadcasted_argnums=(2,),
    )

    representation_fn_knn = functools.partial(
        train_eval_steps.representation_fn_eval,
        flax_model=model.flax_model,
        project_feats=config.project_feats_knn,
        config=config,
    )

    knn_eval_batch_size = config.get("knn_eval_batch_size") or config.batch_size

    knn_evaluator = knn_utils.KNNEvaluator(
        config,
        representation_fn_knn,
        knn_eval_batch_size,
        config.get("extract_only_descrs", False),
    )

    log_eval_steps = config.get("log_eval_steps") or steps_per_epoch
    if not log_eval_steps:
        raise ValueError("'log_eval_steps' should be specified in the config.")

    checkpoint_steps = config.get("checkpoint_steps") or log_eval_steps
    log_summary_steps = config.get("log_summary_steps") or log_eval_steps

    train_metrics = collections.defaultdict(list)
    extra_training_logs = collections.defaultdict(list)

    train_summary, eval_summary = None, None

    chrono.inform(start_step, total_steps, config.batch_size, steps_per_epoch)
    logging.info("Starting training loop at step %d.", start_step + 1)

    report_progress = periodic_actions.ReportProgress(
        num_train_steps=total_steps,
        writer=writer,
    )

    hooks = [report_progress] if lead_host else []

    if start_step == 0:
        step0_log = {"num_trainable_params": num_trainable_params}
        if gflops:
            step0_log["gflops"] = gflops
        writer.write_scalars(1, step0_log)

    _write_note(f"First step compilations...\n{chrono.note}", lead_host)
    _write_note(f"Using classifier: {config.classifier}", lead_host)

    best_val_step = 0
    best_average_common_val_knn_top_1 = float("-inf")

    if config.pretrained_train_descriptors_dir != "":
        logging.info("Loading pretrained descriptors for the training set")
        pretrained_descriptors = {}

        for dataset_name in dataset_dict.meta_data["dataset_name"].split(","):
            dataset_name = dataset_name.strip()
            path = f"{config.pretrained_train_descriptors_dir}/{dataset_name}/train.npy"
            pretrained_descriptors[dataset_name] = knn_utils.load_split_descrs_func(path)
    else:
        pretrained_descriptors = None

    results = None

    for step in range(start_step + 1, total_steps + 1):
        with jax.profiler.StepTraceAnnotation("train", step_num=step):
            if (
                config.get("update_sampler", False)
                and step % config.update_sampler_every_steps == 0
                and step > config.update_sampler_after_epochs * steps_per_epoch
            ):
                sampler.update_ds_indices(
                    train_metrics,
                    results if config.update_sampler_mode == "train_loss+val_loss" else None,
                    step,
                )

                sample_weights_log = {
                    f"sampling_weights/{dataset_name}": weight
                    for dataset_name, weight in sampler.sampling_weights.items()
                }
                writer.write_scalars(step, sample_weights_log)

            train_batch, batch_dataset_idx, batch_dataset_name = (
                sampler.get_next_train_batch(step)
            )

            if pretrained_descriptors is not None:
                batch_indices = train_batch["index"].reshape(-1)
                train_batch["descriptors"] = pretrained_descriptors[batch_dataset_name][
                    batch_indices
                ].reshape(
                    train_batch["index"].shape[0],
                    train_batch["index"].shape[1],
                    -1,
                )

            train_state, t_metrics, t_logs = train_step_pmapped(
                train_state,
                train_batch,
                int(batch_dataset_idx),
            )

            train_metrics[batch_dataset_name].append(t_metrics)
            t_logs = jax.tree_util.tree_map(jax_utils.unreplicate, t_logs)
            extra_training_logs[batch_dataset_name].append(t_logs)

        for h in hooks:
            h(step)

        # Training summaries.
        if (
            (step % log_summary_steps == 0)
            or (step == total_steps)
            or (lead_host and chrono.warmup)
        ):
            chrono.pause()

            if lead_host:
                chrono.tick(step, writer, lambda note: _write_note(note, lead_host))

            train_summary = {}
            for dataset_name in train_metrics:
                train_summary.update(
                    logging_utils.log_train_summary(
                        step=step,
                        train_metrics=jax.tree_util.tree_map(
                            train_utils.unreplicate_and_get,
                            train_metrics[dataset_name],
                        ),
                        extra_training_logs=jax.tree_util.tree_map(
                            jax.device_get,
                            extra_training_logs[dataset_name],
                        ),
                        writer=writer,
                        prefix=f"train/{dataset_name}/",
                        key_separator="",
                    )
                )

            train_metrics = collections.defaultdict(list)
            extra_training_logs = collections.defaultdict(list)

            chrono.resume()

        # Regular checkpointing.
        if (
            ((step % checkpoint_steps == 0) and (step > 0)) or (step == total_steps)
        ) and config.checkpoint and not config.only_best_checkpoint:
            chrono.pause(wait_for=(train_state.params, train_state.opt_state))

            with report_progress.timed("checkpoint"):
                _save_checkpoint(
                    train_state,
                    workdir,
                    chrono,
                    lead_host,
                    max_to_keep=config.get("max_to_keep", 10),
                )

            chrono.resume()

        # k-NN evaluation.
        if (
            (step % log_eval_steps == 0)
            or (step == total_steps)
            or (config.do_knn_at_start and step == 1)
        ):
            chrono.pause(wait_for=(train_state.params,))

            if config.get("do_knn"):
                with report_progress.timed("knn"):
                    results = knn_utils.knn_step(
                        knn_evaluator,
                        train_state,
                        config,
                        workdir,
                        step,
                        writer,
                        load_descrs=False,
                    )
                    eval_summary = results

                    current_val_metric = _get_best_val_metric(results, config)

                    if current_val_metric > best_average_common_val_knn_top_1:
                        best_val_step = step
                        best_average_common_val_knn_top_1 = current_val_metric

                        if config.checkpoint and config.only_best_checkpoint:
                            with report_progress.timed("checkpoint"):
                                _save_best_checkpoint(
                                    train_state,
                                    workdir,
                                    chrono,
                                    lead_host,
                                )

                    writer.write_scalars(
                        step,
                        {
                            "best_val_step": best_val_step,
                            "best_average_common_val_knn_top_1": (
                                best_average_common_val_knn_top_1
                            ),
                        },
                    )

            if config.get("do_text_eval_during_validation"):
                logging.info("Running text evaluation during validation.")
                text_eval_utils.get_text_results(
                    knn_evaluator,
                    train_state,
                    config,
                    workdir,
                    step,
                    writer,
                )

            writer.flush()
            chrono.resume()

    # Final testing from best checkpoint.
    if config.do_final_testing and config.checkpoint:
        if config.only_best_checkpoint:
            train_state, _ = train_utils.restore_checkpoint(
                workdir,
                train_state,
                assert_exist=True,
                step=-1,
            )
        else:
            train_state, _ = train_utils.restore_checkpoint(
                workdir,
                train_state,
                assert_exist=True,
                step=int(best_val_step),
            )

        train_state = jax_utils.replicate(train_state)

        config.disabled_separate_knns = config.disabled_separate_final_eval_knns
        config.disabled_merged_knns = config.disabled_merged_final_eval_knns
        config.knn_eval_names = config.knn_eval_names_final
        config.calc_descriptor_information = False

        final_step = -1 if config.only_best_checkpoint else int(best_val_step)

        results = knn_utils.knn_step(
            knn_evaluator,
            train_state,
            config,
            workdir,
            final_step,
            writer,
            load_descrs=False,
            final_eval=True,
        )
        eval_summary = results

        if config.do_text_eval:
            logging.info("Running final text evaluation.")
            text_eval_utils.get_text_results(
                knn_evaluator,
                train_state,
                config,
                workdir,
                final_step,
                writer,
                final_eval=True,
            )

    train_utils.barrier_across_hosts()

    return train_state, train_summary, eval_summary