import functools
import json
import os
import sys

from absl import app
from absl import flags
from absl import logging
from clu import metric_writers
from clu import platform
import flax
import flax.linen as nn
import jax
from ml_collections import config_flags
import tensorflow as tf
from tensorflow.io import gfile
import wandb

from universal_embedding import utils


FLAGS = flags.FLAGS


# General flags used across projects.
config_flags.DEFINE_config_file(
    "config",
    None,
    "Training configuration.",
    lock_config=False,  # Needed because config values are modified at runtime.
)
flags.DEFINE_string("workdir", None, "Work unit directory.")
flags.DEFINE_string(
    "dataset_service_address",
    None,
    "Address of the tf.data service.",
)

# Weights & Biases flags.
flags.DEFINE_string("wandb_project", None, "Wandb project name.")
flags.DEFINE_string("wandb_name", None, "Wandb experiment name.")
flags.DEFINE_string("wandb_group", None, "Wandb group name.")
flags.DEFINE_string("wandb_entity", None, "Wandb team/entity name.")
flags.DEFINE_boolean("use_wandb", False, "Whether to log to Weights & Biases.")

# Debugging flag.
flags.DEFINE_boolean("debug_on_tpu", False, "Use CPU instead of TPU for debugging.")

flags.mark_flags_as_required(["config", "workdir"])

# Keep Scenic/Flax behavior as in your original file.
flax.config.update("flax_use_orbax_checkpointing", False)


def run(main, knn=False, descr_eval=False):
    """Entry point wrapper.

    Args:
        main: Callable with signature main(rng, config, workdir, writer).
        knn: Whether this is a k-NN evaluation run.
        descr_eval: Whether this is a descriptor evaluation run.
    """
    # Expose JAX backend flags such as:
    # --jax_backend_target and --jax_xla_backend
    jax.config.config_with_absl()
    app.run(
        functools.partial(
            _run_main,
            main=main,
            knn=knn,
            descr_eval=descr_eval,
        )
    )


def _maybe_update_config_for_knn():
    """Updates FLAGS.config with training-dependent values for k-NN eval."""
    if not FLAGS.config.no_finetune:
        train_config_path = os.path.join(FLAGS.config.train_dir, "config.json")
        train_config_params = utils.read_config(train_config_path)
        train_config_params.update(FLAGS.config)
        FLAGS.config = train_config_params
    else:
        utils.calc_train_dependent_config_values(FLAGS.config, knn=True)


def _maybe_save_training_config():
    """Computes dependent config values and saves final config to workdir."""
    utils.calc_train_dependent_config_values(FLAGS.config)

    gfile.makedirs(FLAGS.workdir)
    config_path = os.path.join(FLAGS.workdir, "config.json")

    with gfile.GFile(config_path, "w") as f:
        json.dump(
            json.loads(FLAGS.config.to_json_best_effort()),
            f,
            indent=4,
        )


def _setup_runtime():
    """Applies runtime configuration for JAX, TF, and Flax."""
    if FLAGS.debug_on_tpu:
        jax.config.update("jax_platform_name", "cpu")

    # Prevent TensorFlow from reserving GPU memory that JAX may need.
    tf.config.experimental.set_visible_devices([], "GPU")

    # Enable named_call wrapping for easier profiling.
    nn.enable_named_call()

    if FLAGS.jax_backend_target:
        logging.info("Using JAX backend target %s", FLAGS.jax_backend_target)
        jax_xla_backend = (
            "None" if FLAGS.jax_xla_backend is None else FLAGS.jax_xla_backend
        )
        logging.info("Using JAX XLA backend %s", jax_xla_backend)

    logging.info("JAX host: %d / %d", jax.process_index(), jax.process_count())
    logging.info("JAX devices: %r", jax.devices())

    platform.work_unit().set_task_status(
        f"host_id: {jax.process_index()},host_count: {jax.process_count()}"
    )

    if jax.process_index() == 0:
        platform.work_unit().create_artifact(
            platform.ArtifactType.DIRECTORY,
            FLAGS.workdir,
            "Workdir",
        )


def _init_wandb():
    """Initializes Weights & Biases if enabled."""
    if not FLAGS.use_wandb:
        return

    wandb.init(
        project=FLAGS.wandb_project,
        name=FLAGS.wandb_name,
        group=FLAGS.wandb_group,
        entity=FLAGS.wandb_entity,
        sync_tensorboard=True,
        config=json.loads(FLAGS.config.to_json_best_effort()),
    )


def _finish_wandb():
    """Closes Weights & Biases run if enabled."""
    if FLAGS.use_wandb:
        wandb.finish()


def _run_main(argv, *, main, knn, descr_eval):
    """Runs the provided main function after initial setup."""
    del argv
    if knn:
        _maybe_update_config_for_knn()
    elif not descr_eval:
        _maybe_save_training_config()

    _setup_runtime()

    rng = jax.random.PRNGKey(FLAGS.config.rng_seed)
    logging.info("RNG: %s", rng)

    _init_wandb()

    writer = metric_writers.create_default_writer(
        FLAGS.workdir,
        just_logging=jax.process_index() > 0,
        asynchronous=False,
    )

    try:
        main(
            rng=rng,
            config=FLAGS.config,
            workdir=FLAGS.workdir,
            writer=writer,
        )
    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
        sys.exit(0)
    finally:
        _finish_wandb()