import logging
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp

from scenic.train_lib import train_utils


class Sampler:
    """Controls which dataset is sampled at each training step.

    The sampler precomputes a vector of dataset indices of length `total_steps`.
    At step `t`, the trainer reads the dataset index at position `t - 1` and
    fetches the next batch from that dataset's iterator.

    This class supports two categories of sampling behavior:

    1. Fixed strategies
       These produce a static schedule at initialization time, for example:
       - proportional to dataset size
       - perfectly balanced across datasets
       - round-robin over datasets
       - a user-defined "specialist" allocation

    2. Dynamic strategies
       These update dataset weights later based on observed training metrics
       (and optionally validation metrics), then regenerate the remaining
       sampling schedule.

    Notes:
    - Step indexing in the public API is 1-based, because the rest of the
      training loop uses step numbers starting from 1.
    - Dataset order is taken from `dataset_dict.meta_data["dataset_samples"]`.
      In your pipeline this is expected to be an ordered mapping.
    """

    def __init__(self, config, dataset_dict, total_steps, prng_key):
        """Initializes the sampler and creates the initial sampling schedule.

        Args:
            config: Experiment configuration with sampling options.
            dataset_dict: Dataset container with metadata and per-dataset iterators.
            total_steps: Total number of training steps for the run.
            prng_key: JAX PRNGKey used to shuffle the step schedule.
        """
        self.config = config
        self.dataset_dict = dataset_dict
        self.total_steps = total_steps
        self.prng_key = prng_key

        # These fields are currently only partially used. They are meant to
        # support future logic that reacts to validation trends over time.
        self.prev_val_domain_metrics = None
        self.delta_val_domain_metrics = None
        self.avg_other_deltas = None
        self.last_domain_used = None

        self.dataset_indices_per_step, self.sampling_weights = self._build_step_schedule(
            prng_key
        )

    @property
    def dataset_names(self):
        """Dataset names in the exact order used by the sampler."""
        return list(self.dataset_dict.meta_data["dataset_samples"].keys())

    def get_next_train_batch(self, step):
        """Returns the next batch for the dataset assigned to the given step.

        Args:
            step: Current training step, using 1-based indexing.

        Returns:
            A tuple:
              - batch: the next batch from the selected dataset iterator
              - dataset_idx: integer index of the selected dataset
              - dataset_name: dataset name corresponding to that index
        """
        dataset_idx = int(self.dataset_indices_per_step[step - 1])
        dataset_name = self.dataset_names[dataset_idx]
        batch = next(self.dataset_dict.train_iter[dataset_name])
        return batch, dataset_idx, dataset_name

    def calc_sampling_weights(
        self,
        train_domain_metrics,
        val_domain_metrics=None,
    ) -> Dict[str, float]:
        """Computes updated dataset sampling weights from training metrics.

        The current implementation uses per-dataset training losses to create a
        normalized weight distribution. Higher loss leads to a larger sampling
        weight, meaning that harder datasets get sampled more often.

        Optionally, validation metrics can also be read and stored, but the
        current code does not yet use them to modify the weights.

        Args:
            train_domain_metrics: Mapping from dataset name to training metrics
                accumulated over a logging window.
            val_domain_metrics: Optional validation metrics structure.

        Returns:
            A dict mapping dataset names to normalized sampling weights.
        """
        sampler_update_type = self.config.update_sampler_logit_type
        epoch_domain_loss = {}

        # Convert each dataset's accumulated metrics into a summarized scalar
        # loss that will drive the new sampling weights.
        for dataset_name in train_domain_metrics:
            dataset_train_metrics = jax.tree_util.tree_map(
                train_utils.unreplicate_and_get,
                train_domain_metrics[dataset_name],
            )
            dataset_train_metrics = train_utils.stack_forest(dataset_train_metrics)

            summary = train_utils.normalize_metrics_summary(
                jax.tree_util.tree_map(lambda x: x.sum(), dataset_train_metrics),
                "train",
            )
            epoch_domain_loss[dataset_name] = summary[
                f"{sampler_update_type}_classifier_loss"
            ]

        # Normalize losses into probabilities.
        normalizer = sum(epoch_domain_loss.values()) + 1e-8
        raw_weights = {
            dataset_name: loss / normalizer
            for dataset_name, loss in epoch_domain_loss.items()
        }

        # Optional smoothing over time.
        if self.config.do_ema_on_sampler:
            # Standard EMA convention:
            #   new_weight = decay * old_weight + (1 - decay) * raw_weight
            ema_decay = self.config.do_ema_on_sampler_decay
            new_sampling_weights = {}
            for dataset_name, raw_weight in raw_weights.items():
                old_weight = self.sampling_weights.get(dataset_name, raw_weight)
                new_sampling_weights[dataset_name] = (
                    ema_decay * old_weight + (1.0 - ema_decay) * raw_weight
                )
        else:
            new_sampling_weights = dict(raw_weights)

        # Make sure all datasets are present, even if one was absent in the
        # provided metrics dictionary for some reason.
        for dataset_name in self.sampling_weights:
            new_sampling_weights.setdefault(
                dataset_name,
                self.sampling_weights[dataset_name],
            )

        # Renormalize after any fallback insertions.
        total_weight = sum(new_sampling_weights.values()) + 1e-8
        new_sampling_weights = {
            k: v / total_weight for k, v in new_sampling_weights.items()
        }

        # Validation tracking is currently informational only.
        if val_domain_metrics is not None:
            new_val_metrics = {
                dataset_name: val_domain_metrics["map_results"][
                    f"{sampler_update_type}_embedd"
                ][f"{dataset_name}:common:val_knn:map_5"]
                * 100.0
                for dataset_name in self.sampling_weights
            }

            if self.prev_val_domain_metrics is None:
                self.prev_val_domain_metrics = new_val_metrics
                self.avg_other_deltas = {
                    dataset_name: 0.0 for dataset_name in new_val_metrics
                }
            elif self.prev_val_domain_metrics != new_val_metrics:
                self.delta_val_domain_metrics = {
                    k: new_val_metrics[k] - self.prev_val_domain_metrics[k]
                    for k in new_val_metrics
                }
                self.prev_val_domain_metrics = new_val_metrics

        return new_sampling_weights

    def update_ds_indices(self, train_domain_metrics, val_domain_metrics, current_step):
        """Updates sampling weights and rebuilds the future step schedule.

        This is meant to be called during training whenever the sampler should
        adapt to new metric information.

        Args:
            train_domain_metrics: Per-dataset training metrics.
            val_domain_metrics: Optional validation metrics.
            current_step: Current training step.

        Notes:
            The current implementation rebuilds the full step schedule for
            `total_steps`, not only the suffix after `current_step`. This keeps
            the logic simple, but it means already-consumed steps are also
            regenerated even though they will never be used again.
        """
        del current_step  # Kept in the signature because it may become useful later.

        self.sampling_weights = self.calc_sampling_weights(
            train_domain_metrics,
            val_domain_metrics,
        )

        # Advance the RNG so each resampling event produces a fresh permutation.
        self.prng_key, new_prng_key = jax.random.split(self.prng_key)

        self.dataset_indices_per_step = self._indices_from_weights(
            self.sampling_weights,
            new_prng_key,
        )

    def _indices_from_weights(
        self,
        sampling_weights: Dict[str, float],
        prng_key,
    ) -> jnp.ndarray:
        """Builds a shuffled dataset-index schedule from normalized weights.

        The number of occurrences of each dataset index is proportional to its
        sampling weight. Any rounding remainder is assigned to the last dataset,
        ensuring that the final schedule has exactly `total_steps` entries.

        Args:
            sampling_weights: Normalized dataset sampling weights.
            prng_key: JAX PRNGKey used to shuffle the schedule.

        Returns:
            A 1D JAX array of shape [total_steps] containing dataset indices.
        """
        dataset_indices = []
        step_counter = 0
        num_datasets = len(self.dataset_names)

        for i, dataset_name in enumerate(self.dataset_names):
            if i == num_datasets - 1:
                # Give the remaining steps to the last dataset so that the final
                # schedule length is exactly `total_steps`.
                ds_steps = self.total_steps - step_counter
            else:
                ds_steps = int(self.total_steps * sampling_weights[dataset_name])

            dataset_indices.append(jnp.full((ds_steps,), i))
            step_counter += ds_steps

        dataset_indices = jnp.concatenate(dataset_indices)
        dataset_indices = jax.random.permutation(prng_key, dataset_indices)
        return dataset_indices

    def _initial_sampling_weights(self) -> Dict[str, float]:
        """Returns the initial dataset sampling weights from the configured strategy."""
        num_datasets = len(self.dataset_names)

        if self.config.sampling_strategy == "dataset_size":
            total_samples = self.dataset_dict.meta_data["num_train_examples"]
            return {
                name: samples / total_samples
                for name, samples in self.dataset_dict.meta_data["dataset_samples"].items()
            }

        if self.config.sampling_strategy == "balanced":
            return {name: 1.0 / num_datasets for name in self.dataset_names}

        if self.config.sampling_strategy == "specialist_top_steps":
            top_steps = self.config.specialist_top_steps
            total_top_steps = sum(top_steps)
            return {
                name: top_steps[i] / total_top_steps
                for i, name in enumerate(self.dataset_names)
            }

        raise ValueError(f"Unknown sampling strategy: {self.config.sampling_strategy}")

    def _build_step_schedule(self, prng_key) -> Tuple[jnp.ndarray, Dict[str, float]]:
        """Builds the initial step schedule and the corresponding weights.

        Supported strategies:
        - round_robin:
            cycles through datasets in fixed order
        - dataset_size:
            allocates steps proportionally to dataset size
        - balanced:
            allocates equal probability to each dataset
        - specialist_top_steps:
            allocates according to user-provided step counts

        Args:
            prng_key: JAX PRNGKey used for shuffling when needed.

        Returns:
            A tuple:
              - dataset_indices_per_step: array of dataset indices for each step
              - sampling_weights: dict of dataset sampling probabilities
        """
        logging.info("Creating sampling indices.")
        logging.info("Sampling strategy: %s", self.config.sampling_strategy)

        num_datasets = len(self.dataset_names)

        if self.config.sampling_strategy == "round_robin":
            one_round = jnp.arange(num_datasets)
            times_to_repeat = self.total_steps // num_datasets

            dataset_indices_per_step = jnp.tile(one_round, times_to_repeat)

            steps_left = self.total_steps - dataset_indices_per_step.shape[0]
            if steps_left > 0:
                dataset_indices_per_step = jnp.concatenate(
                    [dataset_indices_per_step, one_round[:steps_left]]
                )

            sampling_weights = {
                name: 1.0 / num_datasets for name in self.dataset_names
            }
            return dataset_indices_per_step, sampling_weights

        sampling_weights = self._initial_sampling_weights()
        dataset_indices_per_step = self._indices_from_weights(
            sampling_weights,
            prng_key,
        )
        return dataset_indices_per_step, sampling_weights