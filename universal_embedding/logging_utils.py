from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from clu import metric_writers
from scenic.train_lib.train_utils import normalize_metrics_summary, stack_forest


def vector_psum_metric_normalizer(
    metrics: Tuple[jnp.ndarray, jnp.ndarray],
    axis_name: Union[str, Tuple[str, ...]] = "batch",
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Applies psum over the given tuple of (metric, normalizer)."""
    psumed_metric = jax.lax.psum(jnp.sum(metrics[0], axis=0), axis_name=axis_name)
    psumed_normalizer = jax.lax.psum(jnp.sum(metrics[1]), axis_name=axis_name)
    return psumed_metric, psumed_normalizer


def log_train_summary(
    step: int,
    *,
    writer: metric_writers.MetricWriter,
    train_metrics: Sequence[Dict[str, Tuple[float, int]]],
    extra_training_logs: Optional[Sequence[Dict[str, Any]]] = None,
    metrics_normalizer_fn: Optional[
        Callable[[Dict[str, Tuple[float, int]], str], Dict[str, float]]
    ] = None,
    prefix: str = "train",
    key_separator: str = "_",
    flush_writer: bool = True,
) -> Dict[str, float]:
    """Computes and logs train metrics."""
    # Get metrics from devices.
    train_metrics = stack_forest(train_metrics)

    # Separate vector metrics from scalar metrics.
    vector_train_metrics = {}

    # Log vector metrics as histograms.
    for metric_name, metric_value in vector_train_metrics.items():
        # Expect metric_value to be something like (values, normalizers).
        value_sum = metric_value[0].sum(axis=0)
        normalizer_sum = metric_value[1].sum(axis=0)

        # Avoid divide-by-zero.
        final_vector = np.asarray(
            jnp.where(normalizer_sum > 0, value_sum / normalizer_sum, 0.0)
        )

        # Convert to integer counts for histogram expansion.
        counts = np.rint(final_vector * 1000).astype(np.int32)
        counts = np.clip(counts, 0, None)

        histogram_values = np.repeat(np.arange(len(counts)), counts)

        writer.write_histograms(
            step,
            {
                key_separator.join((prefix, metric_name)): histogram_values,
            },
        )

    # Scalar metrics.
    train_metrics_summary = jax.tree_util.tree_map(lambda x: x.sum(), train_metrics)
    metrics_normalizer_fn = metrics_normalizer_fn or normalize_metrics_summary
    train_metrics_summary = metrics_normalizer_fn(train_metrics_summary, "train")

    # Additional training logs.
    extra_training_logs = extra_training_logs or [{}]
    train_logs = stack_forest(extra_training_logs)

    writer.write_scalars(
        step,
        {
            key_separator.join((prefix, key)): val
            for key, val in train_metrics_summary.items()
        },
    )

    writer.write_scalars(
        step,
        {key: val.mean() for key, val in train_logs.items()},
    )

    if flush_writer:
        writer.flush()

    return train_metrics_summary