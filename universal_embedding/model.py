"""Universal embedding classification model and loss/metric helpers."""

import functools
from typing import Dict, List, Optional, Tuple, Union

from flax.training import common_utils
import jax
import jax.numpy as jnp
import ml_collections

from scenic.model_lib.base_models import base_model
from scenic.model_lib.base_models import model_utils
from scenic.model_lib.base_models import multilabel_classification_model

from universal_embedding import loss_utils


MetricDict = Dict[str, Tuple[float, int]]
LossDict = Dict[str, jnp.ndarray]
WeightDict = Dict[str, float]


def _split_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_float_csv(value: str) -> List[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _classifier_loss_specs(config, domain: int):
    names = _split_csv(config.loss.classif_losses_on[str(domain)])
    weights = _parse_float_csv(config.loss.classif_losses_weights[str(domain)])
    types = _split_csv(config.loss.classif_losses_types[str(domain)])
    margins = _parse_float_csv(config.loss.classif_losses_margins[str(domain)])

    if not (len(names) == len(weights) == len(types) == len(margins)):
        raise ValueError(
            f"Classifier loss config lengths do not match for domain {domain}: "
            f"{len(names)}, {len(weights)}, {len(types)}, {len(margins)}"
        )

    return list(zip(names, weights, types, margins))


def _get_multihot_targets(batch, logits, target_is_multihot: bool):
    if target_is_multihot:
        return batch["label"]
    return common_utils.onehot(batch["label"], logits.shape[-1])


def _compute_classifier_loss(
    *,
    logits,
    one_hot_targets,
    classif_loss_type: str,
    classif_loss_margin: float,
    config,
    model_params,
    domain: int,
    classif_loss_on: str,
):

    transformed_logits = loss_utils._transform_logits(
        logits,
        one_hot_targets,
        classif_loss_type,
        classif_loss_margin,
        config.loss,
        model_params=model_params,
        domain=domain,
        classif_loss_on=classif_loss_on,
    )

    return model_utils.weighted_softmax_cross_entropy(
        transformed_logits,
        one_hot_targets,
        label_smoothing=config.get("label_smoothing"),
    )


def _collect_loss_terms(
    *,
    outputs: Dict,
    batch: base_model.Batch,
    domain: int,
    config: ml_collections.ConfigDict,
    target_is_multihot: bool,
    model_params: Optional[jnp.ndarray],
    pretrained_params: Optional[jnp.ndarray],
) -> Tuple[LossDict, WeightDict]:
    """Builds all configured loss terms and their aggregation weights."""
    loss_terms: LossDict = {}
    loss_weights: WeightDict = {}

    if model_params is not None and pretrained_params is not None and config.loss.pretrained_weights_loss:
        loss_terms["pretrained_weights_loss"] = loss_utils.pretrained_weights_loss_fn(
            model_params,
            pretrained_params,
        )
        loss_weights["pretrained_weights_loss"] = config.loss.pretrained_weights_loss_weight

    for classif_loss_on, classif_loss_weight, classif_loss_type, classif_loss_margin in _classifier_loss_specs(config, domain):
        logits = outputs["classifier"][f"{classif_loss_on}_logits"]
        one_hot_targets = _get_multihot_targets(batch, logits, target_is_multihot)

        classif_loss = _compute_classifier_loss(
            logits=logits,
            one_hot_targets=one_hot_targets,
            classif_loss_type=classif_loss_type,
            classif_loss_margin=classif_loss_margin,
            config=config,
            model_params=model_params,
            domain=domain,
            classif_loss_on=classif_loss_on,
        )
        loss_terms[f"{classif_loss_on}_classifier_loss"] = classif_loss
        loss_weights[f"{classif_loss_on}_classifier_loss"] = classif_loss_weight

    if str(domain) in _split_csv(config.loss.pretrained_embedding_distill_loss_on):
        pretrained_embedding_distill_loss = loss_utils.embedding_distillation_loss(
            batch["descriptors"],
            outputs["embeddings"]["backbone_out_embedd"],
        ).mean()

        loss_terms["pretrained_embedding_distill_loss"] = pretrained_embedding_distill_loss
        loss_weights["pretrained_embedding_distill_loss"] = (
            config.loss.pretrained_embedding_distill_loss_weight
        )

    return loss_terms, loss_weights


def _aggregate_loss(
    loss_dict: LossDict,
    weight_dict: WeightDict,
    aggregation_type: str,
) -> float:
    total_loss = 0.0
    total_weight = 0.0

    for loss_name, loss_value in loss_dict.items():
        weight = weight_dict.get(loss_name, 1.0)
        total_loss += loss_value * weight
        total_weight += weight

    if aggregation_type == "weighted_average":
        return total_loss / total_weight if total_weight != 0 else total_loss
    if aggregation_type == "weighted_sum":
        return total_loss

    raise ValueError(f"Unknown aggregation type: {aggregation_type}")


def _add_metric(
    metrics: MetricDict,
    name: str,
    value,
    normalizer,
    axis_name,
):
    metrics[name] = model_utils.psum_metric_normalizer(
        (value, normalizer),
        axis_name=axis_name,
    )


def _add_classifier_metrics(
    *,
    metrics: MetricDict,
    outputs: Dict,
    batch: base_model.Batch,
    domain: int,
    config: ml_collections.ConfigDict,
    target_is_multihot: bool,
    model_params: Optional[jnp.ndarray],
    axis_name,
):
    weights = batch.get("batch_mask")

    for classif_loss_on, _, classif_loss_type, classif_loss_margin in _classifier_loss_specs(config, domain):
        logits = outputs["classifier"][f"{classif_loss_on}_logits"]
        targets = _get_multihot_targets(batch, logits, target_is_multihot)

        _add_metric(
            metrics,
            f"{classif_loss_on}_classifier_prec@1",
            model_utils.weighted_top_one_correctly_classified(logits, targets, weights),
            model_utils.num_examples(logits, targets, weights),
            axis_name,
        )

        transformed_logits = loss_utils._transform_logits(
            logits,
            targets,
            classif_loss_type,
            classif_loss_margin,
            config.loss,
            model_params=model_params,
            domain=domain,
            classif_loss_on=classif_loss_on,
        )
        _add_metric(
            metrics,
            f"{classif_loss_on}_classifier_loss",
            model_utils.weighted_unnormalized_softmax_cross_entropy(
                transformed_logits,
                targets,
                weights,
            ),
            model_utils.num_examples(transformed_logits, targets, weights),
            axis_name,
        )

        if config.loss.trainable_scale:
            _add_metric(
                metrics,
                "trainable_scale",
                model_params["Embedding_Head"][f"{classif_loss_on}_logit_scale_{domain}"],
                1.0,
                axis_name,
            )


def _add_generic_loss_metrics(
    *,
    metrics: MetricDict,
    outputs: Dict,
    batch: base_model.Batch,
    domain: int,
    config: ml_collections.ConfigDict,
    model_params: Optional[jnp.ndarray],
    pretrained_params: Optional[jnp.ndarray],
    axis_name,
):

    if batch.get("descriptors") is not None:
        pretrained_embedding_distill_loss = loss_utils.embedding_distillation_loss(
            batch["descriptors"],
            outputs["embeddings"]["backbone_out_embedd"],
        )
        _add_metric(
            metrics,
            "pretrained_embedding_distill_loss",
            pretrained_embedding_distill_loss.sum(),
            pretrained_embedding_distill_loss.size,
            axis_name,
        )

    if model_params is not None and pretrained_params is not None:
        pretrained_weights_loss = loss_utils.pretrained_weights_loss_fn(
            model_params,
            pretrained_params,
        )
        _add_metric(
            metrics,
            "pretrained_weights_loss",
            pretrained_weights_loss,
            1.0,
            axis_name,
        )


def classification_metrics_function(
    outputs: Dict,
    batch: base_model.Batch,
    domain: int,
    config: ml_collections.ConfigDict,
    target_is_multihot: bool = False,
    model_params: Optional[jnp.ndarray] = None,
    axis_name: Union[str, Tuple[str, ...]] = "batch",
    pretrained_params: Optional[jnp.ndarray] = None,
) -> MetricDict:
    """Computes training/eval metrics for the universal embedding model.

    This includes:
    - classifier top-1 accuracy and classifier losses
    - distillation losses
    - optional pretrained regularization metrics
    """
    metrics: MetricDict = {}

    _add_classifier_metrics(
        metrics=metrics,
        outputs=outputs,
        batch=batch,
        domain=domain,
        config=config,
        target_is_multihot=target_is_multihot,
        model_params=model_params,
        axis_name=axis_name,
    )

    _add_generic_loss_metrics(
        metrics=metrics,
        outputs=outputs,
        batch=batch,
        domain=domain,
        config=config,
        model_params=model_params,
        pretrained_params=pretrained_params,
        axis_name=axis_name,
    )

    return metrics


class UniversalEmbeddingModel(
    multilabel_classification_model.MultiLabelClassificationModel
):
    """Base class for universal embedding classification models.

    This wrapper exposes:
    - `get_metrics_fn(...)` for trainer-side metric logging
    - `loss_function(...)` for optimization
    - project-specific bookkeeping for dynamic loss weighting
    """

    def __init__(self, config, dataset_dict_meta_data) -> None:
        super(multilabel_classification_model.MultiLabelClassificationModel, self).__init__(
            config,
            dataset_dict_meta_data,
        )

    def get_metrics_fn(self, split: Optional[str] = None) -> base_model.MetricFn:
        """Returns the metric function used by the trainer."""
        del split
        return functools.partial(
            classification_metrics_function,
            target_is_multihot=self.dataset_meta_data.get("target_is_onehot", False),
            config=self.config,
        )

    def loss_function(
        self,
        outputs: Dict,
        batch: base_model.Batch,
        domain: int,
        model_params: Optional[jnp.ndarray] = None,
        pretrained_params: Optional[jnp.ndarray] = None,
    ) -> float:
        """Builds and aggregates all configured training losses for one batch."""
        loss_terms, loss_weights = _collect_loss_terms(
            outputs=outputs,
            batch=batch,
            domain=domain,
            config=self.config,
            target_is_multihot=self.dataset_meta_data.get("target_is_onehot", False),
            model_params=model_params,
            pretrained_params=pretrained_params,
        )

        return _aggregate_loss(
            loss_terms,
            loss_weights,
            self.config.loss.aggregation_type,
        )
