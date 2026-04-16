import logging
import os
from typing import Dict, Iterable, List, Tuple

import flax
import numpy as np

from universal_embedding import grain_datasets


def _l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(x, ord=2, axis=-1, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms


def _recall_at_k_from_rankings(
    ranked_ids: List[List],
    ground_truth,
    ks: Iterable[int] = (1, 5, 10),
) -> Dict[int, float]:
    """Computes recall@K given ranked predictions and ground truth."""
    ks = tuple(ks)
    correct = {k: 0 for k in ks}
    n_queries = len(ranked_ids)

    if n_queries == 0:
        return {k: 0.0 for k in ks}

    for i, preds in enumerate(ranked_ids):
        gt = ground_truth[i]

        if isinstance(gt, (list, tuple, set)):
            gt_set = set(gt)
            for k in ks:
                if any(pred in gt_set for pred in preds[:k]):
                    correct[k] += 1
        else:
            for k in ks:
                if gt in preds[:k]:
                    correct[k] += 1

    return {k: correct[k] / n_queries for k in ks}


def _get_text_eval_datasets(config) -> Dict[str, Dict[str, str]]:
    """Returns text-eval dataset paths.
    """

    if hasattr(config, "text_datasets") and config.text_datasets:
        datasets = config.text_datasets.split(",")
        result = {}
        model_class = config.model_class.lower()
        if "siglip" in model_class:
            model_suffix = "siglip"
        elif "tips" in model_class:
            model_suffix = "tips"
        else:
            raise ValueError(f"Unsupported model_class for text eval: {config.model_class}")
        for dataset in datasets:
            dataset = dataset.strip()
            result[dataset] = {
                "tfrecord_path": f"{dataset}/queries.tfrecord",
                "text_embeddings_path": f"{dataset}/{dataset.split('/')[-1]}_text_embeddings_{model_suffix}.npy",
                "gt_path": f"{dataset}/{dataset.split('/')[-1]}_gt.npy",
            }

        return result


def _extract_image_descriptors(
    config,
    knn_evaluator,
    train_state,
    tfrecord_path,
    descriptor_type: str,
):
    dataset_dict = grain_datasets.get_extract_dataset(
        config,
        tfrecord_path,
        config.get("eval_batch_size", config.batch_size),
    )

    query_image_descriptors = knn_evaluator._get_repr_no_indices(
        train_state,
        dataset_dict.extract_iter,
        domain_idx=-1,
    )

    return {
        "ids": dataset_dict.meta_data["ids"],
        "descriptors": query_image_descriptors[descriptor_type],
    }


def _load_text_embeddings(text_embeddings_path: str):
    raw = np.load(text_embeddings_path, allow_pickle=True).item()
    return {
        "ids": list(raw.keys()),
        "descriptors": np.asarray(list(raw.values())),
    }


def _maybe_project_text_embeddings(
    text_embeddings: np.ndarray,
    image_embeddings: np.ndarray,
    train_state,
) -> np.ndarray:
    """Projects text embeddings if they do not match image embedding dimensionality."""
    if text_embeddings.shape[1] == image_embeddings.shape[1]:
        return text_embeddings

    train_state_unreplicated = flax.jax_utils.unreplicate(train_state)
    encoder_params = train_state_unreplicated.params["Embedding_Head"]["encoder_projection_domain_0"]["Dense_0"]

    weight_matrix = encoder_params["kernel"]
    bias = encoder_params["bias"]

    return np.dot(text_embeddings, weight_matrix) + bias


def _rank_ids_from_similarity(
    query_descriptors: np.ndarray,
    index_descriptors: np.ndarray,
    index_ids: List,
) -> List[List]:
    similarities = np.dot(query_descriptors, index_descriptors.T)
    ranked_neighbors = np.argsort(-similarities, axis=1)
    return [[index_ids[idx] for idx in ranking] for ranking in ranked_neighbors]


def eval_one_dataset(
    config,
    knn_evaluator,
    train_state,
    writer,
    tfrecord_path,
    text_embeddings_path,
    gt_path,
    dataset_name,
    step,
    descriptor_type,
    final_eval: bool = False,
):
    """Runs bidirectional image-text retrieval evaluation for one dataset."""
    image_query_embeddings = _extract_image_descriptors(
        config,
        knn_evaluator,
        train_state,
        tfrecord_path,
        descriptor_type,
    )

    text_index_embeddings = _load_text_embeddings(text_embeddings_path)
    text_index_embeddings["descriptors"] = _maybe_project_text_embeddings(
        text_index_embeddings["descriptors"],
        image_query_embeddings["descriptors"],
        train_state,
    )

    text_index_embeddings["descriptors"] = _l2_normalize(text_index_embeddings["descriptors"])
    image_query_embeddings["descriptors"] = _l2_normalize(image_query_embeddings["descriptors"])

    ground_truth_i2t = np.load(gt_path, allow_pickle=True).item()

    # Image -> Text
    ranked_text_ids = _rank_ids_from_similarity(
        image_query_embeddings["descriptors"],
        text_index_embeddings["descriptors"],
        text_index_embeddings["ids"],
    )
    query_ids = image_query_embeddings["ids"]
    i2t_gt = [ground_truth_i2t[qid] for qid in query_ids]
    i2t_recalls = _recall_at_k_from_rankings(ranked_text_ids, i2t_gt, ks=(1, 5, 10))

    logging.info(
        "%s / %s I->T R@1=%.4f R@5=%.4f R@10=%.4f",
        dataset_name,
        descriptor_type,
        i2t_recalls[1],
        i2t_recalls[5],
        i2t_recalls[10],
    )

    results_to_log = {
        f"text_eval/{descriptor_type}/{dataset_name}_I->T/R@1": i2t_recalls[1],
        f"text_eval/{descriptor_type}/{dataset_name}_I->T/R@5": i2t_recalls[5],
        f"text_eval/{descriptor_type}/{dataset_name}_I->T/R@10": i2t_recalls[10],
    }

    # Text -> Image
    caption_to_image = {}
    for image_id, caption_ids in ground_truth_i2t.items():
        for caption_id in caption_ids:
            caption_to_image[caption_id] = image_id

    ranked_image_ids = _rank_ids_from_similarity(
        text_index_embeddings["descriptors"],
        image_query_embeddings["descriptors"],
        image_query_embeddings["ids"],
    )
    t2i_gt = [caption_to_image[qid] for qid in text_index_embeddings["ids"]]
    t2i_recalls = _recall_at_k_from_rankings(ranked_image_ids, t2i_gt, ks=(1, 5, 10))

    logging.info(
        "%s / %s T->I R@1=%.4f R@5=%.4f R@10=%.4f",
        dataset_name,
        descriptor_type,
        t2i_recalls[1],
        t2i_recalls[5],
        t2i_recalls[10],
    )

    results_to_log.update(
        {
            f"text_eval/{descriptor_type}/{dataset_name}_T->I/R@1": t2i_recalls[1],
            f"text_eval/{descriptor_type}/{dataset_name}_T->I/R@5": t2i_recalls[5],
            f"text_eval/{descriptor_type}/{dataset_name}_T->I/R@10": t2i_recalls[10],
        }
    )

    if final_eval:
        results_to_log = {f"final_eval/{k}": v for k, v in results_to_log.items()}

    writer.write_scalars(step, results_to_log)
    return results_to_log


def get_text_results(
    knn_evaluator,
    train_state,
    config,
    workdir,
    step,
    writer,
    final_eval: bool = False,
):
    """Runs text retrieval evaluation for all configured datasets."""
    del workdir

    eval_datasets = _get_text_eval_datasets(config)

    for absolute_dataset_path, dataset_config in eval_datasets.items():
        dataset_name = absolute_dataset_path.split('/')[-1]
        logging.info("Evaluating text retrieval on %s.", dataset_name)

        tfrecord_path = dataset_config["tfrecord_path"]
        text_embeddings_path = dataset_config["text_embeddings_path"]
        gt_path = dataset_config["gt_path"]

        for descriptor_type in [x.strip() for x in config.embedd_to_eval.split(",") if x.strip()]:
            logging.info("Evaluating descriptor type %s.", descriptor_type)

            eval_one_dataset(
                config,
                knn_evaluator,
                train_state,
                writer,
                tfrecord_path,
                text_embeddings_path,
                gt_path,
                dataset_name,
                step,
                descriptor_type,
                final_eval=final_eval,
            )
