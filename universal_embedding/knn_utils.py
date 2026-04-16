"""Utils for K-nearest neighbor evaluation."""

import collections
import copy
import functools
import os
from typing import Any, Dict, Optional, Set

from absl import logging
from clu import metric_writers
from flax import jax_utils
import flax
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow.io import gfile
from tqdm import tqdm

from universal_embedding import dataset_infos
from universal_embedding import grain_datasets
from universal_embedding import loss_utils
from universal_embedding import metrics


def _parse_csv_set(value: str) -> Set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def _safe_l2_normalize(x: np.ndarray, axis: int) -> np.ndarray:
    norms = np.linalg.norm(x, axis=axis, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return x / norms


class KNNEvaluator:
    """Class for KNN evaluation."""

    def __init__(
        self,
        config,
        representation_fn,
        knn_query_batch_size,
        extract_only_descriptors: bool = False,
    ):
        self.config = config
        self.extract_only_descriptors = extract_only_descriptors

        if representation_fn is not None:
            self.repr_fn = jax.pmap(
                functools.partial(representation_fn),
                donate_argnums=(1,),
                axis_name="batch",
                static_broadcasted_argnums=(2,),
            )
        else:
            self.repr_fn = None

        self.knn_query_batch_size = knn_query_batch_size
        self.compute_knn_metrics_fun = self.compute_knn_metrics

    @staticmethod
    def _run_knn(k, index_descrs, query_descrs):
        all_similarities = jnp.matmul(query_descrs, jnp.transpose(index_descrs))
        similarities_k_sorted, indices_k_sorted = jax.lax.top_k(all_similarities, k)
        return similarities_k_sorted, indices_k_sorted

    def _get_repr(
        self,
        train_state,
        data,
        domain_idx,
    ):
        """Compute representations for a dataset split and restore original order."""
        embedding_dict_final = {}
        all_indices = []

        logging.info("Extracting representations.")
        data = tqdm(
            data,
            desc="Extracting representations",
            unit="batch",
            leave=True,
            dynamic_ncols=True,
        )

        for batch in data:
            embeddings_dict, batch = self.repr_fn(
                train_state,
                batch,
                domain_idx,
            )

            mask = np.asarray(jax_utils.unreplicate(batch["batch_mask"])).astype(bool)
            correct_indices = np.asarray(jax_utils.unreplicate(batch["index"]))[mask]
            all_indices.append(correct_indices)

            for embed_type, embed_value in embeddings_dict.items():
                embedding_dict_final.setdefault(embed_type, [])
                embedding_dict_final[embed_type].append(
                    np.asarray(jax_utils.unreplicate(embed_value))[mask]
                )

        all_indices = np.concatenate(all_indices, axis=0)
        sorted_order = all_indices.argsort()

        for embed_type in embedding_dict_final:
            embedding_dict_final[embed_type] = np.concatenate(
                embedding_dict_final[embed_type],
                axis=0,
            )[sorted_order]

        logging.info("Finished extracting representations.")
        return embedding_dict_final

    def _get_repr_no_indices(
        self,
        train_state,
        data,
        domain_idx,
    ):
        """Compute representations for a dataset split without index-based reordering."""
        embedding_dict_final = {}

        logging.info("Extracting representations.")
        data = tqdm(
            data,
            desc="Extracting representations",
            unit="batch",
            leave=True,
            dynamic_ncols=True,
        )

        for batch in data:
            embeddings_dict, batch = self.repr_fn(
                train_state,
                batch,
                domain_idx,
            )

            mask = np.asarray(jax_utils.unreplicate(batch["batch_mask"])).astype(bool)

            for embed_type, embed_value in embeddings_dict.items():
                embedding_dict_final.setdefault(embed_type, [])
                embedding_dict_final[embed_type].append(
                    np.asarray(jax_utils.unreplicate(embed_value))[mask]
                )

        for embed_type in embedding_dict_final:
            embedding_dict_final[embed_type] = np.concatenate(
                embedding_dict_final[embed_type],
                axis=0,
            )

        logging.info("Finished extracting representations.")
        return embedding_dict_final

    def compute_knn_metrics(
        self,
        lookup_key,
        query_results,
        index_results,
        query_paths,
        index_paths,
        throw_first,
        top_k,
        config=None,
        query_labels=None,
        index_labels=None,
        query_domains=None,
        index_domains=None,
        embed_types=None,
    ):
        """Compute KNN metrics on the query and index descriptors."""
        del lookup_key, query_paths, index_paths, config

        if embed_types is None:
            raise ValueError("embed_types must be provided.")

        actual_top_k = top_k
        retrieval_top_k = top_k + int(throw_first)
        pred_rank = int(throw_first)

        query_labels = np.asarray(query_labels, dtype=object)
        index_labels = np.asarray(index_labels, dtype=object)
        query_domains = np.asarray(query_domains)
        index_domains = np.asarray(index_domains)

        unique_query_domains = np.unique(query_domains)
        if unique_query_domains.shape[0] != 1:
            raise ValueError("Expected a single unique query domain for evaluation.")

        query_domain_val = unique_query_domains[0]

        domain_mask = index_domains == query_domain_val
        index_labels_in_domain = np.asarray(index_labels[domain_mask], dtype=object)
        index_label_counter = collections.Counter(index_labels_in_domain.tolist())

        classes_in_index = 0
        for domain in np.unique(index_domains):
            domain_labels = index_labels[index_domains == domain]
            classes_in_index += len(np.unique(domain_labels))

        results_dict = {}

        for embed_type in embed_types:
            query_emb = np.asarray(query_results[embed_type])
            index_emb = np.asarray(index_results[embed_type])

            num_query = query_emb.shape[0]
            batch_size = self.knn_query_batch_size
            num_batch = (num_query + batch_size - 1) // batch_size

            logging.info("Embed type: %s", embed_type)
            logging.info("Num query embeddings: %d", num_query)
            logging.info("Num index embeddings: %d", index_emb.shape[0])
            logging.info("Embedding dimension: %d", query_emb.shape[-1])
            logging.info("Classes in index: %d", classes_in_index)
            logging.info("Num eval batches: %d", num_batch)

            num_knn_correct = 0.0
            cumulative_mmp = 0.0
            cumulative_map = 0.0

            pmapped_run_knn = jax.pmap(
                functools.partial(
                    self._run_knn,
                    k=retrieval_top_k,
                    index_descrs=index_emb,
                )
            )

            for i in range(num_batch):
                start = i * batch_size
                end = min((i + 1) * batch_size, num_query)
                batch_queries = query_emb[start:end]

                array_batches, masks = self.split_and_pad(batch_queries)
                similarities_k_sorted, indices_k_sorted = pmapped_run_knn(
                    query_descrs=array_batches
                )

                similarities_k_sorted = np.asarray(similarities_k_sorted[masks])
                indices_k_sorted = np.asarray(indices_k_sorted[masks])

                for m in range(end - start):
                    global_idx = start + m

                    nearest = [
                        (
                            index_labels[indices_k_sorted[m, l]],
                            index_domains[indices_k_sorted[m, l]],
                            similarities_k_sorted[m, l],
                        )
                        for l in range(similarities_k_sorted.shape[1])
                    ]

                    pred_label = nearest[pred_rank][0]
                    pred_domain = nearest[pred_rank][1]

                    true_label = query_labels[global_idx]
                    true_domain = query_domains[global_idx]

                    num_knn_correct += metrics.universal_classif_accuracy(
                        pred_label,
                        pred_domain,
                        true_label,
                        true_domain,
                    )

                    if isinstance(true_label, (int, np.integer)):
                        num_index_label = index_label_counter[true_label]
                    else:
                        num_index_label = sum(index_label_counter[label] for label in true_label)

                    num_true_index_label = (
                        num_index_label - 1 if throw_first else num_index_label
                    )

                    mp_sample, _ = metrics.universal_mmp_at_k(
                        retrieval_top_k,
                        actual_top_k,
                        num_index_label,
                        num_true_index_label,
                        nearest,
                        true_label,
                        true_domain,
                        throw_first,
                    )
                    map_sample, _ = metrics.universal_map_at_k(
                        retrieval_top_k,
                        actual_top_k,
                        num_true_index_label,
                        nearest,
                        true_label,
                        true_domain,
                        throw_first,
                    )

                    cumulative_mmp += mp_sample
                    cumulative_map += map_sample

            results_dict[embed_type] = {
                "dimensionality": int(query_emb.shape[-1]),
                "mean_acc": float(np.round(num_knn_correct / num_query, 3)),
                "mean_mmp_at_k": float(np.round(cumulative_mmp / num_query, 3)),
                "mean_map_at_k": float(np.round(cumulative_map / num_query, 3)),
            }

        return results_dict

    def run_unified_knn(
        self,
        train_state,
        base_dir,
        dataset_names: str,
        batch_size: int,
        mode: str = "separate",
        disabled_knns: str = "",
        all_descriptors_dict=None,
        config=None,
    ):
        """Run unified KNN evaluation in separate or merged mode."""
        if all_descriptors_dict is None:
            all_descriptors_dict = {}

        if config is None:
            config = self.config

        disabled_knns_set = _parse_csv_set(disabled_knns)
        datasets_list = [name.strip() for name in dataset_names.split(",") if name.strip()]

        dataset = grain_datasets.get_knn_eval_datasets(
            self.config,
            base_dir,
            datasets_list,
            batch_size,
            disabled_knns=disabled_knns,
        )
        knn_info = dataset.knn_info
        formatted_total_results = {}

        for ds in datasets_list:
            ds_knn = knn_info.get("knn_setup", {}).get(ds)
            if ds_knn is None:
                logging.warning("No knn setup for dataset %s; skipping.", ds)
                continue

            ds_inference = all_descriptors_dict.get(ds, {})

            for knn_split, split_info in ds_knn.items():
                if knn_split in disabled_knns_set:
                    logging.info("Skipping disabled knn split %s for dataset %s.", knn_split, ds)
                    continue

                for part in [split_info["query"], split_info["index"]]:
                    lookup_key = grain_datasets.dataset_lookup_key(ds, part)

                    if (part not in ds_inference or ds_inference[part] is None) and train_state is not None:
                        logging.info("Extracting representation for %s", lookup_key)
                        domain_idx = config.knn_eval_names.split(",").index(ds)
                        ds_inference[part] = self._get_repr(
                            train_state,
                            knn_info[lookup_key],
                            domain_idx=domain_idx,
                        )
                    else:
                        logging.info("Descriptors already extracted for %s", lookup_key)

            all_descriptors_dict[ds] = ds_inference

        if self.extract_only_descriptors:
            return formatted_total_results, all_descriptors_dict

        for ds in datasets_list:
            ds_knn = knn_info.get("knn_setup", {}).get(ds)
            if ds_knn is None:
                continue

            for knn_split, split_info in ds_knn.items():
                if knn_split in disabled_knns_set:
                    logging.info("Skipping disabled knn split %s for dataset %s.", knn_split, ds)
                    continue

                query_part = split_info["query"]
                index_part = split_info["index"]
                throw_first = query_part == index_part

                lookup_key_query = grain_datasets.dataset_lookup_key(ds, query_part)
                query_data = knn_info["json_data"][lookup_key_query]

                query_labels = query_data["labels"]
                query_domains = query_data["domains"]
                query_paths = query_data["paths"]

                query_repr = all_descriptors_dict[ds].get(query_part)
                if query_repr is None:
                    logging.warning(
                        "Missing query representation for %s in dataset %s",
                        query_part,
                        ds,
                    )
                    continue

                if mode == "separate":
                    lookup_key_index = grain_datasets.dataset_lookup_key(ds, index_part)
                    index_data = knn_info["json_data"][lookup_key_index]

                    index_labels = index_data["labels"]
                    index_domains = index_data["domains"]
                    index_paths = index_data["paths"]

                    index_repr = all_descriptors_dict[ds].get(index_part)
                    if index_repr is None:
                        logging.warning(
                            "Missing index representation for %s in dataset %s",
                            index_part,
                            ds,
                        )
                        continue
                else:
                    merged_index_repr = None
                    merged_index_labels = None
                    merged_index_domains = None
                    merged_index_paths = None

                    for ds2 in datasets_list:
                        ds2_knn = knn_info.get("knn_setup", {}).get(ds2)
                        if ds2_knn is None or knn_split not in ds2_knn:
                            continue

                        split_info_ds2 = ds2_knn[knn_split]
                        lookup_key_index2 = grain_datasets.dataset_lookup_key(
                            ds2,
                            split_info_ds2["index"],
                        )
                        data2 = knn_info["json_data"][lookup_key_index2]

                        labels2 = data2["labels"]
                        domains2 = data2["domains"]
                        paths2 = data2["paths"]

                        ds2_inference = all_descriptors_dict.get(ds2, {})
                        repr2 = ds2_inference.get(split_info_ds2["index"])
                        if repr2 is None:
                            continue

                        repr2 = copy.deepcopy(repr2)

                        if merged_index_repr is None:
                            merged_index_repr = repr2
                            merged_index_labels = labels2
                            merged_index_domains = domains2
                            merged_index_paths = paths2
                        else:
                            for emb_key in merged_index_repr:
                                merged_index_repr[emb_key] = np.concatenate(
                                    (merged_index_repr[emb_key], repr2[emb_key]),
                                    axis=0,
                                )
                            merged_index_labels = np.concatenate((merged_index_labels, labels2), axis=0)
                            merged_index_domains = np.concatenate((merged_index_domains, domains2), axis=0)
                            merged_index_paths = np.concatenate((merged_index_paths, paths2), axis=0)

                    index_repr = merged_index_repr
                    index_labels = merged_index_labels
                    index_domains = merged_index_domains
                    index_paths = merged_index_paths

                    if index_repr is None:
                        logging.warning("Merged index representation is missing for %s / %s", ds, knn_split)
                        continue

                results_dict = self.compute_knn_metrics_fun(
                    lookup_key_query,
                    query_repr,
                    index_repr,
                    query_paths,
                    index_paths,
                    throw_first,
                    dataset.meta_data["top_k"],
                    config,
                    query_labels,
                    index_labels,
                    query_domains,
                    index_domains,
                    embed_types=[et.strip() for et in config.embedd_to_eval.split(",") if et.strip()],
                )

                formatted_temp = self.format_results(
                    results_dict,
                    config,
                    ds,
                    knn_split,
                    dataset,
                    separate=(mode == "separate"),
                )
                formatted_total_results = merge(formatted_total_results, formatted_temp)

        average_results = self.average_datasets(formatted_total_results)
        formatted_total_results = merge(formatted_total_results, average_results)

        return formatted_total_results, all_descriptors_dict

    def format_results(
        self,
        results_dict: Dict[str, Dict[str, Any]],
        config: Any,
        dataset_name: str,
        knn_name: str,
        dataset: Any,
        separate: bool,
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Format evaluation results into the project result schema."""
        keyword = ":separate:" if separate else ":common:"
        knn_results = {}
        mp_results = {}
        map_results = {}
        dimensionality = {}

        embed_types = [et.strip() for et in config.embedd_to_eval.split(",") if et.strip()]

        for embed_type in embed_types:
            top_key = f"{dataset_name}{keyword}{knn_name}:top_1"
            mp_key = f"{dataset_name}{keyword}{knn_name}:mp_{dataset.meta_data['top_k']}"
            map_key = f"{dataset_name}{keyword}{knn_name}:map_{dataset.meta_data['top_k']}"
            dim_key = f"{dataset_name}{keyword}{knn_name}"

            knn_results.setdefault(embed_type, {})[top_key] = results_dict[embed_type]["mean_acc"]
            mp_results.setdefault(embed_type, {})[mp_key] = results_dict[embed_type]["mean_mmp_at_k"]
            map_results.setdefault(embed_type, {})[map_key] = results_dict[embed_type]["mean_map_at_k"]
            dimensionality.setdefault(embed_type, {})[dim_key] = results_dict[embed_type]["dimensionality"]

        return {
            "knn_results": knn_results,
            "mp_results": mp_results,
            "map_results": map_results,
            "dimensionality": dimensionality,
        }

    def average_datasets(self, results: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Average metric values across datasets for each split and embed type."""
        new_results = {}
        knn_splits = ["train_knn", "val_knn", "test_knn"]

        for metric, metric_dict in results.items():
            new_results.setdefault(metric, {})
            for embed_type, embed_dict in metric_dict.items():
                new_results[metric].setdefault(embed_type, {})
                for knn_split in knn_splits:
                    values = [value for key, value in embed_dict.items() if knn_split in key]
                    if not values:
                        continue
                    sample_key = next(key for key in embed_dict if knn_split in key)
                    parts = sample_key.split(":")
                    new_key = "average:" + ":".join(parts[1:])
                    new_results[metric][embed_type][new_key] = np.round(np.mean(values), 3)
        return new_results

    def log_knn_summary(
        self,
        writer: metric_writers.MetricWriter,
        step,
        results,
        final_eval: bool = False,
    ):
        """Log KNN scalar summaries."""
        scalars = {}

        for embed_type, embed_type_result in results["knn_results"].items():
            for knn_name, result in embed_type_result.items():
                scalars[f"knn/{embed_type}/{knn_name}"] = result

        for embed_type, embed_type_result in results["mp_results"].items():
            for mp_name, result in embed_type_result.items():
                scalars[f"mp/{embed_type}/{mp_name}"] = result

        for embed_type, embed_type_result in results["dimensionality"].items():
            for dim_name, result in embed_type_result.items():
                scalars[f"dimensionality/{embed_type}/{dim_name}"] = result

        for embed_type, embed_type_result in results["map_results"].items():
            for map_name, result in embed_type_result.items():
                scalars[f"map_results/{embed_type}/{map_name}"] = result

        if final_eval:
            scalars = {f"final_eval/{k}": v for k, v in scalars.items()}

        writer.write_scalars(step, scalars)

    @staticmethod
    def split_and_pad(array):
        """Split an array across local devices, pad, and stack."""
        num_devices = jax.local_device_count()
        list_of_arrays = np.array_split(np.asarray(array), num_devices)
        list_of_arrays = [jnp.asarray(arr) for arr in list_of_arrays]

        max_len = max(arr.shape[0] for arr in list_of_arrays)
        padded_arrays = []
        masks = []

        for subarray in list_of_arrays:
            cur_len = subarray.shape[0]
            mask = jnp.ones(cur_len, dtype=bool)
            pad_len = max_len - cur_len

            if pad_len > 0:
                padded_subarray = jnp.pad(subarray, ((0, pad_len), (0, 0)))
                padded_mask = jnp.pad(mask, (0, pad_len))
            else:
                padded_subarray = subarray
                padded_mask = mask

            padded_arrays.append(padded_subarray)
            masks.append(padded_mask)

        return jnp.stack(padded_arrays), jnp.stack(masks)


def load_split_descrs_func(descr_base_dir):
    with gfile.GFile(descr_base_dir, "rb") as f:
        return np.load(f, allow_pickle=True)


def knn_step(
    knn_evaluator,
    train_state,
    config,
    train_dir,
    step,
    writer,
    load_descrs: bool = True,
    final_eval: bool = False,
):
    knn_dataset_names = [name.strip() for name in config.knn_eval_names.split(",") if name.strip()]
    embedds_to_eval = [name.strip() for name in config.embedd_to_eval.split(",") if name.strip()]

    descr_root = config.descr_save_path if config.descr_save_path is not None else train_dir
    descr_base_dir = os.path.join(descr_root, "descriptors", str(step))

    all_descriptors_dict = {}

    disabled_separate = _parse_csv_set(config.disabled_separate_knns)
    disabled_merged = _parse_csv_set(config.disabled_merged_knns)
    potentially_used_splits = disabled_separate.symmetric_difference(disabled_separate.union(disabled_merged))
    del potentially_used_splits  # left here only to reflect the original logic path

    for embedd_type in embedds_to_eval:
        embedd_type_descr_base_dir = os.path.join(descr_base_dir, embedd_type)

        for dataset in knn_dataset_names:
            all_descriptors_dict.setdefault(dataset, {})

            for split in ["train", "val", "test"]:
                split_name = f"{split}_knn"
                if split_name in disabled_separate and split_name in disabled_merged:
                    continue

                split_info = dataset_infos.DATASET_INFO[dataset]["knn"][split_name]
                query_part = split_info["query"]
                index_part = split_info["index"]

                all_descriptors_dict[dataset].setdefault(query_part, {})
                query_path = os.path.join(
                    embedd_type_descr_base_dir,
                    dataset,
                    f"{query_part}.npy",
                )

                if gfile.exists(query_path) and load_descrs:
                    logging.info(
                        "Loading descriptors for %s / %s / %s / %s",
                        dataset,
                        split,
                        query_part,
                        embedd_type,
                    )
                    all_descriptors_dict[dataset][query_part][embedd_type] = np.asarray(
                        load_split_descrs_func(query_path)
                    )
                else:
                    all_descriptors_dict[dataset][query_part] = None

                if index_part != query_part:
                    all_descriptors_dict[dataset].setdefault(index_part, {})
                    index_path = os.path.join(
                        embedd_type_descr_base_dir,
                        dataset,
                        f"{index_part}.npy",
                    )

                    if gfile.exists(index_path) and load_descrs:
                        logging.info(
                            "Loading descriptors for %s / %s / %s / %s",
                            dataset,
                            split,
                            index_part,
                            embedd_type,
                        )
                        all_descriptors_dict[dataset][index_part][embedd_type] = np.asarray(
                            load_split_descrs_func(index_path)
                        )
                    else:
                        all_descriptors_dict[dataset][index_part] = None

    knn_datasets_dir = config.eval_dataset_dir
    knn_dataset_names_csv = ",".join(knn_dataset_names)

    logging.info("Running KNN evals using separate database.")
    results, all_descriptors_dict = knn_evaluator.run_unified_knn(
        train_state,
        knn_datasets_dir,
        knn_dataset_names_csv,
        config.get("eval_batch_size", config.batch_size),
        "separate",
        config.get("disabled_separate_knns", ""),
        all_descriptors_dict,
        config,
    )

    logging.info(
        "Running KNN evals using common database made of %s.",
        knn_dataset_names_csv,
    )
    merged_results, all_descriptors_dict = knn_evaluator.run_unified_knn(
        train_state,
        knn_datasets_dir,
        knn_dataset_names_csv,
        config.get("eval_batch_size", config.batch_size),
        "merged",
        config.get("disabled_merged_knns", ""),
        all_descriptors_dict,
        config,
    )

    results = merge(dict(results), merged_results)

    if not config.extract_only_descrs:
        logging.info("Step %s KNN results: %s", step, results)

        if config.write_summary:
            if config.universal_embedding_is is not None:
                results = create_universal_embedding_entry(results, config)

            knn_evaluator.log_knn_summary(
                writer=writer,
                step=step,
                results=results,
                final_eval=final_eval,
            )

    if config.save_descriptors:
        for dataset in all_descriptors_dict:
            for split in all_descriptors_dict[dataset]:
                split_entry = all_descriptors_dict[dataset][split]
                if split_entry is None:
                    continue

                for embed_type in split_entry:
                    if embed_type not in embedds_to_eval:
                        continue

                    descr_to_save = split_entry[embed_type]
                    descr_save_path = os.path.join(
                        descr_base_dir,
                        embed_type,
                        dataset,
                        f"{split}.npy",
                    )
                    gfile.makedirs(os.path.dirname(descr_save_path))

                    with gfile.GFile(descr_save_path, "wb") as f:
                        np.save(f, descr_to_save)
                    logging.info("Descriptors file complete: %s", descr_save_path)

    return results


def create_universal_embedding_entry(
    results_dict,
    config,
):
    results_dict["knn_results"]["universal"] = results_dict["knn_results"][config.universal_embedding_is]
    results_dict["mp_results"]["universal"] = results_dict["mp_results"][config.universal_embedding_is]
    results_dict["map_results"]["universal"] = results_dict["map_results"][config.universal_embedding_is]
    results_dict["dimensionality"]["universal"] = results_dict["dimensionality"][config.universal_embedding_is]
    return results_dict


def merge(a: dict, b: dict, path=None):
    if path is None:
        path = []

    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge(a[key], b[key], path + [str(key)])
            elif a[key] != b[key]:
                raise ValueError("Conflict at " + ".".join(path + [str(key)]))
        else:
            a[key] = b[key]
    return a