"""Dataset builders for universal embedding training and KNN evaluation."""

import collections
import functools
import glob
import json
import os
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Union

from absl import logging
from flax import jax_utils
import grain.python as grain
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from scenic.dataset_lib import dataset_utils
import tensorflow as tf
from tensorflow.io import gfile

import universal_embedding.info_utils as info_utils
from universal_embedding import subset_sampler
from universal_embedding.dataset_infos import DATASET_INFO


tf.config.experimental.set_visible_devices([], "GPU")

PRNGKey = jnp.ndarray


UniversalEmbeddingTrainingDataset = collections.namedtuple(
    "TrainingDataset",
    ["train_iter", "meta_data"],
)

UniversalEmbeddingKnnEvalDataset = collections.namedtuple(
    "KnnDataset",
    ["knn_info", "meta_data"],
)

ExtractDataset = collections.namedtuple(
    "ExtractDataset",
    ["extract_iter", "meta_data"],
)


def _get_model_configs(config):
    model_class = config.model_class.lower()
    if "siglip" in model_class:
        return info_utils.SigLIP_ViT_configs
    if "tips" in model_class:
        return info_utils.TIPS_ViT_configs
    raise ValueError(
        f"Unsupported model_class '{config.model_class}'. "
        "Expected a model class containing 'siglip' or 'tips'."
    )


def _get_model_preprocess_config(config):
    model_cfgs = _get_model_configs(config)
    model_cfg = model_cfgs[config.model_type]
    return {
        "normalization_statistics": model_cfg["normalization_statistics"],
        "image_size": model_cfg.get("image_size"),
        "image_resize": model_cfg.get("image_resize"),
    }


def _parse_disabled_knns(disabled_knns: str) -> set[str]:
    return {item.strip() for item in disabled_knns.split(",") if item.strip()}


def _dataset_lookup_key(dataset_name: str, split: str) -> str:
    return f"{dataset_name}:{split}"


def _load_split_json(info_files_dir: str, dataset_name: str, split: str) -> Dict[str, List]:
    json_path = os.path.join(info_files_dir, dataset_name, f"{split}.json")
    with gfile.GFile(json_path, "r") as f:
        split_info = json.load(f)

    labels = [sample["class_id"] for sample in split_info]
    paths = [sample["path"] for sample in split_info]
    domains = [DATASET_INFO[dataset_name]["domain"]] * len(labels)

    return {
        "paths": paths,
        "labels": labels,
        "domains": domains,
    }


def _normalize_image(image: tf.Tensor, normalization_statistics: Dict[str, Sequence[float]]) -> tf.Tensor:
    image = tf.cast(image, tf.float32) / 255.0
    image -= tf.constant(normalization_statistics["MEAN_RGB"], shape=[1, 1, 3], dtype=tf.float32)
    image /= tf.constant(normalization_statistics["STDDEV_RGB"], shape=[1, 1, 3], dtype=tf.float32)
    return image


def _resize(image: tf.Tensor, image_size: int) -> tf.Tensor:
    return tf.image.resize(
        image,
        [image_size, image_size],
        method=tf.image.ResizeMethod.BILINEAR,
    )


def _resize_smaller_edge(image: tf.Tensor, image_size: int):
    shape = tf.shape(image)
    height, width = shape[0], shape[1]

    if height <= width:
        new_height = image_size
        new_width = tf.cast((width / height) * image_size, tf.int32)
    else:
        new_width = image_size
        new_height = tf.cast((height / width) * image_size, tf.int32)

    resized = tf.image.resize(
        image,
        [new_height, new_width],
        method=tf.image.ResizeMethod.BILINEAR,
    )
    return resized, (new_height, new_width)


def _process_train_image(image: tf.Tensor, image_resize: int, image_size: int) -> tf.Tensor:
    image = _resize(image, image_resize)
    image = tf.image.random_crop(image, [image_size, image_size, 3])
    image = tf.image.random_flip_left_right(image)
    return image


def _process_eval_image(image: tf.Tensor, image_size: int) -> tf.Tensor:
    image, new_size = _resize_smaller_edge(image, image_size)
    h, w = new_size

    if h > w:
        image = tf.image.crop_to_bounding_box(image, (h - w) // 2, 0, w, w)
    else:
        image = tf.image.crop_to_bounding_box(image, 0, (w - h) // 2, h, h)

    return image


def _parse_example(serialized_example: bytes, include_domain: bool = True):
    features_spec = {
        "image_bytes": tf.io.FixedLenFeature([], tf.string),
        "key": tf.io.FixedLenFeature([], tf.string),
        "index": tf.io.FixedLenFeature([], tf.int64),
    }
    if include_domain:
        features_spec["domain"] = tf.io.FixedLenFeature([], tf.int64)
        features_spec["class_id"] = tf.io.FixedLenFeature([], tf.int64)

    return tf.io.parse_single_example(serialized_example, features=features_spec)


class PreprocessExample(grain.MapTransform):
    """Grain map transform for train/eval image examples."""

    def __init__(
        self,
        *,
        is_train: bool,
        domain: int,
        domain_idx: int,
        normalization_statistics,
        image_size: int,
        image_resize: Optional[int] = None,
        dtype=tf.float32,
        label_offset: int = 0,
    ):
        self.is_train = is_train
        self.domain = domain
        self.domain_idx = domain_idx
        self.normalization_statistics = normalization_statistics
        self.image_size = image_size
        self.image_resize = image_resize
        self.dtype = dtype
        self.label_offset = label_offset

    def _process_one(self, example):
        features = _parse_example(example, include_domain=self.is_train)

        image = tf.io.decode_jpeg(features["image_bytes"], channels=3)

        if self.is_train:
            image = _process_train_image(
                image,
                image_resize=self.image_resize,
                image_size=self.image_size,
            )
        else:
            image = _process_eval_image(image, image_size=self.image_size)

        image = _normalize_image(image, self.normalization_statistics)
        image = tf.cast(image, self.dtype)

        output = {
            "inputs": image,
            "index": features["index"],
            "domain_idx": self.domain_idx,
        }

        if self.is_train:
            output["label"] = features["class_id"] + self.label_offset
            output["domain"] = features["domain"]
        else:
            output["domain"] = self.domain

        return output

    def map(self, example):
        return self._process_one(example)


def _create_arrayrecord_source(base_dir: str, dataset_name: str, split: str):
    dataset_info = DATASET_INFO[dataset_name]
    file_name = f"{split}_files"
    path = os.path.join(base_dir, dataset_info[file_name]).replace("tfrecord", "array_record")

    files = sorted(glob.glob(path + "*"))
    if not files:
        raise FileNotFoundError(f"No array_record files found for {dataset_name} split {split} at {path}*")

    return grain.ArrayRecordDataSource(files)


def _make_sampler(
    *,
    dataset_name: str,
    num_records: int,
    is_knn: bool,
    config,
    domain_idx: int,
    data_rng: Optional[int],
    selected_indices=None,
):
    host_index = jax.process_index()
    host_count = jax.process_count()

    if is_knn:
        if dataset_name in {"our_imagenet_split", "imagenet"} and "val" in {"val"}:
            pass  # kept behavior elsewhere instead of here

        return grain.SequentialSampler(
            num_records=num_records,
            shard_options=grain.ShardOptions(
                shard_index=host_index,
                shard_count=host_count,
                drop_remainder=False,
            ),
        )

    if data_rng is None:
        raise ValueError("data_rng must be provided for training dataloaders.")

    seed = int(data_rng) + domain_idx

    if dataset_name in {"imagenet", "our_imagenet_split"}:
        fraction = 1.0
        return subset_sampler.SubsetSampler(
            num_records=num_records,
            num_epochs=None,
            fraction=fraction,
            shard_options=grain.ShardOptions(
                shard_index=host_index,
                shard_count=host_count,
                drop_remainder=True,
            ),
            shuffle=True,
            seed=seed,
            selected_indices=selected_indices,
        )

    if dataset_name == "laion":
        fraction = 0.3
        return subset_sampler.SubsetSampler(
            num_records=num_records,
            num_epochs=None,
            fraction=fraction,
            shard_options=grain.ShardOptions(
                shard_index=host_index,
                shard_count=host_count,
                drop_remainder=True,
            ),
            shuffle=True,
            seed=seed,
            selected_indices=selected_indices,
        )

    return grain.IndexSampler(
        num_records=num_records,
        num_epochs=None,
        shard_options=grain.ShardOptions(
            shard_index=host_index,
            shard_count=host_count,
            drop_remainder=True,
        ),
        shuffle=True,
        seed=seed,
    )


def _create_tfrecord_extract_source(base_dir: Union[str, Sequence[str]]):
    """Creates a TFRecordDataset for extraction.

    Args:
        base_dir: A TFRecord file path, glob pattern, or a sequence of file paths.

    Returns:
        tf.data.TFRecordDataset
    """
    if isinstance(base_dir, (list, tuple)):
        files = list(base_dir)
    else:
        # Allow either a direct file path or a glob pattern.
        matched = sorted(glob.glob(base_dir))
        files = matched if matched else [base_dir]

    if not files:
        raise FileNotFoundError(f"No TFRecord files found for extract dataset: {base_dir}")

    return tf.data.TFRecordDataset(files)


def _parse_extract_example(serialized_example: tf.Tensor):
    """Parses a TFRecord example for extraction."""
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            "bytes_image": tf.io.FixedLenFeature([], tf.string),
            "id": tf.io.VarLenFeature(tf.string),
        },
    )

    image = tf.io.decode_jpeg(features["bytes_image"], channels=3)

    id_dense = tf.sparse.to_dense(features["id"], default_value="")
    sample_id = id_dense[0]

    return image, sample_id


class PreprocessExtractExample:
    """TF data map transform for extraction examples."""

    def __init__(
        self,
        *,
        normalization_statistics,
        image_size: int,
        dtype=tf.float32,
    ):
        self.normalization_statistics = normalization_statistics
        self.image_size = image_size
        self.dtype = dtype

    def __call__(self, serialized_example):
        image, sample_id = _parse_extract_example(serialized_example)

        image = _process_eval_image(image, image_size=self.image_size)
        image = _normalize_image(image, self.normalization_statistics)
        image = tf.cast(image, self.dtype)

        return {
            "inputs": image,
            # "id": sample_id,
        }


def _collect_extract_ids(base_dir: Union[str, Sequence[str]]) -> List[str]:
    """Reads TFRecord files once to collect ids in dataset order."""
    raw_ds = _create_tfrecord_extract_source(base_dir)
    ids = []

    for raw_record in raw_ds:
        features = tf.io.parse_single_example(
            raw_record,
            features={
                "bytes_image": tf.io.FixedLenFeature([], tf.string),
                "id": tf.io.VarLenFeature(tf.string),
            },
        )
        id_dense = tf.sparse.to_dense(features["id"], default_value="")
        ids.append(id_dense[0].numpy().decode("utf-8"))

    return ids


def create_one_dataloader(
    *,
    base_dir: str,
    dataset_name: str,
    split: str,
    config,
    total_classes: int,
    domain_idx: int,
    is_knn: bool,
    label_offset: int = 0,
    data_rng: Optional[int] = None,
    selected_indices=None,
):
    del total_classes  # currently unused

    dataset = _create_arrayrecord_source(base_dir, dataset_name, split)
    json_data = _load_split_json(config.info_files_dir, dataset_name, split)

    preprocess_cfg = _get_model_preprocess_config(config)

    num_records = len(dataset)
    if is_knn and dataset_name in {"our_imagenet_split", "imagenet"} and split == "val":
        num_records = int(num_records * 0.1)

    sampler = _make_sampler(
        dataset_name=dataset_name,
        num_records=num_records,
        is_knn=is_knn,
        config=config,
        domain_idx=domain_idx,
        data_rng=data_rng,
        selected_indices=selected_indices,
    )

    transform = PreprocessExample(
        is_train=not is_knn,
        domain=DATASET_INFO[dataset_name]["domain"],
        domain_idx=domain_idx,
        normalization_statistics=preprocess_cfg["normalization_statistics"],
        image_size=preprocess_cfg["image_size"],
        image_resize=preprocess_cfg["image_resize"],
        label_offset=label_offset,
    )

    if is_knn:
        batch_size = config.eval_batch_size // jax.process_count()
        drop_remainder = False
    else:
        batch_size = config.batch_size // jax.process_count()
        drop_remainder = True

    dataloader = grain.DataLoader(
        data_source=dataset,
        operations=[
            transform,
            grain.Batch(batch_size=batch_size, drop_remainder=drop_remainder),
        ],
        sampler=sampler,
        worker_count=config.grain_worker_count,
        worker_buffer_size=config.worker_buffer_size,
    )

    return dataloader, json_data



def create_extract_dataloader(
    *,
    base_dir: Union[str, Sequence[str]],
    config,
):
    """Creates the TF extraction dataloader and id list."""
    preprocess_cfg = _get_model_preprocess_config(config)

    dataset = _create_tfrecord_extract_source(base_dir)
    ids = _collect_extract_ids(base_dir)

    transform = PreprocessExtractExample(
        normalization_statistics=preprocess_cfg["normalization_statistics"],
        image_size=preprocess_cfg["image_size"],
        dtype=tf.float32,
    )

    dataset = dataset.map(
        transform,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,  # keep order aligned with ids
    )

    return dataset, ids



def merge_two_first_dims(pytree):
    def _merge(array):
        return array.reshape((array.shape[0] * array.shape[1],) + array.shape[2:])
    return jax.tree_util.tree_map(_merge, pytree)


def build_ds_iter(
    ds,
    maybe_pad_batches,
    shard_batches,
    prefetch_buffer_size,
):
    ds_iter = iter(ds)
    ds_iter = map(maybe_pad_batches, ds_iter)
    ds_iter = map(shard_batches, ds_iter)
    ds_iter = jax_utils.prefetch_to_device(ds_iter, prefetch_buffer_size)

    return ds_iter


def build_ds_iter_tfrecord(
    ds,
    maybe_pad_batches,
    shard_batches,
    prefetch_buffer_size,
):
    """Build iterator for TF-based extract datasets."""
    ds_iter = iter(ds)
    ds_iter = map(dataset_utils.tf_to_numpy, ds_iter)
    ds_iter = map(maybe_pad_batches, ds_iter)
    ds_iter = map(shard_batches, ds_iter)
    ds_iter = jax_utils.prefetch_to_device(ds_iter, prefetch_buffer_size)
    return ds_iter


def build_universal_embedding_dataset(
    *,
    base_dir: str,
    dataset_names: Sequence[str],
    split: str,
    config,
    is_knn: bool,
    data_rng: Optional[int] = None,
    selected_indices=None,
):
    total_classes = sum(DATASET_INFO[name]["num_train_classes"] for name in dataset_names)

    ds_dict = {}
    metadata_ds_dict = {}
    offset = 0

    for i, dataset_name in enumerate(dataset_names):
        dataloader, json_data = create_one_dataloader(
            base_dir=base_dir,
            dataset_name=dataset_name,
            split=split,
            config=config,
            total_classes=total_classes,
            domain_idx=i,
            is_knn=is_knn,
            label_offset=offset,
            data_rng=data_rng,
            selected_indices=selected_indices,
        )

        if config.classifier == "joint":
            offset += DATASET_INFO[dataset_name]["num_train_classes"]

        ds_dict[dataset_name] = dataloader
        metadata_ds_dict[dataset_name] = json_data

    return ds_dict, metadata_ds_dict


def build_extract_dataset(
    *,
    base_dir: Union[str, Sequence[str]],
    batch_size: int,
    config,
    repeat: bool = False,
):
    """Builds a batched TF extract dataset."""
    dataset, ids = create_extract_dataloader(
        base_dir=base_dir,
        config=config,
    )

    dataset = dataset.batch(batch_size, drop_remainder=False)

    options = tf.data.Options()
    options.experimental_optimization.parallel_batch = True
    dataset = dataset.with_options(options)

    if repeat:
        dataset = dataset.repeat()

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset, ids


def get_training_dataset(
    config: ml_collections.ConfigDict,
    num_local_shards: Optional[int] = None,
    prefetch_buffer_size: int = 2,
    dataset_configs: Optional[ml_collections.ConfigDict] = None,
    data_rng=None,
    prototypes=None,
    selected_indices=None,
):
    del dataset_configs, prototypes

    device_count = jax.device_count()
    logging.info("device_count: %d", device_count)
    logging.info("num_hosts: %d", jax.process_count())
    logging.info("host_id: %d", jax.process_index())

    dataset_names = [name.strip() for name in config.dataset_name.split(",") if name.strip()]
    batch_size = config.batch_size

    if batch_size % device_count != 0:
        raise ValueError(
            f"Batch size ({batch_size}) must be divisible by number of devices ({device_count})."
        )

    local_batch_size = batch_size // jax.process_count()
    logging.info("local_batch_size: %d", local_batch_size)

    eval_batch_size = config.get("eval_batch_size", batch_size)
    local_eval_batch_size = eval_batch_size // jax.process_count()
    logging.info("local_eval_batch_size: %d", local_eval_batch_size)

    num_local_shards = num_local_shards or jax.local_device_count()

    maybe_pad_batches_train = functools.partial(
        dataset_utils.maybe_pad_batch,
        train=True,
        batch_size=local_batch_size,
    )
    shard_batches = functools.partial(dataset_utils.shard, n_devices=num_local_shards)

    train_ds_dict, train_metadata_ds_dict = build_universal_embedding_dataset(
        base_dir=config.train_dataset_dir,
        dataset_names=dataset_names,
        split="train",
        config=config,
        is_knn=False,
        data_rng=data_rng,
        selected_indices=selected_indices,
    )

    train_iter_dict = {
        ds_name: build_ds_iter(
            ds,
            maybe_pad_batches_train,
            shard_batches,
            prefetch_buffer_size,
        )
        for ds_name, ds in train_ds_dict.items()
    }

    preprocess_cfg = _get_model_preprocess_config(config)
    image_size = preprocess_cfg["image_size"]

    num_train_examples = 0
    num_classes = 0
    dataset_samples = OrderedDict()
    classes_per_dataset = OrderedDict()

    for name in dataset_names:
        num_train_examples += DATASET_INFO[name]["num_train_examples"]
        num_classes += DATASET_INFO[name]["num_train_classes"]
        dataset_samples[name] = DATASET_INFO[name]["num_train_examples"]
        classes_per_dataset[name] = DATASET_INFO[name]["num_train_classes"]

    meta_data = {
        "dataset_name": config.dataset_name,
        "domain_indices": [DATASET_INFO[name]["domain"] for name in dataset_names],
        "num_classes": num_classes,
        "input_shape": (-1, image_size, image_size, 3),
        "num_train_examples": num_train_examples,
        "input_dtype": getattr(jnp, config.data_dtype_str),
        "target_is_onehot": False,
        "dataset_samples": dataset_samples,
        "classes_per_dataset": classes_per_dataset,
        "labels": {name: train_metadata_ds_dict[name]["labels"] for name in dataset_names},
    }

    return UniversalEmbeddingTrainingDataset(
        train_iter_dict,
        meta_data,
    )


def dataset_lookup_key(dataset_name, split):
    return _dataset_lookup_key(dataset_name, split)


def get_knn_eval_datasets(
    config,
    base_dir,
    dataset_names: Union[List[str], str],
    eval_batch_size: int,
    disabled_knns: str = "",
    prefetch_buffer_size: int = 2,
):
    del base_dir

    if isinstance(dataset_names, str):
        dataset_names = [name.strip() for name in dataset_names.split(",") if name.strip()]

    device_count = jax.device_count()
    logging.info("device_count: %d", device_count)
    logging.info("num_hosts: %d", jax.process_count())
    logging.info("host_id: %d", jax.process_index())

    local_eval_batch_size = eval_batch_size // jax.process_count()
    logging.info("local_eval_batch_size: %d", local_eval_batch_size)

    num_local_shards = jax.local_device_count()
    disabled_knn_set = _parse_disabled_knns(disabled_knns)

    maybe_pad_batches_eval = functools.partial(
        dataset_utils.maybe_pad_batch,
        train=False,
        batch_size=local_eval_batch_size,
    )
    shard_batches = functools.partial(
        dataset_utils.shard,
        n_devices=num_local_shards,
    )

    knn_info = {"json_data": {}}
    knn_setup = {}
    size_info = {}

    for dataset_name in dataset_names:
        knn_splits = set()
        for val in DATASET_INFO[dataset_name]["knn"].values():
            knn_splits.add(val["query"])
            knn_splits.add(val["index"])

        for split in knn_splits:
            if f"{split}_knn" in disabled_knn_set:
                continue

            if config.get("descr_eval"):
                split_knn_iter = None
            else:
                split_knn_ds, _ = build_universal_embedding_dataset(
                    base_dir=config.eval_dataset_dir,
                    dataset_names=[dataset_name],
                    split=split,
                    config=config,
                    is_knn=True,
                )

                split_knn_iter = build_ds_iter(
                    split_knn_ds[dataset_name],
                    maybe_pad_batches_eval,
                    shard_batches,
                    prefetch_buffer_size,
                )

            knn_info[_dataset_lookup_key(dataset_name, split)] = split_knn_iter
            knn_info["json_data"][_dataset_lookup_key(dataset_name, split)] = _load_split_json(
                config.info_files_dir,
                dataset_name,
                split,
            )

        knn_setup[dataset_name] = DATASET_INFO[dataset_name]["knn"]
        size_info[dataset_name] = {
            "num_train_examples": DATASET_INFO[dataset_name]["num_train_examples"],
            "num_test_examples": DATASET_INFO[dataset_name].get("num_test_examples"),
            "num_val_examples": DATASET_INFO[dataset_name].get("num_val_examples"),
        }

    knn_info["knn_setup"] = knn_setup

    preprocess_cfg = _get_model_preprocess_config(config)
    image_size = preprocess_cfg["image_size"]

    meta_data = {
        "input_shape": (-1, image_size, image_size, 3),
        "dataset_names": ",".join(dataset_names),
        "dataset_name": ",".join(dataset_names),
        "top_k": int(config.top_k),
        "size_info": size_info,
        "num_classes": -1,
        "input_dtype": getattr(jnp, config.data_dtype_str),
    }

    return UniversalEmbeddingKnnEvalDataset(
        knn_info,
        meta_data,
    )


def get_extract_dataset(
    config: ml_collections.ConfigDict,
    base_dir: Union[str, Sequence[str]],
    eval_batch_size: int,
    prefetch_buffer_size: int = 2,
):
    """Returns generators for TFRecord extraction dataset."""

    device_count = jax.device_count()
    logging.info("device_count: %d", device_count)
    logging.info("num_hosts: %d", jax.process_count())
    logging.info("host_id: %d", jax.process_index())

    if eval_batch_size % device_count != 0:
        raise ValueError(
            f"Eval batch size ({eval_batch_size}) must be divisible by "
            f"number of devices ({device_count})."
        )

    local_eval_batch_size = eval_batch_size // jax.process_count()
    logging.info("local_eval_batch_size: %d", local_eval_batch_size)

    num_local_shards = jax.local_device_count()

    maybe_pad_batches_eval = functools.partial(
        dataset_utils.maybe_pad_batch,
        train=False,
        batch_size=local_eval_batch_size,
    )
    shard_batches = functools.partial(
        dataset_utils.shard,
        n_devices=num_local_shards,
    )

    extract_ds, ids = build_extract_dataset(
        base_dir=base_dir,
        batch_size=local_eval_batch_size,
        config=config,
        repeat=False,
    )

    extract_iter = build_ds_iter_tfrecord(
        extract_ds,
        maybe_pad_batches_eval,
        shard_batches,
        prefetch_buffer_size,
    )

    preprocess_cfg = _get_model_preprocess_config(config)
    image_size = preprocess_cfg["image_size"]

    meta_data = {
        "input_shape": (-1, image_size, image_size, 3),
        "num_classes": -1,
        "input_dtype": getattr(jnp, config.data_dtype_str),
        "ids": ids,
    }

    return ExtractDataset(
        extract_iter=extract_iter,
        meta_data=meta_data,
    )