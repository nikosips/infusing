import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Union

import tensorflow as tf
from array_record.python.array_record_module import ArrayRecordWriter


DOMAINS = {
    "cars": 0,
    "sop": 1,
    "inshop": 2,
    "inat": 3,
    "food2k": 6,
    "imagenet": 8,
}


def load_json_info(info_file: str) -> List[Dict[str, Union[str, int, List[int]]]]:
    with open(info_file, "r") as infile:
        return json.load(infile)


def build_info_from_image_folders(base_dir: str) -> List[Dict[str, Union[str, int]]]:
    """Builds split metadata from a class-per-folder image directory."""
    info_data = []

    classes = sorted(
        d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))
    )
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    for cls in classes:
        cls_dir = os.path.join(base_dir, cls)
        for filename in sorted(os.listdir(cls_dir)):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                info_data.append(
                    {
                        "path": os.path.join(cls, filename),
                        "class_id": class_to_idx[cls],
                    }
                )

    return info_data


def maybe_subset_info_data(
    info_data: List[Dict[str, Union[str, int, List[int]]]],
    *,
    split: str,
    percentage: int | None,
    shuffle: bool,
    seed: int,
) -> List[Dict[str, Union[str, int, List[int]]]]:
    if shuffle:
        random.seed(seed)
        random.shuffle(info_data)

    if percentage is None:
        return info_data

    if not 0 <= percentage <= 100:
        raise ValueError("percentage must be between 0 and 100.")

    subset_size = len(info_data) * percentage // 100
    if split == "val":
        return info_data[-subset_size:]
    return info_data[:subset_size]


def create_example(
    *,
    index: int,
    image_file: str,
    label: Union[int, str, List[int], List[str]],
    files_dir: str,
    domain: str,
) -> tf.train.Example:
    if not isinstance(label, list):
        label = [label]

    image_path = os.path.join(files_dir, image_file)
    feature = {
        "index": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
        "domain": tf.train.Feature(
            int64_list=tf.train.Int64List(value=[DOMAINS[domain]])
        ),
        "image_bytes": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[tf.io.read_file(image_path).numpy()])
        ),
        "class_id": tf.train.Feature(int64_list=tf.train.Int64List(value=label)),
        "key": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_file.encode()])),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def write_array_record_shards(
    *,
    info_data: List[Dict[str, Union[str, int, List[int]]]],
    output_file: str,
    files_dir: str,
    num_shards: int,
    domain: str,
) -> None:
    writers = [
        ArrayRecordWriter(
            f"{output_file}-{i:05d}-of-{num_shards:05d}",
            "group_size:1",
        )
        for i in range(num_shards)
    ]

    num_examples = len(info_data)
    examples_per_shard = num_examples // num_shards
    remainder = num_examples % num_shards
    shard_counts = [
        examples_per_shard + (1 if i < remainder else 0)
        for i in range(num_shards)
    ]

    shard_index = 0
    shard_limit = shard_counts[0] if shard_counts else 0

    for i, file_info in enumerate(info_data):
        if i >= shard_limit and shard_index < num_shards - 1:
            shard_index += 1
            shard_limit += shard_counts[shard_index]

        print(f"Processing: {file_info['path']}")
        example = create_example(
            index=i,
            image_file=file_info["path"],
            label=file_info["class_id"],
            files_dir=files_dir,
            domain=domain,
        )
        writers[shard_index].write(example.SerializeToString())

    for writer in writers:
        writer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert image datasets into ArrayRecord shards.",
    )
    parser.add_argument(
        "--input_format",
        choices=("json", "folder"),
        default="json",
        help="Use 'json' for split metadata files or 'folder' for class-per-folder scans.",
    )
    parser.add_argument(
        "--info_file",
        help="Path to JSON split metadata. Required when --input_format=json.",
    )
    parser.add_argument(
        "--base_dir",
        help="Base directory of class-per-folder images. Required when --input_format=folder.",
    )
    parser.add_argument(
        "--files_dir",
        required=True,
        help="Root directory used to resolve relative image paths.",
    )
    parser.add_argument(
        "--output_file",
        required=True,
        help="Output ArrayRecord path prefix.",
    )
    parser.add_argument(
        "--domain",
        required=True,
        choices=sorted(DOMAINS),
        help="Dataset domain identifier.",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        required=True,
        help="Number of ArrayRecord shards to write.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Split name used when subsetting folder-based inputs.",
    )
    parser.add_argument(
        "--percentage",
        type=int,
        help="Keep only this percentage of folder-based inputs after optional shuffling.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle folder-based inputs before applying --percentage.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used with --shuffle.",
    )
    parser.add_argument(
        "--save_info_json",
        action="store_true",
        help="Save the resolved input metadata next to the output prefix.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)

    if args.input_format == "json":
        if not args.info_file:
            raise ValueError("--info_file is required when --input_format=json.")
        info_data = load_json_info(args.info_file)
    else:
        if not args.base_dir:
            raise ValueError("--base_dir is required when --input_format=folder.")
        info_data = build_info_from_image_folders(args.base_dir)
        info_data = maybe_subset_info_data(
            info_data,
            split=args.split,
            percentage=args.percentage,
            shuffle=args.shuffle,
            seed=args.seed,
        )

    if args.save_info_json:
        info_data_json_path = os.path.splitext(args.output_file)[0] + "_info_data.json"
        with open(info_data_json_path, "w") as json_file:
            json.dump(info_data, json_file, indent=2)
        print(f"Info data saved to {info_data_json_path}")

    print(f"Output File: {args.output_file}")
    print(f"Files Directory: {args.files_dir}")
    print(f"Examples: {len(info_data)}")

    write_array_record_shards(
        info_data=info_data,
        output_file=args.output_file,
        files_dir=args.files_dir,
        num_shards=args.num_shards,
        domain=args.domain,
    )


if __name__ == "__main__":
    main()
