#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${ROOT_DIR}/data"
BASE_URL="https://login.rci.cvut.cz/~ypsilnik/infusing_data"

download_file() {
  local relative_path="$1"
  local destination="${DATA_DIR}/${relative_path}"

  mkdir -p "$(dirname "${destination}")"
  echo "Downloading ${relative_path}"
  curl -L --fail --show-error "${BASE_URL}/${relative_path}" -o "${destination}"
}

download_sharded_series() {
  local prefix="$1"
  local total_shards="$2"

  for ((i=0; i<total_shards; i++)); do
    printf -v shard_path "%s-%05d-of-%05d" "${prefix}" "${i}" "${total_shards}"
    download_file "${shard_path}"
  done
}

# Public checkpoints
download_file "models/siglip/webli_en_b16_224_63724782.npz"
download_file "models/tips/tips_oss_b14_highres_distilled_vision.npz"

# Public dataset metadata
download_file "info_files/cars/train.json"
download_file "info_files/cars/val.json"
download_file "info_files/cars/test.json"

download_file "info_files/food2k/train.json"
download_file "info_files/food2k/val.json"
download_file "info_files/food2k/test.json"

download_file "info_files/imagenet/train.json"
download_file "info_files/imagenet/val.json"
download_file "info_files/imagenet/test.json"

download_file "info_files/inat/train.json"
download_file "info_files/inat/val.json"
download_file "info_files/inat/test.json"

download_file "info_files/inshop/train.json"
download_file "info_files/inshop/val.json"
download_file "info_files/inshop/test.json"
download_file "info_files/inshop/index.json"

download_file "info_files/laion/train.json"
download_file "info_files/laion/val.json"

download_file "info_files/our_imagenet_split/train.json"
download_file "info_files/our_imagenet_split/val.json"

download_file "info_files/sop/train.json"
download_file "info_files/sop/val.json"
download_file "info_files/sop/test.json"

# Public ArrayRecord shards
download_file "array_records/cars/train/cars.train.array_record-00000-of-00001"
download_file "array_records/cars/val/cars.val.array_record-00000-of-00001"
download_file "array_records/cars/test/cars.test.array_record-00000-of-00001"

download_sharded_series "array_records/sop/train/sop.train.array_record" 3
download_sharded_series "array_records/sop/test/sop.test.array_record" 3
download_file "array_records/sop/val/sop.val.array_record-00000-of-00001"

download_sharded_series "array_records/our_imagenet_split/train/imagenet.train.array_record" 120
download_file "array_records/our_imagenet_split/train/imagenet.train_info_data.json"
download_sharded_series "array_records/our_imagenet_split/val/imagenet.val.array_record" 30
download_file "array_records/our_imagenet_split/val/imagenet.val_info_data.json"

# Text-image evaluation assets
download_file "text_image/flickr30k/queries.tfrecord"
download_file "text_image/flickr30k/index.tfrecord"
download_file "text_image/flickr30k/gt.tfrecord"
download_file "text_image/flickr30k/flickr30k_gt.npy"
download_file "text_image/flickr30k/flickr30k_text_embeddings_siglip.npy"
download_file "text_image/flickr30k/flickr30k_text_embeddings_tips.npy"
download_file "text_image/flickr30k/flickr30k_text_index.npy"

download_file "text_image/mscoco/queries.tfrecord"
download_file "text_image/mscoco/index.tfrecord"
download_file "text_image/mscoco/gt.tfrecord"
download_file "text_image/mscoco/mscoco_gt.npy"
download_file "text_image/mscoco/mscoco_text_embeddings_siglip.npy"
download_file "text_image/mscoco/mscoco_text_embeddings_tips.npy"
download_file "text_image/mscoco/mscoco_text_index.npy"

cat <<EOF
Done.

Downloaded public assets from:
  ${BASE_URL}

Placed under:
  ${DATA_DIR}

This script currently downloads:
  - model checkpoints
  - dataset info files
  - published ArrayRecord shards
  - text-image evaluation assets

This script does not download:
  - precomputed descriptor features
EOF
