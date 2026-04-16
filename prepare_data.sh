#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONVERTER="${ROOT_DIR}/convert_to_array_record.py"

INFO_DIR="${ROOT_DIR}/data/info_files"
IMAGE_DIR="${ROOT_DIR}/data/images"
ARRAY_RECORD_DIR="${ROOT_DIR}/data/array_records"

# Update this path for your local ImageNet installation when using the helper
# below.
IMAGENET_BASE_DIR=""

convert_from_json() {
  local dataset="$1"
  local split="$2"
  local num_shards="$3"

  python "${CONVERTER}" \
    --input_format=json \
    --info_file "${INFO_DIR}/${dataset}/${split}.json" \
    --files_dir "${IMAGE_DIR}" \
    --output_file "${ARRAY_RECORD_DIR}/${dataset}/${split}/${dataset}.${split}.array_record" \
    --num_shards "${num_shards}" \
    --domain "${dataset}"
}

convert_from_folders() {
  local output_dataset="$1"
  local split="$2"
  local files_dir="$3"
  local base_dir="$4"
  local num_shards="$5"
  local percentage="$6"

  python "${CONVERTER}" \
    --input_format=folder \
    --base_dir "${base_dir}" \
    --files_dir "${files_dir}" \
    --output_file "${ARRAY_RECORD_DIR}/${output_dataset}/${split}/imagenet.${split}.array_record" \
    --num_shards "${num_shards}" \
    --domain imagenet \
    --split "${split}" \
    --percentage "${percentage}" \
    --shuffle \
    --save_info_json
}

show_examples() {
  cat <<EOF
Examples:

  # Fine-grained datasets described by JSON metadata
  convert_from_json cars train 1
  convert_from_json cars val 1
  convert_from_json cars test 1

  convert_from_json food2k train 50
  convert_from_json food2k val 5
  convert_from_json food2k test 1

  convert_from_json inat train 60
  convert_from_json inat val 15
  convert_from_json inat test 30

  convert_from_json inshop train 1
  convert_from_json inshop val 1
  convert_from_json inshop test 1
  convert_from_json inshop index 1

  convert_from_json sop train 3
  convert_from_json sop val 1
  convert_from_json sop test 3

  # Our ImageNet split, built from class folders
  convert_from_folders our_imagenet_split train "${IMAGENET_BASE_DIR}/train" "${IMAGENET_BASE_DIR}/train" 120 80
  convert_from_folders our_imagenet_split val "${IMAGENET_BASE_DIR}/train" "${IMAGENET_BASE_DIR}/train" 30 20

Usage:

  1. Edit the directory variables at the top of this file if needed.
  2. Uncomment the conversions you want to run in the main() function.
  3. Run:
       bash prepare_data.sh
EOF
}

main() {
  # Uncomment the dataset conversions you want to run.

  # convert_from_json cars train 1
  # convert_from_json cars val 1
  # convert_from_json cars test 1

  # convert_from_json food2k train 50
  # convert_from_json food2k val 5
  # convert_from_json food2k test 1

  # convert_from_json inat train 60
  # convert_from_json inat val 15
  # convert_from_json inat test 30

  # convert_from_json inshop train 1
  # convert_from_json inshop val 1
  # convert_from_json inshop test 1
  # convert_from_json inshop index 1

  # convert_from_json sop train 3
  # convert_from_json sop val 1
  # convert_from_json sop test 3

  # convert_from_folders our_imagenet_split train "${IMAGENET_BASE_DIR}/train" "${IMAGENET_BASE_DIR}/train" 120 80
  # convert_from_folders our_imagenet_split val "${IMAGENET_BASE_DIR}/train" "${IMAGENET_BASE_DIR}/train" 30 20
}

if [[ "${1:-}" == "--examples" ]]; then
  show_examples
  exit 0
fi

main
