# Infusing Fine-Grained Visual Knowledge to Vision-Language Models

This repository contains the training and evaluation code for the paper _Infusing fine-grained visual knowledge to Vision-Language Models_.

The codebase builds on Universal Image Embeddings / UnED and uses [Scenic](https://github.com/google-research/scenic), JAX/Flax, Grain, and supporting components from Big Vision and TIPS.

The main workflow in this repo is:

1. Prepare image datasets and metadata.
2. Convert the datasets into `array_record` files for Grain-based loading.
3. Download pretrained backbone checkpoints and optional pretrained descriptors.
4. Optionally extract pretrained train descriptors for the target-dataset plus generic-dataset pair.
5. Train a universal embedding model with a SigLIP- or TIPS-based ViT backbone.
6. Run test-set kNN and optional text evaluation from the saved best checkpoint.

## Repository layout

```text
.
├── universal_embedding/
│   ├── main.py                        # Training entrypoint
│   ├── knn_main.py                    # Standalone kNN evaluation entrypoint
│   ├── app.py                         # Shared CLI / runtime wrapper
│   ├── grain_datasets.py              # Grain dataset builders
│   ├── classification_with_knn_eval_trainer.py
│   ├── model_init.py
│   ├── train_eval_steps.py
│   ├── knn_utils.py
│   ├── text_eval_utils.py
│   └── configs/
│       ├── config_train_vit.py
│       ├── config_knn_vit.py
│       └── config_knn_vit_no_finetune.py
├── prepare_data.sh                    # Example conversion commands for dataset records
├── download_data.sh                   # Downloads pretrained checkpoints / features
├── setup.sh                           # Environment bootstrap script
├── scripts/                           # Main experiment launchers
└── convert_to_array_record.py
```

## What the code supports

- Fine-tuning on a target fine-grained dataset while optionally infusing generic visual data such as `our_imagenet_split`.
- ViT backbones initialized from SigLIP or TIPS checkpoints.
- Grain dataloading from `array_record` shards.
- Periodic and final kNN evaluation during training.
- Standalone checkpoint evaluation with `universal_embedding/knn_main.py`.
- Optional text evaluation hooks for paired image-text benchmarks.
- Multi-device JAX execution through the standard Scenic/JAX runtime.

## Requirements

The repository assumes a Linux environment with Python 3.10 and a working JAX setup.

At a minimum you will need:

- Python `3.10`
- `venv`
- JAX / Flax
- Scenic
- Grain
- TensorFlow and `tensorflow_text`
- `wandb` if you want experiment tracking

The included [setup.sh](setup.sh) script bootstraps a local environment and installs the main dependencies. It also clones Scenic, Big Vision, and TIPS during setup.

Important notes about `setup.sh`:

- It uses `sudo apt` commands.
- It creates a virtual environment named `scenic_venv`.
- It installs `jax[tpu]` by default, so you will likely want to adjust it if you are running on GPU or CPU only.
- It clones `big_vision` and `tips` next to the repository, as siblings of `infusing`, because the launch scripts add the parent directory to `PYTHONPATH`.

If you prefer a manual setup, use `setup.sh` as a starting point rather than assuming it is portable as-is.

## Data preparation

The training configs in this repository default to the following fine-grained datasets:

- `food2k`
- `cars`
- `sop`
- `inshop`
- `inat`

### Expected inputs

The code expects:

- raw images on disk
- JSON metadata files per dataset split under an `info_files` directory
- `array_record` shards per split for training / evaluation

For evaluation and some initialization paths, you will also need pretrained checkpoints and optionally pre-extracted descriptors.

If you enable image-text evaluation, you will also need the text-image evaluation assets under `data/text_image/`.

### Converting datasets to ArrayRecord

This project uses Grain with `array_record` files rather than reading raw image folders directly during training.

The helper scripts are:

- [convert_to_array_record.py](convert_to_array_record.py)
- [prepare_data.sh](prepare_data.sh)

`convert_to_array_record.py` supports both:

- `--input_format=json` for datasets described by split JSON files
- `--input_format=folder` for ImageNet-style class-per-folder directories

`prepare_data.sh` is now a thin wrapper around that unified converter. It defines reusable shell functions for JSON-based datasets and ImageNet-style folder scans, plus a few example conversions you can uncomment.

In practice, you should:

1. Prepare the dataset JSON split files under your metadata directory.
2. Update the paths in `prepare_data.sh` or call `convert_to_array_record.py` directly.
3. Generate `array_record` shards for each split you plan to train or evaluate on.

### Expected dataset directory structure

The dataset loader builds record paths from [universal_embedding/dataset_infos.py](universal_embedding/dataset_infos.py). A typical local layout looks like:

```text
data/
├── array_records/
│   ├── cars/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── food2k/
│   ├── inat/
│   ├── inshop/
│   └── sop/
├── info_files/
│   ├── cars/
│   │   ├── train.json
│   │   ├── val.json
│   │   └── test.json
│   └── ...
└── models/
    ├── siglip/
    └── tips/
```

You will need to point the config fields below at your local paths:

- `train_dataset_dir`
- `eval_dataset_dir`
- `info_files_dir`
- `pretrained_ckpt_dir`
- `pretrained_train_descriptors_dir` when applicable

## Pretrained checkpoints and features

The helper script [download_data.sh](download_data.sh) downloads the public assets from:

```text
https://login.rci.cvut.cz/~ypsilnik/infusing_data
```

and places them into the repository `data/` directory.

It provides:

- model checkpoints under `data/models/`
- dataset metadata JSON files under `data/info_files/`
- public ArrayRecord files under `data/array_records/`
- text-image evaluation assets under `data/text_image/`

Run:

```bash
bash download_data.sh
```

The script still does not download:

- Precomputed descriptor features

If you use pretrained descriptor distillation or text-evaluation features, make sure those assets exist and update the config paths accordingly.

## Text-image evaluation data

The training and evaluation scripts enable image-text retrieval evaluation through:

```text
data/text_image/flickr30k
data/text_image/mscoco
```

These paths are passed through `--config.text_datasets` in the shell scripts.

For each text-image dataset, the code expects:

- `queries.tfrecord`
- `<dataset_name>_text_embeddings_siglip.npy` or `<dataset_name>_text_embeddings_tips.npy`
- `<dataset_name>_gt.npy`

For example:

```text
data/text_image/flickr30k/
├── queries.tfrecord
├── flickr30k_text_embeddings_siglip.npy
├── flickr30k_text_embeddings_tips.npy
└── flickr30k_gt.npy

data/text_image/mscoco/
├── queries.tfrecord
├── mscoco_text_embeddings_siglip.npy
├── mscoco_text_embeddings_tips.npy
└── mscoco_gt.npy
```

If these assets are not available, disable text evaluation in your run configuration or script invocation.

## Scripts

The main user-facing entrypoints in this repository are the shell scripts in [scripts](scripts).

Run them from the repository root:

```bash
bash scripts/<script_name>.sh ...
```

Available scripts:

- [scripts/one_domain_pipeline.sh](scripts/one_domain_pipeline.sh): full 3-step pipeline for a target-domain run: descriptor extraction, training, then final test evaluation.
- [scripts/baseline.sh](scripts/baseline.sh): baseline training run on `finetuning_dataset,generic_dataset`, without the proposed regularization terms used in the full method.
- [scripts/full_method.sh](scripts/full_method.sh): training run with pretrained embedding and weight regularization enabled.
- [scripts/evaluation.sh](scripts/evaluation.sh): standalone kNN and text evaluation for a saved training run.
- [scripts/extract_pretrained_feats.sh](scripts/extract_pretrained_feats.sh): extract descriptors from an off-the-shelf pretrained model.

The scripts accept an optional first argument for `wandb_entity`. Pass an empty string if you do not want to log to Weights & Biases.

## Recommended workflow

The main public entrypoint for a complete experiment is [scripts/one_domain_pipeline.sh](scripts/one_domain_pipeline.sh), which performs:

1. descriptor extraction for `finetuning_dataset,generic_dataset`
2. full-method training
3. final test-set kNN evaluation on the saved best checkpoint

Example public run:

```bash
bash scripts/one_domain_pipeline.sh "" sop our_imagenet_split siglip 3 cars,sop,inshop,inat,imagenet,food2k round_robin 764 1000 10000
```

This example fine-tunes on `sop`, uses `our_imagenet_split` as the generic source dataset, initializes from a SigLIP backbone, extracts pretrained train descriptors, trains the full method, and then evaluates on `cars,sop,inshop,inat,imagenet,food2k`.

## When to use which script

Use [scripts/one_domain_pipeline.sh](scripts/one_domain_pipeline.sh) when you want the full intended workflow but with your own datasets, schedule, and weights.

Use [scripts/baseline.sh](scripts/baseline.sh) when you want to train the comparison baseline without the proposed regularization.

Use [scripts/full_method.sh](scripts/full_method.sh) directly when you already have pretrained train descriptors extracted and only want to train while monitoring validation kNN during training.

Use [scripts/evaluation.sh](scripts/evaluation.sh) directly when you already have a trained run and only want final test evaluation.

Use [scripts/extract_pretrained_feats.sh](scripts/extract_pretrained_feats.sh) directly when you only need off-the-shelf descriptors.

## Configuration

The primary configuration files are:

- [universal_embedding/configs/config_train_vit.py](universal_embedding/configs/config_train_vit.py)
- [universal_embedding/configs/config_knn_vit.py](universal_embedding/configs/config_knn_vit.py)
- [universal_embedding/configs/config_knn_vit_no_finetune.py](universal_embedding/configs/config_knn_vit_no_finetune.py)

The default training config is set up for:

- model class: `siglip_vit_with_embedding`
- model type: `B/16`
- datasets: `food2k,cars,sop,inshop,inat`
- training epochs: `10`
- batch size: `128`
- kNN evaluation enabled during training

Before launching a run, review at least these fields:

- `model_class`
- `model_type`
- `dataset_name`
- `knn_eval_names`
- `train_dataset_dir`
- `eval_dataset_dir`
- `info_files_dir`
- `pretrained_ckpt_dir`
- `train_dir`
- `batch_size`
- `num_training_epochs`
- `use_grain_dataloader`

## Descriptor flow

The full method depends on pretrained train descriptors produced from an off-the-shelf model.

The expected flow is:

1. [scripts/extract_pretrained_feats.sh](scripts/extract_pretrained_feats.sh) writes descriptors under:

```text
data/exps/experiments/off-the-shelf/features/<pretraining>_vitB_pretrained_embeddings/descriptors/0/backbone_out_embedd/
```

2. Inside that directory, descriptors are stored per dataset and split, for example:

```text
data/exps/experiments/off-the-shelf/features/siglip_vitB_pretrained_embeddings/descriptors/0/backbone_out_embedd/
├── sop/train.npy
└── our_imagenet_split/train.npy
```

3. [scripts/full_method.sh](scripts/full_method.sh) loads those `train.npy` files through `--config.pretrained_train_descriptors_dir`.

If those descriptor files already exist for your `finetuning_dataset,generic_dataset` pair, you can skip extraction and launch training directly.

## Training

Training runs through [universal_embedding/main.py](universal_embedding/main.py), but in normal use you should launch it through the scripts in [scripts](scripts), especially:

- [scripts/one_domain_pipeline.sh](scripts/one_domain_pipeline.sh)
- [scripts/full_method.sh](scripts/full_method.sh)
- [scripts/baseline.sh](scripts/baseline.sh)

During training, the repository runs kNN evaluation on the validation split according to the configured cadence, so you can monitor validation behavior without running the final evaluation script separately.

During startup, the shared wrapper in [universal_embedding/app.py](universal_embedding/app.py) saves the resolved configuration to:

```text
<workdir>/config.json
```

That saved config is later reused by standalone kNN evaluation.

## Standalone kNN evaluation

Standalone evaluation runs through [universal_embedding/knn_main.py](universal_embedding/knn_main.py), and the normal entrypoint is [scripts/evaluation.sh](scripts/evaluation.sh).

In the intended one-domain workflow, this script is the final pipeline step and evaluates the test split using the best checkpoint saved by training.

For evaluation, make sure the config points to the training directory:

- `train_dir`: directory containing the checkpoints and saved `config.json`

Behavior to know:

- If `only_best_knn=True`, the script evaluates only the best checkpoint.
- If `only_best_knn=False`, it evaluates checkpoints over the configured epoch range.
- If `test_pretrained_features=True`, it evaluates the initialized model before checkpoint restoration.
- If `no_finetune=True`, the script can evaluate a pretrained backbone without loading finetuned checkpoints.

To extract pretrained descriptors without a finetuned run, use [scripts/extract_pretrained_feats.sh](scripts/extract_pretrained_feats.sh).

## Logging and outputs

The runtime uses CLU metric writers and can optionally initialize Weights & Biases.

Outputs are written under the specified `workdir`, including:

- saved `config.json`
- TensorBoard-compatible event files and summaries
- checkpoints
- optional saved descriptors
- optional nearest-neighbor outputs

The event files written to `workdir` can be viewed directly with TensorBoard. If Weights & Biases logging is enabled, the same run is also synced there for remote logging and visualization.

## Notes

- `setup.sh` is not fully self-contained. Review and adapt it before running.
- `download_data.sh` is the actual script for model downloads; the previous README referred to `download_models.sh`, which does not exist in this repository.
- The training and evaluation configs leave several path fields empty by default. They must be filled in for your environment.
- `prepare_data.sh` is a helper wrapper, not a full data-ingestion pipeline.
- Some features referenced by the configs, such as pretrained descriptors and text-evaluation assets, are expected to exist externally.

## Citation

If you use this repository in academic work, cite:

```bibtex
@inproceedings{ypsilantis2025infusing,
  title={Infusing fine-grained visual knowledge to Vision-Language Models},
  author={Ypsilantis, Nikolaos-Antonios and Chen, Kaifeng and Araujo, Andr{\'e} and Chum, Ondrej},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={4226--4235},
  year={2025}
}
```

## Acknowledgements

This codebase builds on and/or depends on:

- Scenic
- Universal Image Embeddings / UnED
- Big Vision
- TIPS
- JAX / Flax
- Grain
