import ml_collections


def get_config():
  """Builds the default configuration for the standalone kNN evaluation script.

  These values are used by the kNN script and may overwrite default values from
  the training configuration when both configs are combined.
  """

  config = ml_collections.ConfigDict()

  # ---------------------------------------------------------------------------
  # Base / inherited configuration placeholders
  # ---------------------------------------------------------------------------
  config.model_class = ''
  config.train_dir = ''

  config.pretrained_ckpt_dir = ''

  config.model = ml_collections.ConfigDict()
  config.loss = ml_collections.ConfigDict()

  # ---------------------------------------------------------------------------
  # Dataloader
  # ---------------------------------------------------------------------------
  config.use_grain_dataloader = False
  config.grain_worker_count = 8
  config.worker_buffer_size = 1  # Grain prefetch / buffering setting.

  # ---------------------------------------------------------------------------
  # kNN-specific overrides
  # ---------------------------------------------------------------------------
  config.do_knn = True
  config.no_finetune = False

  # Values below override the corresponding default values from the training
  # config when this config is used for standalone kNN evaluation.
  config.embedd_to_eval = 'backbone_out_embedd,'
  config.universal_embedding_is = 'backbone_out_embedd'

  config.knn_eval_names = 'food2k,cars,sop,inshop,inat'

  # Skip train / val kNN and evaluate test splits only by default.
  config.disabled_separate_knns = 'train_knn,val_knn'
  config.disabled_merged_knns = 'train_knn,val_knn'

  config.eval_batch_size = 1024
  config.knn_eval_batch_size = 2048

  # ---------------------------------------------------------------------------
  # Evaluation mode
  # ---------------------------------------------------------------------------
  config.preextracted = False
  config.test_pretrained_features = False
  config.extract_only_descrs = False

  config.text_datasets = ''
  config.do_text_eval = False
  config.do_image_eval = True

  # ---------------------------------------------------------------------------
  # Output / logging
  # ---------------------------------------------------------------------------
  config.write_summary = True
  config.save_descriptors = True
  config.save_neighbors = False
  config.log_csv = False
  config.debug_eval = False

  # ---------------------------------------------------------------------------
  # Dataset / metadata paths
  # ---------------------------------------------------------------------------
  config.eval_dataset_dir = ''
  config.train_dataset_dir = ''
  config.info_files_dir = ''
  config.descr_save_path = None  # Save in train_dir.

  # ---------------------------------------------------------------------------
  # kNN behavior
  # ---------------------------------------------------------------------------
  config.project_feats_knn = True

  # Number of nearest neighbors to store / inspect.
  config.top_k = 5

  config.calc_descriptor_information = False

  # If True, evaluate only the best checkpoint; epoch range below is ignored.
  config.only_best_knn = True

  # If only_best_knn is False, evaluate checkpoints in this epoch range.
  config.knn_start_epoch = 3
  # Set lower than knn_start_epoch to effectively disable ranged kNN eval.
  config.knn_end_epoch = 7

  return config