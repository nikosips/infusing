import ml_collections


def get_config():
  """Builds the default experiment configuration for universal embedding ViT.

  These defaults are intended to be overridden by command-line arguments when
  needed. The configuration covers dataset loading, model setup, optimization,
  evaluation, logging, and checkpointing.
  """

  config = ml_collections.ConfigDict()

  # ---------------------------------------------------------------------------
  # Experiment
  # ---------------------------------------------------------------------------
  config.experiment_name = 'universal-embedding-vit'

  # ---------------------------------------------------------------------------
  # Dataset
  # ---------------------------------------------------------------------------
  config.use_grain_dataloader = False
  config.grain_worker_count = 8
  config.worker_buffer_size = 1  # Grain prefetch / buffering setting.
  config.dataset_type = 'one_sample'

  config.dataset_name = 'food2k,cars,sop,inshop,inat'
  config.knn_eval_names = 'food2k,cars,sop,inshop,inat'
  config.data_dtype_str = 'float32'

  config.dataset_configs = ml_collections.ConfigDict()
  config.pretrained_train_descriptors_dir = ''

  # ---------------------------------------------------------------------------
  # Sampling
  # ---------------------------------------------------------------------------
  # Possible values used in this project include:
  #   - "dataset_size"
  #   - "balanced"
  #   - "round_robin"
  config.sampling_strategy = 'round_robin'

  config.update_sampler = False
  config.update_sampler_mode = 'train_loss'  # Or "train_loss+val_loss".
  config.update_sampler_every_steps = 1000
  config.update_sampler_logit_type = 'encoded'  # Or "decoded".
  config.update_sampler_after_epochs = 0
  # Ensures validation results exist if using "train_loss+val_loss".

  config.do_ema_on_sampler = False
  config.do_ema_on_sampler_decay = 0.3

  # ---------------------------------------------------------------------------
  # General experiment options
  # ---------------------------------------------------------------------------
  config.classifier = 'separate'
  config.count_flops = False

  # ---------------------------------------------------------------------------
  # Model
  # ---------------------------------------------------------------------------
  config.model_class = 'siglip_vit_with_embedding'
  config.model_type = 'B/16'  # Example alternative used in the project: "S/16".

  config.model = ml_collections.ConfigDict()
  config.model.representation_size = None
  config.model.output_dim = 64
  config.model.classifier = 'token'
  config.model.attention_dropout_rate = 0.0
  config.model.dropout_rate = 0.0
  config.model.positional_embedding = 'learned_1d'  # Alternative: "none".

  # MLP-related model options.
  config.model.encoder_mlp_dim = (64,)
  config.model.stopgrad_backbone_out_to_encoded = False
  config.model.use_skip_on_mlp = False

  config.model_dtype_str = 'float32'
  # Alternative used in some runs:
  # config.model_dtype_str = 'bfloat16'

  # ---------------------------------------------------------------------------
  # Checkpoint initialization / restoration
  # ---------------------------------------------------------------------------
  config.pretrained_ckpt_dir = ''
  config.start_from_scratch = False

  config.init_ckpt = ''
  config.restore_only_backbone = True
  config.keys_to_restore = [
      'Transformer',
      'cls',
      'embedding',
      'encoder_projection_domain_0',
  ]

  # ---------------------------------------------------------------------------
  # Training
  # ---------------------------------------------------------------------------
  # Scenic uses AdamW behavior when weight decay is set.
  config.optimizer = 'adam'
  config.optimizer_configs = ml_collections.ConfigDict()
  config.optimizer_configs.beta1 = 0.9
  config.optimizer_configs.beta2 = 0.999
  config.optimizer_configs.weight_decay = 1e-6

  config.explicit_weight_decay = None
  config.l2_decay_factor = None
  config.label_smoothing = None

  config.num_training_epochs = 10
  config.batch_size = 128
  config.eval_batch_size = 1024
  config.knn_eval_batch_size = 2048

  # During training, skip kNN on these splits.
  config.disabled_separate_knns = 'train_knn,test_knn'
  config.disabled_merged_knns = 'train_knn,test_knn'

  # During final evaluation, evaluate only on test splits.
  config.disabled_separate_final_eval_knns = 'train_knn,val_knn'
  config.disabled_merged_final_eval_knns = 'train_knn,val_knn'
  config.knn_eval_names_final = ''

  config.rng_seed = 0

  # ---------------------------------------------------------------------------
  # Loss
  # ---------------------------------------------------------------------------
  config.loss = ml_collections.ConfigDict()
  config.loss.m = 0.0
  config.loss.scale = 16
  config.loss.trainable_scale = False
  config.loss.transform_logits_type = 'normface'

  config.loss.classif_losses_on_string = "{'0':'encoded','1':'encoded,decoded'}"
  config.loss.classif_losses_weights_string = "{'0':'1.0','1':'1.0,1.0'}"
  config.loss.stopgrad_on_classifier_on_string = (
      "{'0':'encoded','1':'encoded,decoded'}"
  )

  config.loss.classif_losses_types_string = "{'0':'normface,normface'}"
  config.loss.classif_losses_margins_string = "{'0':'0.0,0.0'}"

  # Comma-separated domain indices.
  config.loss.pretrained_embedding_distill_loss_on = ''
  config.loss.pretrained_embedding_distill_loss_weight = 1.0

  config.loss.aggregation_type = 'weighted_average'

  config.loss.pretrained_weights_loss = False
  config.loss.pretrained_weights_loss_weight = 1.0

  # ---------------------------------------------------------------------------
  # Checkpoint retention / logging frequencies
  # ---------------------------------------------------------------------------
  config.max_to_keep = 1000

  # These "frequency" values indicate how many times something happens per epoch
  # in the surrounding training code.
  config.log_eval_steps_frequency = 1       # kNN eval logging.
  config.log_eval_steps = -1                # kNN eval logging control.
  config.log_summary_steps_frequency = 10   # Training metric logging.
  config.checkpoint_steps_frequency = 1     # Checkpointing frequency.
  config.mlp_weights_steps_frequency = 2    # Encoded vs decoded loss weights.

  # ---------------------------------------------------------------------------
  # Learning rate
  # ---------------------------------------------------------------------------
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant'
  # If cosine decay is used, it currently applies only to randomly initialized
  # params, not those using classifier_lr_fn in the current project code.
  # Example:
  # config.lr_configs.factors = 'constant * cosine_decay'
  config.lr_configs.base_learning_rate = 1e-3

  config.lr_configs.backbone = ml_collections.ConfigDict()

  # For the first frozen_epochs, train only the classifier / non-backbone params.
  config.frozen_epochs = 2
  config.backbone_learning_rate_multiplier = 1e-2

  # Only used if 'prompt' is part of config.model_class.
  config.params_early_train = [
      'output_projection',
      'encoder_projection',
      'logit_scale',
  ]

  # ---------------------------------------------------------------------------
  # kNN evaluation
  # ---------------------------------------------------------------------------
  config.do_knn = True
  config.do_knn_at_start = False
  config.domain_agnostic_knn = True

  config.embedd_to_eval = 'backbone_out_embedd'
  config.universal_embedding_is = 'backbone_out_embedd'

  config.do_final_testing = True
  config.save_descriptors = False
  config.extract_only_descrs = False
  config.project_feats_knn = True
  config.save_neighbors = False
  config.top_k = 5

  # ---------------------------------------------------------------------------
  # Logging / checkpointing / debug
  # ---------------------------------------------------------------------------
  config.write_summary = True
  config.checkpoint = True
  config.only_best_checkpoint = True

  config.debug_train = False
  config.debug_eval = False

  config.log_domain_acc = True
  config.log_csv = False

  # ---------------------------------------------------------------------------
  # Dataset / metadata paths
  # ---------------------------------------------------------------------------
  config.eval_dataset_dir = ''
  config.train_dataset_dir = ''
  config.info_files_dir = ''

  config.descr_save_path = None

  # ---------------------------------------------------------------------------
  # Text evaluation
  # ---------------------------------------------------------------------------
  config.text_datasets = ''
  config.do_text_eval_during_validation = False
  config.do_text_eval = False

  # Can be "in-domain", "out-of-domain", "all", or "".
  config.best_val_knn_on = 'in-domain'

  return config
