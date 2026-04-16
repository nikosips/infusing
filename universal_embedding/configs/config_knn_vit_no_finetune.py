import ml_collections


def get_config():
  """Builds the default configuration for the ViT universal-embedding setup.

  These defaults can be overridden by command-line arguments. This config covers
  training, tuple mining, kNN evaluation, checkpointing, reranking, asymmetric
  retrieval, and diffusion options.
  """

  config = ml_collections.ConfigDict()

  # ---------------------------------------------------------------------------
  # Experiment / evaluation checkpoint selection
  # ---------------------------------------------------------------------------
  config.experiment_name = 'universal-embedding-vit'
  config.train_dir = ''

  config.test_pretrained_features = True
  config.preextracted = False

  config.text_datasets = ''
  config.do_text_eval = False
  config.do_image_eval = True

  # If True, evaluate only the best checkpoint for kNN.
  config.only_best_knn = False

  # If only_best_knn is True, the range below is ignored.
  config.knn_start_epoch = 1
  # Set lower than knn_start_epoch to effectively disable ranged kNN eval.
  config.knn_end_epoch = 0

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

  # ---------------------------------------------------------------------------
  # Sampling
  # ---------------------------------------------------------------------------
  # Supported values used in this project include:
  #   - "dataset_size"
  #   - "balanced"
  #   - "round_robin"
  config.sampling_strategy = 'round_robin'

  config.update_sampler = False
  config.update_sampler_mode = 'train_loss'
  config.update_sampler_every_steps = 1000
  config.update_sampler_logit_type = 'encoded'  # Alternative used: "decoded".

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

  config.model_dtype_str = 'float32'
  # Alternative used in some runs:
  # config.model_dtype_str = 'bfloat16'

  # MLP settings.
  config.model.encoder_mlp_dim = (64,)

  config.model.encoder_mlp_dim = (64,)
  config.model.stopgrad_backbone_out_to_encoded = False
  config.model.use_skip_on_mlp = False


  config.model.stopgrad_backbone_out_to_encoded = False

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

  config.rng_seed = 0

  # ---------------------------------------------------------------------------
  # Loss
  # ---------------------------------------------------------------------------
  config.loss = ml_collections.ConfigDict()
  config.loss.m = 0.0
  config.loss.scale = 16
  config.loss.transform_logits_type = 'normface'
  # Alternatives used in some runs:
  # config.loss.transform_logits_type = 'arcface'
  # config.loss.transform_logits_type = 'cosface'

  config.loss.classif_losses_on_string = "{'0':'encoded','1':'encoded,decoded'}"
  config.loss.classif_losses_weights_string = "{'0':'1.0','1':'1.0,1.0'}"
  config.loss.stopgrad_on_classifier_on_string = (
      "{'0':'encoded','1':'encoded,decoded'}"
  )

  config.loss.classif_losses_types_string = "{'0':'normface,normface'}"
  config.loss.classif_losses_margins_string = "{'0':'0.0,0.0'}"

  config.loss.aggregation_type = 'weighted_average'

  # ---------------------------------------------------------------------------
  # Checkpoint retention / logging frequencies
  # ---------------------------------------------------------------------------
  config.max_to_keep = 1000

  # These "frequency" values indicate how many times something happens per epoch
  # in the surrounding training / evaluation code.
  config.log_eval_steps_frequency = 1
  config.log_summary_steps_frequency = 10
  config.checkpoint_steps_frequency = 1

  # ---------------------------------------------------------------------------
  # Learning rate
  # ---------------------------------------------------------------------------
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant'
  config.lr_configs.base_learning_rate = 1e-3  # Classifier LR.

  config.lr_configs.backbone = ml_collections.ConfigDict()

  # For the first frozen_epochs, train only classifier / non-backbone params.
  config.frozen_epochs = 2
  config.backbone_learning_rate_multiplier = 1e-2

  config.params_early_train = [
      'output_projection',
      'encoder_projection',
  ]

  # ---------------------------------------------------------------------------
  # kNN / descriptor evaluation
  # ---------------------------------------------------------------------------
  config.do_knn = True
  config.no_finetune = True

  config.embedd_to_eval = 'backbone_out_embedd'
  config.universal_embedding_is = 'backbone_out_embedd'

  config.do_final_testing = True
  config.domain_agnostic_knn = True

  config.save_descriptors = False
  config.extract_only_descrs = True

  # ---------------------------------------------------------------------------
  # Logging / checkpointing / debug
  # ---------------------------------------------------------------------------
  config.write_summary = True
  config.checkpoint = True
  config.only_best_checkpoint = True

  config.debug_train = False
  config.debug_eval = False

  config.project_feats_knn = True
  config.save_neighbors = False
  config.top_k = 5

  config.log_domain_acc = True
  config.log_csv = False

  # ---------------------------------------------------------------------------
  # Dataset / metadata paths
  # ---------------------------------------------------------------------------
  config.eval_dataset_dir = ''
  config.train_dataset_dir = ''
  config.info_files_dir = ''

  config.descr_save_path = '.'

  return config
