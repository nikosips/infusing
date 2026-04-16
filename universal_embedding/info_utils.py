from universal_embedding import dataset_infos


DATASET_TRAIN_SIZE = { #number of train images per dataset

  'cars': dataset_infos.DATASET_INFO['cars']['num_train_examples'],
  'sop': dataset_infos.DATASET_INFO['sop']['num_train_examples'],
  'inshop': dataset_infos.DATASET_INFO['inshop']['num_train_examples'],
  'inat': dataset_infos.DATASET_INFO['inat']['num_train_examples'],
  'food2k': dataset_infos.DATASET_INFO['food2k']['num_train_examples'],
  'imagenet': dataset_infos.DATASET_INFO['imagenet']['num_train_examples'],
  'our_imagenet_split': dataset_infos.DATASET_INFO['our_imagenet_split']['num_train_examples'],
  'laion': dataset_infos.DATASET_INFO['laion']['num_train_examples'],
  
}

#note: for models without the default image size of 224x224, the image size is mentioned in the config dictionary and subsequently used in the data loader

SigLIP_ViT_configs = {
  'B/16': {
    "hidden_size" : 768,
    "patches_size" : [16, 16],
    "num_heads" : 12,
    "mlp_dim" : 3072,
    "num_layers" : 12,
    "checkpoint" : 'siglip/webli_en_b16_224_63724782.npz:img',
    "image_size": 224,
    "image_resize": 256,
    "normalization_statistics": {
      "MEAN_RGB": [0.5,0.5,0.5],
      "STDDEV_RGB": [0.5,0.5,0.5],
    },
  },

  'L/16': {
    "hidden_size" : 1024,
    "patches_size" : [16, 16],
    "num_heads" : 16,
    "mlp_dim" : 4096,
    "num_layers" : 24,
    #"checkpoint" : 'siglip/webli_en_l16_384_63634585.npz:img',
    "checkpoint" : 'gs://big_vision/siglip/webli_en_l16_384_63634585.npz:img',
    "image_size": 384,
    "image_resize": 432,
    "normalization_statistics": {
      "MEAN_RGB": [0.5,0.5,0.5],
      "STDDEV_RGB": [0.5,0.5,0.5],
    },
  },
}

TIPS_ViT_configs = {
  'B/16': {
    "hidden_size" : 768,
    "patches_size" : [16, 16],
    "num_heads" : 12,
    "mlp_dim" : 3072,
    "num_layers" : 12,
    "checkpoint" : 'tips/tips_oss_b14_highres_distilled_vision.npz',
    "image_size": 224,
    "image_resize": 256,
    "normalization_statistics": {
      "MEAN_RGB": [0.,0.,0.],
      "STDDEV_RGB": [1.,1.,1.],
    },
  },
}



def get_aggregated_size(datasets):

  size = 0

  for dataset in datasets.split(','):
    size += DATASET_TRAIN_SIZE[dataset]

  return size
