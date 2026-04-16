import numpy as np

DATASET_INFO = {
    'cars': {
        'domain': 0,
        'train_files': 'cars/train/cars.train.tfrecord',
        'test_files': 'cars/test/cars.test.tfrecord',
        'val_files': 'cars/val/cars.val.tfrecord',

        'num_train_classes': 78,
        'num_train_examples': 6346,
        'num_test_examples': 8131,
        'num_val_examples': 1708,
        'knn': {
            'train_knn': {
                'query': 'train',
                'index': 'train',
            },
            'val_knn': {
                'query': 'val',
                'index': 'val',
            },
            'test_knn': {
                'query': 'test',
                'index': 'test',
            },
        },
    },
    'sop': {
        'domain': 1,
        'train_files': 'sop/train/sop.train.tfrecord',
        'test_files': 'sop/test/sop.test.tfrecord',
        'val_files': 'sop/val/sop.val.tfrecord',

        'num_train_classes': 9054,
        'num_train_examples': 48942,
        'num_test_examples': 60502,
        'num_val_examples': 10609,
        'knn': {
            'train_knn': {
                'query': 'train',
                'index': 'train',
            },
            'val_knn': {
                'query': 'val',
                'index': 'val',
            },
            'test_knn': {
                'query': 'test',
                'index': 'test',
            },
        },
    },

    'inshop': {
        'domain': 2,
        'train_files': 'inshop/train/inshop.train.tfrecord',
        'test_files': 'inshop/test/inshop.test.tfrecord',
        'val_files': 'inshop/val/inshop.val.tfrecord',
        'index_files': 'inshop/index/inshop.index.tfrecord',  # size 12612

        'num_train_classes': 3198,
        'num_train_examples': 20897,
        'num_test_examples': 14218,
        'num_val_examples': 4982,
        'knn': {
            'train_knn': {
                'query': 'train',
                'index': 'train',
            },
            'val_knn': {
                'query': 'val',
                'index': 'val',
            },
            'test_knn': {
                'query': 'test',
                'index': 'index',
            },
        },
    },

    'inat': {
        'domain': 3,
        'train_files': 'inat/train/inat.train.tfrecord',
        'test_files': 'inat/test/inat.test.tfrecord',
        'val_files': 'inat/val/inat.val.tfrecord',

        'num_train_classes': 4552,
        'num_train_examples': 273929,
        'num_test_examples': 136093,
        'num_val_examples': 51917,
        'knn': {
            'train_knn': {
                'query': 'train',
                'index': 'train',
            },
            'val_knn': {
                'query': 'val',
                'index': 'val',
            },
            'test_knn': {
                'query': 'test',
                'index': 'test',
            },
        },
    },

    'food2k': {
        'domain': 6,
        'train_files': 'food2k/train/food2k.train.tfrecord',
        'test_files': 'food2k/test/food2k.test.tfrecord',
        'val_files': 'food2k/val/food2k.val.tfrecord',


        'num_train_classes': 900,
        'num_train_examples': 472349,
        'num_test_examples': 9979,
        'num_val_examples': 49323,
        'knn': {
            'train_knn': {
                'query': 'train',
                'index': 'train',
            },
            'val_knn': {
                'query': 'val',
                'index': 'val',
            },
            'test_knn': {
                'query': 'test',
                'index': 'test',
            },
        },
    },

    'imagenet': {
        'domain': 8,
        'train_files': 'imagenet/train/imagenet.train.tfrecord',
        'test_files': 'imagenet/val/imagenet.val.tfrecord',
        'val_files': 'imagenet/val/imagenet.val.tfrecord',


        'num_train_classes': 1000,
        #'num_train_examples': 1281167,
        'num_train_examples': int(np.ceil(1281167 * 0.1)),
        #'num_train_examples': int(np.ceil(1281167 * 0.01)),
        'num_test_examples': 50000,
        'num_val_examples': 50000,
        'knn': {
            'train_knn': {
                'query': 'train',
                'index': 'train',
            },
            'val_knn': {
                'query': 'val',
                'index': 'val',
            },
            'test_knn': {
                'query': 'test',
                'index': 'test',
            },
        },
    },

    'our_imagenet_split': {
        'domain': 9,
        'train_files': 'our_imagenet_split/train/imagenet.train.tfrecord',
        
        'val_files': 'our_imagenet_split/val/imagenet.val.tfrecord',


        'num_train_classes': 1000,
        'num_train_examples': 1024933,
        #'num_train_examples': int(np.ceil(1024933 * 0.1)),
        #'num_train_examples': int(np.ceil(1024933 * 0.13)),
        'num_val_examples': 256233,
        'knn': {
            'train_knn': {
                'query': 'train',
                'index': 'train',
            },
            'val_knn': {
                'query': 'val',
                'index': 'val',
            },
        },
    },

    'laion': {
        'domain': 10,
        'train_files': 'laion/train/laion.train.tfrecord',
        'val_files': 'laion/val/laion.val.tfrecord',

        'num_train_classes': 1172303,
        'num_train_examples': 1172303, #30% of our laion subset is equivalent to 10% of imagenet
        'num_val_examples': 130256,
        'knn': {
            'train_knn': {
                'query': 'train',
                'index': 'train',
            },
            'val_knn': {
                'query': 'val',
                'index': 'val',
            },
            # 'test_knn': {
            #     'query': 'test',
            #     'index': 'test',
            # },
        },

    },


}
