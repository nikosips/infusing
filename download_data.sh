#!/bin/bash

#download models
mkdir -p data/models/siglip
mkdir -p data/models/tips

wget https://cmp.felk.cvut.cz/univ_emb/checkpoints/siglip/vit_b16.npy -P data/models/siglip/
wget https://cmp.felk.cvut.cz/univ_emb/checkpoints/tips/vit_b16.npy -P data/models/tips/

#download extracted features of pretrained models on various datasets
mkdir -p data/features/siglip
mkdir -p data/features/tips

#download text encoder features of siglip and tips for mscoco and flickr30k