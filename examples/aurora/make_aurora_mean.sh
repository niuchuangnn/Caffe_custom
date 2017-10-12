#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

EXAMPLE=/home/amax/NiuChuang/KLSA-auroral-images/Data
DATA=/home/amax/NiuChuang/KLSA-auroral-images/Data
TOOLS=build/tools

$TOOLS/compute_image_mean $EXAMPLE/aurora_train_lmdb \
  $DATA/aurora_mean.binaryproto

echo "Done."
