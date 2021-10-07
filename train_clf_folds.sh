#!/bin/sh

for i in 0 1 2 3 4 -1
do
    python src/train_2d_cnn_fold.py --fold=$i \
      --bs=24 \
      --epochs=10 \
      --lr=1e-4 \
      --arch="tf_efficientnet_b7_ns" \
      --ps=0.8 \
      --optim="ranger" \
      --im_sz=256 \
      --loss_name="rocstar"
done

for i in 0 1 2 3 4 -1
do
    python src/train_2d_cnn_fold.py --fold=$i \
      --bs=24 \
      --epochs=10 \
      --lr=1e-4 \
      --arch="resnest269e" \
      --ps=0.8 \
      --optim="ranger" \
      --im_sz=256 \
      --loss_name="rocstar"
done
