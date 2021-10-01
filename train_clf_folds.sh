#!/bin/sh

for i in 0 1 2 3 4 -1
do
    python src/train_2d_cnn_fold.py --fold=$i \
      --bs=32 \
      --epochs=30 \
      --lr=1e-4 \
      --arch="resnet18" \
      --ps=0.7 \
      --optim="ranger" \
      --im_sz=256
done

for i in 0 1 2 3 4 -1
do
    python src/train_2d_cnn_fold.py --fold=$i \
      --bs=32 \
      --epochs=30 \
      --lr=1e-4 \
      --arch="resnet34" \
      --ps=0.7 \
      --optim="ranger" \
      --im_sz=256
done
