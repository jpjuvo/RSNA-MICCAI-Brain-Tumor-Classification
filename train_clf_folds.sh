#!/bin/sh

# Efficientnetv2-l
for i in 0 1 2 3 4 -1
do
    python src/train_2d_cnn_fold.py --fold=$i \
      --bs=32 \
      --epochs=10 \
      --lr=1e-4 \
      --arch="tf_efficientnetv2_l" \
      --npy_dir="./input/aligned_and_cropped_t2w/" \
      --ps=0.8 \
      --optim="ranger" \
      --im_sz=256 \
      --loss_name="libauc"
done

# densenet169
#for i in 0 1 2 3 4 -1
#do
#    python src/train_2d_cnn_fold.py --fold=$i \
#      --bs=32 \
#      --epochs=10 \
#      --lr=1e-4 \
#      --arch="densenet169" \
#      --npy_dir="./input/aligned_and_cropped_t2w/" \
#      --ps=0.8 \
#      --optim="ranger" \
#      --im_sz=256 \
#      --loss_name="libauc"
#done

# resnet101
#for i in 0 1 2 3 4 -1
#do
#    python src/train_2d_cnn_fold.py --fold=$i \
#      --bs=32 \
#      --epochs=20 \
#      --lr=1e-4 \
#      --arch="resnet101" \
#      --npy_dir="./input/aligned_and_cropped_t2w/" \
#      --ps=0.8 \
#      --optim="ranger" \
#      --im_sz=256 \
#      --loss_name="libauc"
#done