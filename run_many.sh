#!/bin/bash
FILES=data/s/*
for f in $FILES
do
  filename=$(basename "$f")
  echo "==========================="
  echo $filename

  CUDA_VISIBLE_DEVICES=$gpu th train.lua -style_image $filename -style_size 600 -image_size 512 -model $model -batch_size 4 -learning_rate $lr -style_weight 10 -style_layers relu1_2,relu2_2,relu3_2,relu4_2 -content_layers relu4_2 -nThreads 8 -pairwise_loss $pairwise -display_port 8003 -add_noise

done