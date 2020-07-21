#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:src/
cd ../
nohup python3.6 src/train.py \
  --do_train=True \
  --do_predict=True \
  --model_dir=./out/ubuntu \
  --num_train_samples=890000 \
  --num_epoches=10 \
  --batch_size=64 \
  --data_dir=dat/ubuntu \
  --vocab_file=ckpt/albert/vocab.txt \
  --model_config=./cfg/poly_encoder.json > log.txt 2>&1 &

tail -f log.txt