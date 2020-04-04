#!/bin/bash

MY_PYTHON="python"
MNIST_ROTA="--n_layers 2 --n_hiddens 100 --data_path data/ --save_path results/ --batch_size 10 --log_every 100 --samples_per_task 1000 --data_file mnist_rotations.pt    --cuda no  --seed 0"


# build datasets
cd data/
cd raw/

$MY_PYTHON raw.py

cd ..

$MY_PYTHON mnist_rotations.py \
	--o mnist_rotations.pt\
	--seed 0 \
	--min_rot 0 \
	--max_rot 180 \
	--n_tasks 20

cd ..

# model "GEM"
$MY_PYTHON main.py $MNIST_ROTA --model gem --lr 0.1 --n_memories 256 --memory_strength 0.5


# plot results
cd results/
$MY_PYTHON plot_results.py
cd ..
