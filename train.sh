
#! /bin/bash
rm -rf data/
mkdir data/

python3 01-mmleakyrelu.py
python3 02-batch_matmul.py
python3 03-fused_feedforward.py
python3 04-fused-softmax.py
python3 05-rmsnorm.py
python3 06-fused-attention.py
