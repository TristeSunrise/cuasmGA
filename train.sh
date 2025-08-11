
#! /bin/bash
rm -rf data/
mkdir data/

python3 benchmarks/01-mmleakyrelu.py
python3 benchmarks/02-batch_matmul.py
python3 benchmarks/03-fused_feedforward.py
python3 benchmarks/04-fused-softmax.py
python3 benchmarks/05-rmsnorm.py
python3 benchmarks/06-fused-attention.py
