
#! /bin/bash

for i in {1..5}; do python3 01-mmleakyrelu.py --load auto --bench; done
for i in {1..5}; do python3 02-batch_matmul.py --load auto --bench; done
for i in {1..5}; do python3 03-fused_feedforward.py --load auto --bench; done
for i in {1..5}; do python3 04-fused-softmax.py --load auto --bench; done
for i in {1..5}; do python3 05-rmsnorm.py --load auto --bench; done
for i in {1..5}; do python3 06-fused-attention.py --load auto --bench; done

