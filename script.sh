#!/bin/bash

devices=(0 2 3 6 7)

for i in {0..4}
do
    export CUDA_VISIBLE_DEVICES=${devices[$i]}
    nohup python synthetic.py --game $i > log$i &
done