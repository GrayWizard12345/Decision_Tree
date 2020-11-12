#!/bin/bash

train='data/data/dt_train.txt'
test='data/data/dt_test.txt'
res='res.txt'
python3.8 dt.py $train $test $res


train='data/data/dt_train1.txt'
test='data/data/dt_test1.txt'
res='res1.txt'
python3.8 dt.py $train $test $res