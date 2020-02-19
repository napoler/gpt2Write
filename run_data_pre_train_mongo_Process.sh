#!/bin/bash
sum=0
echo "请输入您要计算的数字，按 Ctrl+D 组合键结束读取  循环次数"
while read n
do
    python bulid_data.py --do data_pre_train_mongo_Process
    # ((sum += n))
done
# echo "The sum is: $sum"