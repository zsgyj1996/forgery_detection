#!/bin/bash

# 读取文件中的每一行
while IFS= read -r line; do
    # 将文件复制到目标文件夹
    cp "$line" /data/shuozhang/merged/dataset/Tp/
done < "/data/shuozhang/merged/tp_list.txt"
