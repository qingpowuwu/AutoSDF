#!/bin/bash

# 目标路径
target_path="/data/3dPrinter/0_Dataset_Ori/4_DISN_Datasets/1_Downloaded/SDF_v1"

# 链接路径
link_path="/data/3dPrinter/3_AutoSDF-master/data/ShapeNet/SDF_v1_64"

# 创建符号链接
ln -s "$target_path" "$link_path"

# 检查符号链接是否创建成功
if [ -L "$link_path" ]; then
  echo "符号链接创建成功: $link_path -> $target_path"
else
  echo "符号链接创建失败"
fi
