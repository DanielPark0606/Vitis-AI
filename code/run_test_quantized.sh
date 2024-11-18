# Copyright 2019 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
export W_QUANT=1
export CUDA_VISIBLE_DEVICES=2

ratio=0.5
echo " Test QAT deployable model "
cd ../
python code/train/resnet50_qat.py \
        --data_dir data/Imagenet/ \
        --mode test \
        --pretrained float/prune_${ratio}.pth \
        --output_dir quantized/ \
        --sparse_model \
        --val_batch_size 128 \
