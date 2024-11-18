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
import torch
from torch import nn
import numpy as np
import types


def slim_model_from_state_dict(model, state_dict):
  """Modify modules according to their weights in the state dict and
  load the state dict to the model.

    Args:
      model: An torch.nn.Module instance to load state dict.
      state_dict: A state dict to be loaded.

    Returns:
      A modified model that matchs weights shape in the state dict.

  """
  for key, module in model.named_modules():
    weight_key = key + '.weight'
    bias_key = key + '.bias'
    if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d)):
      module.weight = nn.Parameter(state_dict[weight_key])
      module.bias = nn.Parameter(state_dict[bias_key])
      module.running_mean = state_dict[key + '.running_mean']
      module.running_var = state_dict[key + '.running_var']
      module.num_features = module.weight.size(0)
    elif isinstance(module, (nn.Conv2d, nn.Conv3d)):
      assert module.groups == 1
      module.weight = nn.Parameter(state_dict[weight_key])
      if bias_key in state_dict:
        module.bias = nn.Parameter(state_dict[bias_key])
      module.out_channels = module.weight.size(0)
      module.in_channels = module.weight.size(1)
    elif isinstance(module, (nn.ConvTranspose2d, nn.ConvTranspose3d)):
      assert module.groups == 1
      module.weight = nn.Parameter(state_dict[weight_key])
      if bias_key in state_dict:
        module.bias = nn.Parameter(state_dict[bias_key])
      module.in_channels = module.weight.size(0)
      module.out_channels = module.weight.size(1)
    elif isinstance(module, nn.Linear):
      module.weight = nn.Parameter(state_dict[weight_key])
      if bias_key in state_dict:
        module.bias = nn.Parameter(state_dict[bias_key])
      module.out_features = module.weight.size(0)
      module.in_features = module.weight.size(1)
    else:
      pass
  model.load_state_dict(state_dict)
  return model


