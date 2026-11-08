# Copyright 2026 The HuggingFace Team. All rights reserved.
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

from peft.tuners.lora.layer import Linear

from .utils import ConvRotInt8Linear


class ConvRotLoraLinear(Linear):
    def _get_in_out_features(self, module):
        return module.in_features, module.out_features

    @property
    def compute_dtype(self):
        return self.get_base_layer().compute_dtype

    def _move_adapter_to_device_of_base_layer(self, adapter_name, device=None):
        super()._move_adapter_to_device_of_base_layer(adapter_name, device)
        self.lora_A[adapter_name].to(dtype=self.compute_dtype)
        self.lora_B[adapter_name].to(dtype=self.compute_dtype)

    def merge(self, safe_merge=False, adapter_names=None):
        raise ValueError("ConvRot INT8 LoRA adapters cannot be fused. Keep the adapters loaded during inference.")


def register_convrot_lora(adapter_config):
    if adapter_config.use_dora:
        raise ValueError("ConvRot INT8 supports LoRA, but not DoRA.")
    if adapter_config.init_lora_weights not in (True, False, "gaussian"):
        raise ValueError("ConvRot INT8 does not support LoRA initialization that modifies the base weights.")
    adapter_config._register_custom_module({ConvRotInt8Linear: ConvRotLoraLinear})
