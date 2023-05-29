# Copyright 2023 Amirkeivan Mohtashami, Martin Jaggi
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

from contextlib import nullcontext

from .backend import DistributedBackend


class SinlgeNodeBackend(DistributedBackend):

    def transform_model(self, model):
        return model

    def get_context_for_microstep_forward(self, *args, **kwargs):
        return nullcontext()

    def get_adjusted_args_for_process(self, args):
        return args

    def is_master_process(self) -> bool:
        return True

    def get_raw_model(self, model):
        return model

    def get_world_size(self):
        return 1

    def sync(self):
        pass

    def translate_model_parameter_name_for_node(self, parameter_name):
        return [parameter_name]
