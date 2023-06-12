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

import os
import math
from contextlib import contextmanager

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, get_world_size, barrier

from .backend import DistributedBackend


class DataParallelDistributedBackend(DistributedBackend):

    def __init__(self, args):
        self.rank = int(os.environ.get('RANK', -1))
        assert self.rank != -1, "DDP backend can not be used without rank"
        assert "cuda" in args.device, "DDP backend can not be used on non-CUDA devices"
        init_process_group(backend=args.distributed_backend)
        self.local_rank = int(os.environ['LOCAL_RANK'])

    def get_adjusted_args_for_process(self, args):
        effective_batch_size = args.batch_size * args.acc_steps
        world_size = self.get_world_size()
        if effective_batch_size % world_size != 0:
            raise ValueError(f"Effective batch size "
                             "{effective_batch_size} is not divisible "
                             "by the world size {world_size}.")
        acc_steps_div = math.gcd(args.acc_steps, world_size)
        args.acc_steps = args.acc_steps // acc_steps_div
        args.batch_size = args.batch_size // (world_size // acc_steps_div)
        args.device = f'cuda:{self.local_rank}'
        args.seed = args.seed + self.local_rank
        return args

    def transform_model(self, model):
        return DDP(model, device_ids=[self.local_rank], find_unused_parameters=True)

    @contextmanager
    def get_context_for_microstep_forward(self, model, microstep_idx, gradient_accumulation_steps):
        model.require_backward_grad_sync = (
            microstep_idx == gradient_accumulation_steps - 1)
        yield

    def is_master_process(self) -> bool:
        return self.rank == 0

    def get_raw_model(self, model):
        return model.module

    def translate_model_parameter_name_for_node(self, parameter_name):
        return [f'module.{parameter_name}']

    def get_world_size(self):
        return get_world_size()
    
    def sync(self):
        barrier()

    def finalize(self):
        destroy_process_group()
