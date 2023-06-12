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
import sys
import numpy as np
import torch
import inspect
import json
import copy
import argparse
import random
import wandb
import logging

from tqdm import tqdm

import config
import models
from data import get_dataset, prepare_dataset
from optim.base import train_base
import distributed
from optim.utils import get_batch



def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--checkpoint', type=str, required=True)
    
    args, rem_args = parser.parse_known_args()

    if os.path.isfile(args.checkpoint):
        args.checkpoint, args.checkpoint_filename = os.path.split(args.checkpoint)
    else:
        args.checkpoint_filename = "ckpt.pt"

    with open(os.path.join(args.checkpoint, "summary.json")) as f:
        summary = json.load(f)

    for k, v in summary['args'].items():
        if k not in ["device", "dtype"]:
            setattr(args, k, v)

    return config.parse_args_with_format(format=args.config_format, base_parser=argparse.ArgumentParser(allow_abbrev=False), args=rem_args, namespace=args)


def get_as_batch(data, seq_length, batch_size, device='cpu', sample_size=None):
    all_ix = list(range(0, len(data), seq_length))
    assert all_ix[-1] + seq_length + 1 > len(data)
    all_ix.pop()
    if sample_size is not None:
        all_ix = np.random.choice(all_ix, size=sample_size // seq_length, replace=False).tolist()
    
    idx = 0
    for idx in range(0, len(all_ix), batch_size):
        ix = all_ix[idx:idx+batch_size]
        assert all([idx + seq_length + 1 <= len(data) for idx in ix])
        x = torch.stack([torch.from_numpy((data[i:i+seq_length]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+seq_length]).astype(np.int64)) for i in ix])
        if device != 'cpu':
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        yield x, y

def iceildiv(x, y):
    return (x + y - 1) // y

def evaluate(model, data, iterations, acc_steps, batch_size, sequence_length, distributed_backend, extra_args):
    device_type = 'cuda' if 'cuda' in str(extra_args.device) else 'cpu'
    type_ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(
        device_type=device_type, dtype=extra_args.dtype)  # extra_args.dtype)
    itr, substep, best_val_loss, text_table = 0, 0, float('inf'), None # best_val_loss not used atm, early stopping not recommended but possible 

    stats = {}

    num_substeps_per_epoch = len(data['val']) // (batch_size * sequence_length)
    
    if not extra_args.no_compile:
        print(f"Compiling model ...")
        import torch._dynamo as torchdynamo
        torchdynamo.config.guard_nn_modules = True
        # torchdynamo.config.log_level = logging.DEBUG
        model = torch.compile(model) # requires pytorch 2.0+

    model.eval()

    loss_list_val, acc_list = [], []
    loss_step_list_val = []

    max_num_batches = 400
    with torch.no_grad():
        mid_length = extra_args.mid_length
        print(f"Sending sub-sequences of length at most {mid_length}")
        seq_length = extra_args.eval_seq_length 
        print(f"Using seq length {seq_length}")
        torch.set_printoptions(sci_mode=False)
        for idx, (x, y) in tqdm(
            enumerate(
                get_as_batch(
                    data['val'], 
                    seq_length, 
                    batch_size, 
                    device=extra_args.device, 
                    sample_size=extra_args.eval_sample_size
                )
            ),
            total=iceildiv(
                extra_args.eval_sample_size // seq_length if extra_args.eval_sample_size is not None else 
                iceildiv(len(data['val']), seq_length), 
                batch_size
            )
        ):
            val_loss = 0.
            acc = 0.
            cnt = 0
            model.clear_state()
            for part_idx, i in enumerate(range(0, x.shape[1], mid_length)):
                part_len = x[:, i:i + mid_length].shape[1]
                with type_ctx:
                    outputs = model(x[:, i:i + mid_length], targets=y[:, i:i+mid_length].contiguous(), get_logits=True, use_cache=extra_args.use_cache)
                val_loss = outputs['loss'] * part_len + val_loss 
                acc = ((outputs['logits'].argmax(-1) == y[:, i:i+mid_length]).float().sum()) + acc 
                cnt += part_len
                while len(loss_step_list_val) <= part_idx:
                    loss_step_list_val.append([])
                loss_step_list_val[part_idx].append(outputs['loss'].item())
            val_loss /= cnt
            acc /= cnt
            
            loss_list_val.append(val_loss.item())
            acc_list.append(acc.item())
        

    stats['val_acc'] = torch.as_tensor(acc_list).mean().item()
    stats['val_loss'] = torch.as_tensor(loss_list_val).mean().item()
    stats['val_perplexity'] = 2.71828 ** stats['val_loss']
    stats['val_perplexity_per_chunk'] = torch.exp(torch.as_tensor(loss_step_list_val).mean(dim=1))

    return stats

def main(args): 


    torch.backends.cuda.matmul.allow_tf32 = True # allows us to make sure we're able to use tensorfloat32 during training
    torch.backends.cudnn.allow_tf32 = True

    distributed_backend = distributed.make_backend_from_args(args)
    args = distributed_backend.get_adjusted_args_for_process(args)

    args.device = torch.device(args.device)
    torch.cuda.set_device(args.device)
    device_type = 'cuda' if 'cuda' in str(args.device) else 'cpu'
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    print(f"Loading dataset '{args.dataset}'")

    if distributed_backend.is_master_process():
        prepare_dataset(args)
    distributed_backend.sync()
    
    data = get_dataset(args) # data is a dict: {'train': train_tokenized, 'val': eval_tokenized}
        
    print(f"Num training tokens: {len(data['train'])}")
    print(f"Num validation tokens: {len(data['val'])}")
    
    model = models.make_model_from_args(args).to(args.device)

    checkpoint = torch.load(os.path.join(args.checkpoint, args.checkpoint_filename))
    model.load_state_dict({x: y for x, y in checkpoint['model'].items() if "attn.bias" not in x and "wpe" not in x}, strict=False)

    model = distributed_backend.transform_model(model)
    
    print(f"\Evaluating model={args.model} \n{vars(args)}\n")

    stats = evaluate(model, data, args.iterations, args.acc_steps, args.batch_size, args.sequence_length, 
                  distributed_backend=distributed_backend,
                  extra_args=args)

    print(stats)
    
    distributed_backend.finalize()


if __name__ == "__main__":
    args = get_args()
    main(args)
