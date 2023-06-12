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

import torch
import torch.nn.functional as F
import wandb
import time 
import copy
import traceback

from .utils import get_batch, save_checkpoint


@torch.no_grad()
def eval(model, data_tensor, sequence_length, total_sequence_length, batch_size, device='cpu', max_num_batches=24, ctx=nullcontext()):
    assert model.training == False

    loss_list_val, acc_list = [], []

    for _ in range(max_num_batches): 
        x, y = get_batch(data_tensor, total_sequence_length, batch_size, device=device)
        model.clear_state()
        total_loss = None
        for idx in range(0, x.shape[1], sequence_length):
            x_part = x[:, idx:idx+sequence_length]
            y_part = y[:, idx:idx+sequence_length].contiguous()
            with ctx:
                outputs = model(x_part, targets=y_part, get_logits=True, use_cache=True)
            val_loss = outputs['loss']
            if idx == 0:
                total_loss = val_loss
            else:
                total_loss += val_loss
        loss_list_val.append(total_loss)
        acc_list.append((outputs['logits'].argmax(-1) == y_part).float().mean())

    val_acc = torch.stack(acc_list).mean().item()
    val_loss = torch.stack(loss_list_val).mean().item()
    val_perplexity = 2.71828 ** val_loss

    return val_acc, val_loss, val_perplexity

def train_xl(model, opt, data, scheduler, iterations, acc_steps, batch_size, sequence_length, eval_freq, ckpt_path, distributed_backend, extra_args):
    device_type = 'cuda' if 'cuda' in str(extra_args.device) else 'cpu'
    type_ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(
        device_type=device_type, dtype=extra_args.dtype)  # extra_args.dtype)
    itr, substep, best_val_loss, text_table = 0, 0, float('inf'), None # best_val_loss not used atm, early stopping not recommended but possible 

    stats = {'train_loss': [], 'val_loss': [], 'val_pp': [], 'val_acc': []}

    num_substeps_per_epoch = len(data['train']) // (batch_size * sequence_length)
    
    if not extra_args.no_compile:
        print(f"Compiling model ...")
        import torch._dynamo as torchdynamo
        torchdynamo.config.guard_nn_modules = True
        model = torch.compile(model) # requires pytorch 2.0+

    model.train()

    if extra_args.postpone_lm_cache:
        distributed_backend.get_raw_model(model).init_cache()

    t0 = time.time()
    while itr < iterations:
        for microstep_idx in range(acc_steps):  # gradient accumulation
            x, y = get_batch(data['train'], extra_args.total_sequence_length, batch_size, device=extra_args.device)
            distributed_backend.get_raw_model(model).clear_state()
            total_loss = None
            for idx in range(0, x.shape[1], extra_args.sequence_length):
                with type_ctx:
                    with distributed_backend.get_context_for_microstep_forward(model=model, microstep_idx=microstep_idx, gradient_accumulation_steps=acc_steps):
                        outputs = model(x[:, idx:idx+extra_args.sequence_length], targets=y[:, idx:idx+extra_args.sequence_length].contiguous(), use_cache=True)
                
                loss = outputs['loss']
                loss.backward()
                if idx == 0:
                    total_loss = loss
                else:
                    total_loss += loss
            substep += 1

        opt.step()
        scheduler.step()
        opt.zero_grad(set_to_none=True)
        itr += 1

        if itr % eval_freq == 0 or itr == iterations: # from here it's only evaluation code, all the training is above
            if distributed_backend.is_master_process():
                t1 = time.time()
                dt = t1 - t0
                epoch = substep//num_substeps_per_epoch

                model.eval()
                train_loss = loss.detach().cpu().item()
                current_lr = scheduler.get_last_lr()[0] if scheduler is not None else extra_args.lr
                val_acc, val_loss, val_perplexity = eval(distributed_backend.get_raw_model(model), data['val'], sequence_length, extra_args.total_sequence_length, 
                                                         batch_size, extra_args.device, max_num_batches=24, ctx=type_ctx)

                print_string = f"{epoch}/{itr} [train] loss={train_loss:.3f} [val] loss={val_loss:.3f}, pp={val_perplexity:.2f}, acc={val_acc:3f}"
                print_string += f" [time per itr] {dt*1000/eval_freq:.2f}ms"
                if scheduler is not None:
                    print_string += f" [lr] {current_lr:.5f}"
                print(print_string)

                if extra_args.wandb:
                    wandb.log({
                        "iter": itr,
                        "train/loss": train_loss,
                        "val/loss": val_loss,
                        "val/perplexity": val_perplexity,
                        "val/acc": val_acc,
                        "lr": current_lr,
                    })

                model.train()
                t0 = time.time()

    if distributed_backend.is_master_process():
        print(f"saving checkpoint to {ckpt_path}")
        save_checkpoint(distributed_backend=distributed_backend,
                        model=model,
                        opt=opt,
                        scheduler=scheduler,
                        itr=itr,
                        ckpt_path=ckpt_path)

    return stats
