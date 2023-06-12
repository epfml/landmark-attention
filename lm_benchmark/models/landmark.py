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

import math
import inspect

import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F

from . import positional_encoders, caches


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class LandmarkGroupedSoftmaxFunction(torch.autograd.Function):

    # Note that forward, setup_context, and backward are @staticmethods
    @staticmethod
    def forward(ctx, x, dim, mem_cnt, resp_mem_idx, rem_score=None):
        new_shape = list(x.shape)
        new_shape[dim] = mem_cnt # max_mem_cnt.item()
        max_by_group = x.new_zeros((*new_shape,))
        max_by_group.scatter_reduce_(src=x, index=resp_mem_idx, dim=dim, reduce="amax", include_self=False)
          
        maxes = torch.gather(max_by_group, dim, resp_mem_idx)
        x_exp = torch.exp(x - torch.where(torch.isinf(maxes), 0, maxes))
        cumsum_by_group = torch.zeros_like(max_by_group, dtype=x_exp.dtype)

        cumsum_by_group.scatter_add_(dim, resp_mem_idx, x_exp, )
        denom = torch.gather(cumsum_by_group, dim, resp_mem_idx)

        probs = torch.where(denom < 0.5, 0, x_exp / denom)
        
        
        ctx.mem_cnt = mem_cnt
        ctx.dim = dim
        ctx.save_for_backward(resp_mem_idx, probs)

        return probs

    @staticmethod
    def backward(ctx, grad_probs):
        mem_cnt = ctx.mem_cnt
        dim = ctx.dim
        resp_mem_idx, probs = ctx.saved_tensors
        grad_x = grad_dim = grad_mem_cnt = grad_resp_mem_idx = None

        if ctx.needs_input_grad[0] or ctx.needs_input_grad[4]:
            grad_pair = grad_probs * probs

            new_shape = list(probs.shape)
            new_shape[dim] = mem_cnt # max_mem_cnt.item()
            cumsum_by_group = grad_pair.new_zeros((*new_shape,))
            cumsum_by_group.scatter_add_(dim, resp_mem_idx, grad_pair)


        if ctx.needs_input_grad[0]:
            grad_sum = torch.gather(cumsum_by_group, dim, resp_mem_idx)
            grad_x = grad_pair - probs * grad_sum
        assert not ctx.needs_input_grad[1]
        assert not ctx.needs_input_grad[2]
        assert not ctx.needs_input_grad[3]

        return grad_x, grad_dim, grad_mem_cnt, grad_resp_mem_idx

def landmark_grouped_softmax(x, dim, is_mem, attn_mask,  last_section_mask, return_group_prob,
                             max_mem_cnt, p_group_dropout=None):
    
    mask = attn_mask

    last_and_rest_mask = last_section_mask | mask

    full_access_mask =  is_mem | last_and_rest_mask

    # assert max_mem_cnt >= (is_mem.sum(dim=dim).max().view((1,)) + 2 + 1).item()
    mem_group_idx = torch.cumsum(is_mem, dim=dim)
    mem_bucket_id = max_mem_cnt - 1

    resp_mem_idx = torch.where(last_and_rest_mask, 
                                max_mem_cnt - 1,
                                torch.where(is_mem, mem_bucket_id, mem_group_idx))
    probs = LandmarkGroupedSoftmaxFunction.apply(x, dim, max_mem_cnt, resp_mem_idx)


    new_shape = list(x.shape)
    new_shape[dim] = max_mem_cnt
    group_prob = probs.new_zeros((*new_shape, ))
    group_prob.scatter_(dim, torch.where(is_mem, mem_group_idx - 1, max_mem_cnt - 1), probs)
    if p_group_dropout is not None:
        group_prob = torch.nn.functional.dropout(group_prob, p=p_group_dropout, inplace=True)
        group_prob.select(dim, max_mem_cnt - 1).copy_((probs * last_section_mask).sum(dim=dim, )).div_(1 - p_group_dropout)
    else:
        group_prob.select(dim, max_mem_cnt - 1).copy_((probs * last_section_mask).sum(dim=dim, ))
    probs = probs.mul(torch.where(full_access_mask, last_section_mask, torch.gather(group_prob, dim, resp_mem_idx)))


    return probs, (group_prob if return_group_prob else None)

def softmax_ignore_mem(x, dim, is_mem, *args, **kwargs):
    x = x.masked_fill(is_mem, float('-inf'))
    return torch.nn.functional.softmax(x, dim=dim), None

class CausalSelfAttention(nn.Module):

    def __init__(self, config, lm_cache):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.config = config
        
        self.allow_cache_during_training = getattr(config, "allow_cache_during_training", False)

        self.cache_storage = None
        if not config.postpone_lm_cache:
            self.init_cache_storage(lm_cache)

        SOFTMAX_FUNC = {
            "nomem": lambda x, dim, *args, **kwargs: (nn.functional.softmax(x, dim=dim), None),
            "mem_opt": landmark_grouped_softmax,
            "ignore_mem": softmax_ignore_mem,
        }

        self.softmax_func = SOFTMAX_FUNC[config.softmax_func]

        if self.config.enable_rem_score:
            self.rem_token_embedding = nn.Linear(config.n_embd, 1, bias=config.bias)
        else:
            self.rem_token_embedding = None

        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.sequence_length, config.sequence_length))
                                    .view(1, 1, config.sequence_length, config.sequence_length))

    def init_cache_storage(self, lm_cache):
        if self.cache_storage is not None:
            return
        print("Init Storage")
        self.cache_storage = lm_cache.get_storage_for_layer(self)

    def forward(self, x, is_mem, pos_emb_closure, cache_context, start_index):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        q = pos_emb_closure.adapt_queries(q, start_index=start_index)
        if cache_context is not None and self.cache_storage is not None:
            att_prefix, cache_values_dict = \
                self.cache_storage.retrieve_for_query(q, cache_context, pos_emb_closure, start_index)
            if self.training and att_prefix is not None and not self.allow_cache_during_training:
                raise ValueError("Cache is not allowed during training")
        else:
            att_prefix = None
        
        k_before_pos = k
        is_mem_orig = is_mem
        k = pos_emb_closure.adapt_keys(k, start_index=start_index)
        

        # manual implementation of attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = pos_emb_closure.adapt_attention_before_softmax(att, start_query_index=start_index, start_key_index=start_index)
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        attn_mask = self.bias[:,:,:T,:T] == 0
        if att_prefix is not None:
            prefix_size = att_prefix.shape[-1]
            current_size = att.shape[-1]
            att = torch.cat((att_prefix, att), dim=-1)
            if 'is_mem' in cache_values_dict:
                is_mem = torch.cat((cache_values_dict['is_mem'], is_mem), dim=-1)
            else:
                is_mem = torch.cat((is_mem.new_zeros((B, prefix_size)), is_mem), dim=-1)
            attn_mask = torch.cat((attn_mask.new_zeros((1, 1, T, prefix_size)), attn_mask), dim=-1)

        is_mem_mat = is_mem.unsqueeze(1).unsqueeze(2).repeat((1, self.n_head, 1, 1))

        # The following block should be inside landmark_grouped_softmax
        # however, moving it there causes problems with torch.compile.
        is_mem = is_mem_mat
        dim = -1
        with torch.no_grad():
            mem_ids = torch.where(attn_mask, 0., torch.cumsum(is_mem, dim) - is_mem.int())
            last_section_mask = torch.amax(mem_ids, dim, keepdim=True) == mem_ids
            # the landmark token for the current block should be ignored, 
            # even if we are processing the landmark token itself.
            mask = attn_mask.logical_or(is_mem & last_section_mask) 
            last_section_mask.logical_and_(~mask)
            is_mem = is_mem.logical_and(~mask)
        att = att.masked_fill_(mask, float('-inf')) 
        # End of block
        
        att, group_prob = self.softmax_func(att, dim=-1, 
                                            is_mem=is_mem, attn_mask=mask,
                                            last_section_mask=last_section_mask,
                                            return_group_prob=False,
                                            max_mem_cnt=self.config.max_groups_for_softmax,
                                            p_group_dropout=self.config.group_dropout if self.training else None)
        att = self.attn_dropout(att)
        if att_prefix is not None:
            att_prefix, att = torch.split(att, (prefix_size, current_size), dim=-1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        if att_prefix is not None:
            cache_v = cache_values_dict['v']
            if cache_v.ndim == v.ndim:
                y += att_prefix @ cache_v
            elif cache_v.ndim == v.ndim + 1:
                y += (att_prefix.unsqueeze(3) @ cache_v).squeeze(3)
            else:
                raise NotImplementedError
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))

        if cache_context is not None and self.cache_storage is not None:
            with torch.no_grad():
                self.cache_storage.store_in_cache(k_before_pos, {'v': v, 'is_mem': is_mem_orig})
            

        return y, group_prob


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config, lm_cache):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config, lm_cache)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, is_mem, pos_emb_closure, cache_context, start_index):
        x_attn, group_prob = self.attn(self.ln_1(x), is_mem, pos_emb_closure, cache_context, start_index)
        x = x + x_attn
        x = x + self.mlp(self.ln_2(x))
        return x, group_prob
    

class GPTBase(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.sequence_length is not None
        self.config = config
        self.tokenizer = tiktoken.get_encoding("gpt2")

        self.lm_cache = None
        if not config.postpone_lm_cache:
            self.init_cache()

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = positional_encoders.get_encoder(config.positional_encoder)(config),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(
                config, self.lm_cache) for idx in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def init_cache(self):
        self.lm_cache = caches.get_cache(self.config.lm_cache)(self.config)
        for m in self.modules():
            if hasattr(m, "init_cache_storage"):
                m.init_cache_storage(self.lm_cache)

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= sum(p.numel() for p in self.transformer.wpe.parameters()) # TODO: Why do we need this?
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, get_logits=False, use_cache=False):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.sequence_length, f"Cannot forward sequence of length {t}, block size is only {self.config.sequence_length}"
        
        # forward the GPT model itself
        if use_cache:
            idx, index_shift, cache_context = self.lm_cache(idx)
        else:
            index_shift = 0
            cache_context = None
        idx, pos_emb_closure = self.transformer.wpe(idx) # position embeddings of shape (1, t, n_embd)
        is_mem = idx == self.config.landmark_id
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        x = pos_emb_closure.adapt_model_input(tok_emb, start_index=index_shift)
        # print("Index shift: ", index_shift)
        x = self.transformer.drop(x)
        group_prob = None
        for block in self.transformer.h:
            x, group_prob = block(x, is_mem, pos_emb_closure, cache_context, start_index=index_shift)
        x = self.transformer.ln_f(x)

        if use_cache:
            x = self.lm_cache.get_final_logits(x)
        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None
        logits = logits if get_logits else None
        return {'logits': logits, 'loss': loss}

    def clear_state(self):
        self.lm_cache.clear_state()
    
    def crop_sequence_length(self, sequence_length):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert sequence_length <= self.config.sequence_length
        self.config.sequence_length = sequence_length
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:sequence_length])
        for block in self.transformer.h:
            block.attn.bias = block.attn.bias[:,:,:sequence_length,:sequence_length]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        # TODO
        pass

    def get_parameter_group_specs(self):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # subtle: 'transformer.wte.weight' and 'lm_head.weight' are tied, so they
        # will appear in the no_decay and decay sets respectively after the above.
        # In addition, because named_parameters() doesn't return duplicates, it
        # will only return the first occurence, key'd by 'transformer.wte.weight', below.
        # so let's manually remove 'lm_head.weight' from decay set. This will include
        # this tensor into optimization via transformer.wte.weight only, and not decayed.
        decay.remove('lm_head.weight')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        return [
            {"params": sorted(list(decay))},
            {"params": sorted(list(no_decay)), "weight_decay": 0.0},
        ]

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at sequence_length
            idx_cond = idx if idx.size(1) <= self.config.sequence_length else idx[:, -self.config.sequence_length:]
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond, get_logits=True)['logits']
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
    
    @torch.no_grad()
    def generate_from_string(self, in_str, max_new_tokens, temperature=1.0, top_k=None):
        idx = torch.tensor(self.tokenizer.encode(in_str, allowed_special={"<|endoftext|>"})).view(1,-1).to(self.lm_head.weight.device)
        out_idx = self.generate(idx, max_new_tokens, temperature, top_k).view(-1).to('cpu').numpy()
        return self.tokenizer.decode(out_idx)
