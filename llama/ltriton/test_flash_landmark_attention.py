import torch
import math

from flash_landmark_attention import fused_landmark_attention


class LandmarkGroupedSoftmaxFunction(torch.autograd.Function):

    # Note that forward, setup_context, and backward are @staticmethods
    @staticmethod
    def forward(ctx, x, dim, mem_cnt, resp_mem_idx):
        new_shape = list(x.shape)
        new_shape[dim] = mem_cnt # max_mem_cnt.item()
        max_by_group = x.new_zeros((*new_shape,))
        max_by_group.scatter_reduce_(src=x, index=resp_mem_idx, dim=dim, reduce="amax", include_self=False)
            
        maxes = torch.gather(max_by_group, dim, resp_mem_idx)
        #x_exp = torch.exp(x - torch.where(torch.isinf(maxes), 0, maxes))
        x_exp = torch.exp((x - maxes).to(torch.float32))

        cumsum_by_group = torch.zeros_like(max_by_group, dtype=x_exp.dtype)

        cumsum_by_group.scatter_add_(dim, resp_mem_idx, x_exp, )
        denom = torch.gather(cumsum_by_group, dim, resp_mem_idx)

        #probs = torch.where(denom < 0.5, 0, x_exp / denom)
        probs = x_exp / denom
        
        
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

def landmark_grouped_softmax(x, dim, is_mem, last_section_mask):
    
    last_and_rest_mask = last_section_mask # | mask

    full_access_mask =  is_mem | last_and_rest_mask

    max_mem_cnt = 16
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
    probs = probs.mul(torch.where(full_access_mask, last_section_mask, torch.gather(group_prob, dim, resp_mem_idx)))


    return probs

batch = 2
nheads = 8
seqlen_q = 1024
seqlen_k = 1024 #512
d = 128
use_I_for_v = False
mem_freq = 63
q = torch.rand((batch, seqlen_q, nheads, d)).cuda().to(torch.bfloat16).transpose(1, 2)
k = torch.rand((batch, seqlen_k, nheads, d)).cuda().to(torch.bfloat16).transpose(1, 2)
if not use_I_for_v:
    v = torch.rand((batch, seqlen_k, nheads, d)).cuda().to(torch.bfloat16).transpose(1, 2)
else:
    v = torch.eye(seqlen_k, d).cuda().to(torch.bfloat16)
    v = v.view(1, 1, seqlen_k, d).expand(batch, nheads, seqlen_k, d)
q.requires_grad = True
k.requires_grad = True
v.requires_grad = True
block_size = mem_freq + 1
is_mem = torch.arange(0, seqlen_k, device=q.device) % block_size == (block_size - 1)
out = fused_landmark_attention(q, k, v, is_mem, block_size=block_size)

def f():
    import math
    att = q @ k.transpose(-1, -2) / math.sqrt(d)
    att_mask = torch.tril(torch.ones((1, 1, seqlen_q, seqlen_k), device=q.device), diagonal=seqlen_k - seqlen_q)== 1.

    last_section_mask = (torch.arange(0, seqlen_k, device=q.device) // (mem_freq + 1))[None, :] == (torch.arange(seqlen_k - seqlen_q, seqlen_k, device=q.device) // (mem_freq + 1))[:, None]

    last_section_mask = last_section_mask.unsqueeze(0).unsqueeze(1)
    is_mem_ = is_mem.view(1, 1, 1, seqlen_k)
    mask = att_mask & ~(last_section_mask & is_mem_)
    last_section_mask = last_section_mask & mask
    is_mem_ = is_mem_ & mask
    is_mem_ = is_mem_.expand(batch, nheads, seqlen_q, seqlen_k)
    last_section_msak = last_section_mask.expand(batch, nheads, seqlen_q, seqlen_k)
    att.masked_fill_(~mask, float("-inf"))
    

    att = landmark_grouped_softmax(att, -1, is_mem_, last_section_mask).to(q.dtype)
    att.masked_fill_(~mask, 0.0)
    exact_out = att @ v
    return exact_out    
exact_out = f()

def make_f_grad(func):
    def f_():
        exact_out = func()
        return torch.autograd.grad((exact_out**2).sum(), [q, k, v])
    return f_


def f_exact():
    return fused_landmark_attention(q, k, v, is_mem, block_size=block_size)

def f_torch():
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

if use_I_for_v and d >= seqlen_k:
    assert torch.allclose(out.sum(-1), torch.ones_like(out.sum(-1)), atol=1e-03, rtol=1e-03), out.sum(-1)
    assert torch.allclose(exact_out.sum(-1), torch.ones_like(exact_out.sum(-1)))

#print("Exact", exact_out[:, :, mem_freq-1:mem_freq+2])
#print("Fused", out[:, :, mem_freq-1:mem_freq+2])
assert torch.allclose(out, exact_out, rtol=1e-02, atol=1e-02), (out, exact_out)
#print("Diff", (out - exact_out).max(dim=-2))


#print(last_section_mask[0, 0])
#print(att[0, 0])
#print(is_mem[0, 0])
#print(is_mem[0, 0, -1])

grads = torch.autograd.grad((out ** 2).sum(), [q, k, v])
exact_grads = torch.autograd.grad((exact_out ** 2).sum(), [q, k, v])
#print(len(exact_grads))

for grad, exact_grad, t in zip(grads, exact_grads, ["q", "k", "v"]):
    torch.set_printoptions(sci_mode=False)
    if not torch.allclose(grad, exact_grad, rtol=4e-02, atol=4e-02):
        #print((grad, exact_grad, t))
        print("Failed d", t)
    #print(t, (grad - exact_grad).max())
    #print(t, torch.argmax((grad - exact_grad).amax(dim=-1), dim=-1))

print("Done once")    
#print(v.grad)

import timeit
print("Exact: ", timeit.timeit(f, number=500))
print("Fused: ", timeit.timeit(f_exact, number=500))
print("Torch: ", timeit.timeit(f_torch, number=500))

print("Exact Grad: ", timeit.timeit(make_f_grad(f), number=500))
print("Fused Grad: ", timeit.timeit(make_f_grad(f_exact), number=500))
print("Torch Grad: ", timeit.timeit(make_f_grad(f_torch), number=500))
