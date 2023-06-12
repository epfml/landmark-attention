import math
import triton
import torch
import triton.language as tl

@triton.jit
def _fwd_kernel(#debug, sdz, sdh, sdm, sdn,
    Q, K, V, sm_scale, 
    Out,
    sqz, sqh, sqm, sqd, # shape = (Z,H,N_CTX_Q,D)
    skz, skh, skn, skd, # shape = (Z,H,N_CTX_KV,D)
    svz, svh, svn, svd, # shape = (Z,H,N_CTX_KV,D)
    soz, soh, som, sod, # shape = (Z,H,N_CTX_Q,D)
    L, M,
    Z, H, N_CTX_Q, N_CTX_KV, 
    BLOCK: tl.constexpr, # will load BLOCK_M queries, and compute self attention by blocks of BLOCK_N keys
    BLOCK_DMODEL: tl.constexpr, # dimensionality of heads: D
    N_PREFIX_Q: tl.constexpr,
):
    
    start_m = tl.program_id(0) # idx of sequence length chunk of size 128 (BLOCK_N)
    off_hz = tl.program_id(1) # idx of head_batch (unique idx for each head in each batch)

    BLOCK_M: tl.constexpr = BLOCK
    BLOCK_N: tl.constexpr = BLOCK

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M) # indices of queries we want to process
    offs_m_real = (start_m + N_PREFIX_Q) * BLOCK_M + tl.arange(0, BLOCK_M)  # indices of queries we want to process
    offs_m_real += tl.where(tl.arange(0, BLOCK_M) == BLOCK_M - 1, -1, 0)
    offs_n = tl.arange(0, BLOCK_N) # indices of keys we want to process, we start from [0, BLOCK_N-1] and update in the loop
    offs_d = tl.arange(0, BLOCK_DMODEL) # we want to process all the dimensions of a given head

    offs_q = off_hz * sqh + offs_m[:, None] * sqm + offs_d[None, :] * sqd # Q.view(Z*H,N_CTX_Q,D)[off_hz, start_m*BLOCK_M:(start_m+1)*BLOCK_M, :].squeeze() that's a BLOCK_M*D matrix
    offs_k = off_hz * skh + offs_n[None, :] * skn + offs_d[:, None] * skd # K.view(Z*H,N_CTX_KV,D)[off_hz, 0:BLOCK_N, :].transpose(1,2).squeeze() that's a D*BLOCK_N matrix
    offs_v = off_hz * svh + offs_n[:, None] * svn + offs_d[None, :] * svd # V.view(Z*H,N_CTX_KV,D)[off_hz, 0:BLOCK_N, :].squeeze() that's a BLOCK_N*D matrix

    # pointers to m and l
    m_prev = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_prev = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # Load values
    q_vals = tl.load(Q + offs_q, mask=offs_m[:, None] < N_CTX_Q, other=0) 
    

    for start_n in range(0, (N_PREFIX_Q + start_m)):
        # Load values for K and K_idx
        k_vals = tl.load(K + offs_k, mask=offs_n[None, :] < N_CTX_KV, other=0)

        # compute qk
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=q_vals.dtype)
        qk += tl.dot(q_vals, k_vals, allow_tf32=False)
        qk *= sm_scale
        # causal masking
        qk = tl.where(offs_m_real[:,None] >= offs_n[None,:], qk, float("-inf"))
        landmark_qk = tl.max(tl.where(tl.arange(0, BLOCK_N)[None, :] == BLOCK_N - 1, qk, float("-inf")), 1)
        normal_qk = tl.where(tl.arange(0, BLOCK_N)[None, :] == BLOCK_N - 1, float("-inf"), qk)
        normal_m = tl.max(normal_qk, 1)
        normal_p = tl.exp(normal_qk - normal_m[:, None])
        normal_denom = tl.sum(normal_p, 1)

        # compute attention weights
        m_curr = tl.maximum(landmark_qk, m_prev) # compute new m
        m_curr_ = m_curr # tl.where(m_curr != float('-inf'), m_curr, float(0.0))  # ADDITIONAL CHECK IF YOU GET NaNs
        l_prev *= tl.exp(m_prev - m_curr_) # correct old l
        landmark_p = tl.exp(landmark_qk - m_curr_)
        l_curr = landmark_p + l_prev 
        l_rcp = 1. / l_curr # rescale operands of matmuls
        # l_rcp = tl.where((l_rcp == float('inf')), 0, l_rcp)  # ADDITIONAL CHECK IF YOU GET NaNs
        landmark_p *= l_rcp

        acc *= (l_prev * l_rcp)[:, None] # weight for each value vector
        # update acc
        v_vals = tl.load(V + offs_v, mask=offs_n[:, None] < N_CTX_KV, other=0)
        acc += tl.dot((landmark_p[:, None] * normal_p / normal_denom[:, None]).to(Q.dtype.element_ty), v_vals, allow_tf32=False) 


        # update m_i and l_i
        l_prev = l_curr
        m_prev = m_curr

        # update offsets
        offs_n += BLOCK_N
        offs_k += BLOCK_N * skn
        offs_v += BLOCK_N * svn

    k_vals = tl.load(K + offs_k, mask=offs_n[None, :] < N_CTX_KV, other=0)
    # compute qk
    qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=q_vals.dtype)
    qk += tl.dot(q_vals, k_vals, allow_tf32=False)
    qk *= sm_scale
    # causal masking
    qk = tl.where(offs_m_real[:,None] >= offs_n[None,:], qk, float("-inf"))

    m_curr = tl.maximum(tl.max(qk, 1), m_prev) # compute new m
    m_curr_ = m_curr#m_curr_ = tl.where(m_curr != float('-inf'), m_curr, float(0.0))

    l_prev *= tl.exp(m_prev - m_curr_) # correct old l
    p = tl.exp(qk - m_curr_[:, None])
    l_curr = tl.sum(p, 1) + l_prev 

    l_rcp = 1. / l_curr # rescale operands of matmuls
    # l_rcp = tl.where((l_rcp == float('inf')), 0, l_rcp)  # ADDITIONAL CHECK IF YOU GET NaNs
    p *= l_rcp[:, None]
    acc *= (l_prev * l_rcp)[:, None] # weight for each value vector
    # update acc
    p = p.to(Q.dtype.element_ty)
    v_vals = tl.load(V + offs_v, mask=offs_n[:, None] < N_CTX_KV, other=0)
    acc += tl.dot(p, v_vals, allow_tf32=False) 

    l_prev = l_curr
    m_prev = m_curr
 
    # store L and M
    offs_L = off_hz * N_CTX_Q + offs_m # L is of shape (Z*H, N_CTX_Q), here we point to L[off_hz, start_m*Block_M:(start_m+1)*Block_M]
    offs_M = off_hz * N_CTX_Q + offs_m
    tl.store(L + offs_L, l_prev, mask=offs_m < N_CTX_Q)
    tl.store(M + offs_M, m_prev, mask=offs_m < N_CTX_Q)
    # store results to output
    offs_o = off_hz * soh + offs_m[:, None] * som + offs_d[None, :] * sod
    tl.store(Out + offs_o, acc, mask=offs_m[:, None] < N_CTX_Q)


@triton.jit
def _bwd_preprocess(
    Out, soz, soh, som, sod,
    DO, L, slzh, slm,
    NewDO, Delta, N_CTX_Q,
    BLOCK_M: tl.constexpr, D_HEAD: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    off_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_d = tl.arange(0, D_HEAD)
    # load
    off_o = off_hz * soh + off_m[:, None] * som + off_d[None, :] * sod
    off_l = off_hz * slzh + off_m * slm
    o = tl.load(Out + off_o).to(tl.float32)#, mask=off_m[:, None] < N_CTX_Q, other=0.0).to(tl.float32)
    do = tl.load(DO + off_o).to(tl.float32)#, mask=off_m[:, None] < N_CTX_Q, other=0.0).to(tl.float32)
    denom = tl.load(L + off_l).to(tl.float32)#, mask=off_m < N_CTX_Q, other=1.0).to(tl.float32)
    # denom = tl.where(denom == 0, 1.0, denom)  # ADDITIONAL CHECK IF YOU GET NaNs
    # compute
    do = do / denom[:, None]
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(NewDO + off_o, do)#, mask=off_m[:, None] < N_CTX_Q)
    tl.store(Delta + off_l, delta)#, mask=off_m < N_CTX_Q)


@triton.jit
def _bwd_kernel(
    Q, K, V, sm_scale, Out, DO,
    DQ, DK, DV,
    L, M,
    D,
    sqz, sqh, sqm, sqd,
    skz, skh, skn, skd,
    svz, svh, svn, svd,
    Z, H, N_CTX_Q, N_CTX_KV,
    BLOCK: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    N_PREFIX_Q: tl.constexpr,
):
    off_hz = tl.program_id(0)
    off_z = off_hz // H
    off_h = off_hz % H

    BLOCK_M: tl.constexpr = BLOCK
    BLOCK_N: tl.constexpr = BLOCK

    # offset pointers for batch/head
    Q += off_z * sqz + off_h * sqh
    K += off_z * skz + off_h * skh
    V += off_z * svz + off_h * svh
    DO += off_z * sqz + off_h * sqh
    DQ += off_z * sqz + off_h * sqh
    DK += off_z * skz + off_h * skh
    DV += off_z * svz + off_h * svh

    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # pointer to row-wise quantities in value-like data
    D_ptrs = D + off_hz * N_CTX_Q # pointer to D.view(Z*H,N_CTX_Q)[off_hz]
    m_ptrs = M + off_hz * N_CTX_Q # pointer to m.view(Z*H,N_CTX_Q)[off_hz]

    for start_n in range(0, N_CTX_KV, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + tl.arange(0, BLOCK_N)
        # pointers for keys and values
        k_ptrs = K + (offs_n[:, None] * skn + offs_d[None, :] * skd)
        v_ptrs = V + (offs_n[:, None] * svn + offs_d[None, :] * svd)

        # initialize dv amd dk
        dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
        dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)

        k = tl.load(k_ptrs)#, mask=offs_n[:, None] < N_CTX_KV)
        v = tl.load(v_ptrs)#, mask=offs_n[:, None] < N_CTX_KV)

        if start_n < N_PREFIX_Q * BLOCK_M:
            start_q_index = 0
        elif N_CTX_Q <= start_n - N_PREFIX_Q * BLOCK_M:
            start_q_index = start_n - N_PREFIX_Q * BLOCK_M
        else:
            first_start_m = start_n - N_PREFIX_Q * BLOCK_M
            first_start_m = tl.multiple_of(first_start_m, BLOCK_M)
            offs_m = (first_start_m + tl.arange(0, BLOCK_M))
            offs_m_real = offs_m + N_PREFIX_Q * BLOCK_M # indices of queries we want to process
            offs_m_real += tl.where(tl.arange(0, BLOCK_M) == BLOCK_M - 1, -1, 0)    

            q_ptrs = Q + (offs_m[:, None] * sqm + offs_d[None, :] * sqd)
            do_ptrs = DO + (offs_m[:, None] * sqm + offs_d[None, :] * sqd)
            dq_ptrs = DQ + (offs_m[:, None] * sqm + offs_d[None, :] * sqd)
            
            q = tl.load(q_ptrs) #, mask=offs_m[:,None] < N_CTX_Q)
            qk = tl.dot(q, tl.trans(k), allow_tf32=False)
            qk = tl.where(offs_m_real[:,None] >= (offs_n[None,:]), qk, float("-inf"))

            m = tl.load(m_ptrs + offs_m) #, mask=offs_m < N_CTX_Q)
            m_ = m # tl.where(m != float('-inf'), m, 0.0)

            last_p = tl.exp(qk * sm_scale - m_[:, None])

            do = tl.load(do_ptrs) #, mask=offs_m[:,None] < N_CTX_Q)
            # compute dv
            dv += tl.dot(tl.trans(last_p.to(Q.dtype.element_ty)), do, allow_tf32=False)


            Di = tl.load(D_ptrs + offs_m) #, mask=offs_m < N_CTX_Q)
            # compute dp = dot(v, do)
            last_dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - Di[:, None]
            last_dp += tl.dot(do, tl.trans(v), allow_tf32=False)
            # compute ds = p * (dp - delta[:, None])
            ds = last_p * last_dp * sm_scale
            
            # compute dk = dot(ds.T, q)
            dk += tl.dot(tl.trans(ds.to(Q.dtype.element_ty)), q, allow_tf32=False)

            dq = tl.load(dq_ptrs) #, mask=offs_m[:,None] < N_CTX_Q)
            # compute dq
            dq += tl.dot(ds.to(Q.dtype.element_ty), k, allow_tf32=False)
            tl.store(dq_ptrs, dq) #, mask=offs_m[:, None] < N_CTX_Q)
            start_q_index = first_start_m + BLOCK_M

        for start_m in range(start_q_index, N_CTX_Q, BLOCK_M):
            start_m = tl.multiple_of(start_m, BLOCK_M)
            offs_m = (start_m + tl.arange(0, BLOCK_M))
            # offs_m_real = offs_m + N_PREFIX_Q * BLOCK_M # indices of queries we want to process
            # offs_m_real += tl.where(tl.arange(0, BLOCK_M) == BLOCK_M - 1, -1, 0)    

            q_ptrs = Q + (offs_m[:, None] * sqm + offs_d[None, :] * sqd)
            do_ptrs = DO + (offs_m[:, None] * sqm + offs_d[None, :] * sqd)
            dq_ptrs = DQ + (offs_m[:, None] * sqm + offs_d[None, :] * sqd)
            
            q = tl.load(q_ptrs) #, mask=offs_m[:,None] < N_CTX_Q)
            qk = tl.dot(q, tl.trans(k), allow_tf32=False)
            qk *= sm_scale
            # qk = tl.where(offs_m_real[:,None] >= (offs_n[None,:]), qk, float("-inf"))  # This should not be necessary anymore since we separate the first step

            landmark_qk = tl.max(tl.where(tl.arange(0, BLOCK_N)[None, :] == BLOCK_N - 1, qk, float("-inf")), 1)
            normal_qk = tl.where(tl.arange(0, BLOCK_N)[None, :] == BLOCK_N - 1, float("-inf"), qk)

            m = tl.load(m_ptrs + offs_m)#, mask=offs_m < N_CTX_Q)
            m_ = m # tl.where(m != float('-inf'), m, 0.0)  # ADDITIONAL CHECK IF YOU GET NaNs

            p = tl.exp(landmark_qk - m_) # BLOCK_M

            do = tl.load(do_ptrs)#, mask=offs_m[:,None] < N_CTX_Q) # BLOCK_M x H

            normal_m = tl.max(normal_qk, 1)
            normal_p = tl.exp(normal_qk - normal_m[:, None])
            normal_p_normalized = normal_p / tl.sum(normal_p, 1)[:, None] # BLOCK_M x (BLOCK_N - 1)
            normal_kv = tl.dot(normal_p_normalized.to(Q.dtype.element_ty), v, allow_tf32=False) # BLOCK_M x H

            normal_D = tl.sum(do * normal_kv, 1)

            # compute dv
            dv += tl.dot(tl.trans((p[:, None] * normal_p_normalized).to(Q.dtype.element_ty)), do, allow_tf32=False)

            Di = tl.load(D_ptrs + offs_m)#, mask=offs_m < N_CTX_Q)
            # compute dp and ds for landmark
            dp = tl.zeros([BLOCK_M], dtype=tl.float32) - Di
            dp += normal_D 
            landmark_ds = p * dp
            # compute dp and ds for others
            normal_dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - normal_D[:, None]
            normal_dp += tl.dot(do, tl.trans(v), allow_tf32=False)
            normal_ds = p[:, None] * normal_p_normalized * normal_dp 
            # merge
            ds = tl.where(tl.arange(0, BLOCK_N)[None, :] == BLOCK_N - 1, landmark_ds[:, None], normal_ds)
            ds *= sm_scale
            # compute dk = dot(ds.T, q)
            dk += tl.dot(tl.trans(ds.to(Q.dtype.element_ty)), q, allow_tf32=False)

            dq = tl.load(dq_ptrs) #, mask=offs_m[:,None] < N_CTX_Q)
            # compute dq
            dq += tl.dot(ds.to(Q.dtype.element_ty), k, allow_tf32=False)
            tl.store(dq_ptrs, dq) #, mask=offs_m[:, None] < N_CTX_Q)
         
        # write-back
        dv_ptrs = DV + (offs_n[:, None] * svn + offs_d[None, :] * svd)
        dk_ptrs = DK + (offs_n[:, None] * skn + offs_d[None, :] * skd)
        tl.store(dv_ptrs, dv) #, mask=offs_n[:, None] < N_CTX_KV)
        tl.store(dk_ptrs, dk) #, mask=offs_n[:, None] < N_CTX_KV)


class FusedLandmarkAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, n_prefix_q, sm_scale, block_size):
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        # shape constraints
        batch, nheads, seqlen_q, d = q.shape
        _, _, seqlen_k, _ = k.shape
        assert k.shape == (batch, nheads, seqlen_k, d)
        assert v.shape == (batch, nheads, seqlen_k, d)
        assert d <= 128, 'FlashAttention only support head dimensions up to 128'
        assert q.dtype == k.dtype == v.dtype, 'All tensors must have the same type'
        #assert q.dtype in [torch.float16, torch.bfloat16], 'Only support fp16 and bf16'
        assert q.is_cuda and k.is_cuda and v.is_cuda
        
        BLOCK = block_size
        o = torch.empty_like(q)
        grid = (triton.cdiv(q.shape[2], BLOCK), q.shape[0] * q.shape[1], 1)
        L = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        m = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        num_warps = 4 if d <= 64 else 8

        _fwd_kernel[grid](
            q, k, v, sm_scale,
            o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            L, m,
            q.shape[0], q.shape[1], q.shape[2], k.shape[2],
            BLOCK=BLOCK, BLOCK_DMODEL=d,
            N_PREFIX_Q=n_prefix_q,
            num_warps=num_warps, num_stages=2
        )

        ctx.save_for_backward(q, k, v, o, L, m)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = d
        ctx.N_PREFIX_Q = n_prefix_q
        ctx.BLOCK = BLOCK
        return o

    @staticmethod
    def backward(ctx, do):
        BLOCK = ctx.BLOCK
        q, k, v, o, l, m = ctx.saved_tensors
        assert q.shape[2] % BLOCK == 0, "Backward supported only for full blocks"
        assert k.shape[2] % BLOCK == 0, "Backward supported only for full blocks"

        do = do.contiguous()
        dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        do_scaled = torch.empty_like(do)
        delta = torch.empty_like(l)
        _bwd_preprocess[(ctx.grid[0], ctx.grid[1])](
            o, o.stride(0), o.stride(1), o.stride(2), o.stride(3), do, l, l.stride(0), l.stride(1),
            do_scaled, delta, q.shape[2],
            BLOCK_M=BLOCK, D_HEAD=ctx.BLOCK_DMODEL,
        )
        _bwd_kernel[(ctx.grid[1],)](
            q, k, v, ctx.sm_scale,
            o, do_scaled,
            dq, dk, dv,
            l, m,
            delta,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            q.shape[0], q.shape[1], q.shape[2], k.shape[2],
            BLOCK=BLOCK,
            BLOCK_DMODEL=ctx.BLOCK_DMODEL, 
            N_PREFIX_Q=ctx.N_PREFIX_Q,
            num_warps=8,
            num_stages=1,
        )
        return dq, dk, dv, None, None, None

def fused_landmark_attention(q, k, v, is_mem, sm_scale=None, block_size=64):

    expected_is_mem = torch.arange(0, is_mem.shape[-1], device=is_mem.device) % block_size == (block_size - 1)
    assert (is_mem == expected_is_mem).all()

    n_history_kv = k.shape[-2] - q.shape[-2]
    assert n_history_kv % block_size == 0
    n_history_blocks = n_history_kv // block_size

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.size(-1))

    return FusedLandmarkAttention.apply(q, k, v, n_history_blocks, sm_scale, block_size)
