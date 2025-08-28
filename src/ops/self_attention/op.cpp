#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "self_attention_cpu.hpp"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    
    ASSERT(q->ndim() == 3, "Query tensor must be 3D [seqlen, nhead, d]");
    ASSERT(k->ndim() == 3, "Key tensor must be 3D [total_len, nkvhead, d]");
    ASSERT(v->ndim() == 3, "Value tensor must be 3D [total_len, nkvhead, dv]");
    ASSERT(attn_val->ndim() == 3, "Output tensor must be 3D [seqlen, nhead, dv]");
    
    int q_seqlen = q->shape()[0];
    int q_nhead = q->shape()[1];
    int q_d = q->shape()[2];
    
    int k_seqlen = k->shape()[0];
    int k_nhead = k->shape()[1];
    int k_d = k->shape()[2];
    
    int v_seqlen = v->shape()[0];
    int v_nhead = v->shape()[1];
    int v_dv = v->shape()[2];
    
    ASSERT(k_seqlen == v_seqlen, "Key and value sequence lengths must match");
    ASSERT(k_d == q_d, "Query and key feature dimensions must match");
    ASSERT(q_nhead % k_nhead == 0, "Number of query heads must be divisible by number of key/value heads");
    
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(),
           "SelfAttention: all tensors must be contiguous");

    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(attn_val, q, k, v, scale);
    }

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());

    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(attn_val, q, k, v, scale);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
