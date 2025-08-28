#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "rope_cpu.hpp"

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    
    CHECK_SAME_DEVICE(out, in, pos_ids);
    

    
    ASSERT(out->dim() == 3, "Output tensor must be 3D");
    ASSERT(in->dim() == 3, "Input tensor must be 3D");
    ASSERT(pos_ids->dim() == 1, "Position IDs must be 1D");
    
    const auto& out_shape = out->shape();
    const auto& in_shape = in->shape();
    
    ASSERT(out_shape == in_shape, "Output and input shapes must match");
    ASSERT(out_shape[0] == pos_ids->shape()[0], "Sequence length must match position IDs length");
    ASSERT(out_shape[2] % 2 == 0, "Head dimension must be even for RoPE");
    
    ASSERT(out->dtype() == in->dtype(), "Output and input dtypes must match");
    ASSERT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "Position IDs must be int64");
    
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(),
           "RoPE: all tensors must be contiguous");

    const int64_t seq_len = out_shape[0];
    const int64_t n_heads = out_shape[1];
    const int64_t head_dim = out_shape[2];

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(out->data(), in->data(), 
                        static_cast<const int64_t*>(pos_ids->data()),
                        theta, out->dtype(), seq_len, n_heads, head_dim);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
        case LLAISYS_DEVICE_CPU:
            return cpu::rope(out->data(), in->data(), 
                            static_cast<const int64_t*>(pos_ids->data()),
                            theta, out->dtype(), seq_len, n_heads, head_dim);
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
