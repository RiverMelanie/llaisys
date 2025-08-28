// src/ops/rms_norm/op.cpp
#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "rms_norm_cpu.hpp"

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    ASSERT(out->shape() == in->shape(), "RMSNorm: out and in must have same shape");
    ASSERT(weight->ndim() == 1, "RMSNorm: weight must be 1D tensor");
    ASSERT(weight->shape()[0] == in->shape()[in->ndim() - 1], 
           "RMSNorm: weight size must match last dimension of input");
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), 
           "RMSNorm: all tensors must be contiguous");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rms_norm(out->data(), in->data(), weight->data(), 
                           out->dtype(), in->shape(), eps);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm(out->data(), in->data(), weight->data(), 
                           out->dtype(), in->shape(), eps);
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
