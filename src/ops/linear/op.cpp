// src/ops/linear/op.cpp
#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "linear_cpu.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    if (bias) {
        CHECK_SAME_DEVICE(out, bias);
    }
    
    // Check shapes: out = [batch_size, out_features]
    // in = [batch_size, in_features]
    // weight = [out_features, in_features]
    // bias = [out_features] (optional)
    size_t batch_size = out->shape()[0];
    size_t out_features = out->shape()[1];
    size_t in_features = in->shape()[1];
    
    CHECK_ARGUMENT(in->shape()[0] == batch_size, "Input batch size must match output batch size");
    CHECK_ARGUMENT(weight->shape()[0] == out_features, "Weight output features must match output features");
    CHECK_ARGUMENT(weight->shape()[1] == in_features, "Weight input features must match input features");
    
    if (bias) {
        CHECK_ARGUMENT(bias->shape()[0] == out_features, "Bias size must match output features");
        CHECK_SAME_DTYPE(out->dtype(), bias->dtype());
    }
    
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), 
           "Linear: all tensors must be contiguous.");
    if (bias) {
        ASSERT(bias->isContiguous(), "Linear: bias must be contiguous");
    }

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(out->data(), in->data(), weight->data(), 
                          bias ? bias->data() : nullptr,
                          out->dtype(), batch_size, in_features, out_features);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(), in->data(), weight->data(), 
                          bias ? bias->data() : nullptr,
                          out->dtype(), batch_size, in_features, out_features);
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
