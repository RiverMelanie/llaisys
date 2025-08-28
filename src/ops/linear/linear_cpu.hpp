// src/ops/linear/cpu/linear_cpu.hpp
#pragma once

#include "../../../tensor/tensor.hpp"
#include "../../../utils/types.hpp"

namespace llaisys::ops::cpu {

template <typename T>
void linear_impl(void *out, const void *in, const void *weight, const void *bias, 
                 size_t batch_size, size_t in_features, size_t out_features) {
    T *out_ptr = static_cast<T *>(out);
    const T *in_ptr = static_cast<const T *>(in);
    const T *weight_ptr = static_cast<const T *>(weight);
    const T *bias_ptr = static_cast<const T *>(bias);

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t o = 0; o < out_features; ++o) {
            T sum = T(0);
            
            // X * W^T: in_ptr[b * in_features + i] * weight_ptr[o * in_features + i]
            for (size_t i = 0; i < in_features; ++i) {
                sum += in_ptr[b * in_features + i] * weight_ptr[o * in_features + i];
            }
            
            if (bias_ptr) {
                sum += bias_ptr[o];
            }
            
            out_ptr[b * out_features + o] = sum;
        }
    }
}

void linear(void *out, const void *in, const void *weight, const void *bias,
            llaisysDataType_t dtype, size_t batch_size, size_t in_features, size_t out_features) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        linear_impl<float>(out, in, weight, bias, batch_size, in_features, out_features);
        break;
    case LLAISYS_DTYPE_F16:
        linear_impl<llaisys::fp16_t>(out, in, weight, bias, batch_size, in_features, out_features);
        break;
    case LLAISYS_DTYPE_BF16:
        linear_impl<llaisys::bf16_t>(out, in, weight, bias, batch_size, in_features, out_features);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu