#include "swiglu_cpu.hpp"
#include "../../../utils/types.hpp"
#include <cmath>

namespace llaisys::ops::cpu {

template <typename T>
void swiglu_impl(T* out, const T* gate, const T* up, size_t numel) {
    for (size_t i = 0; i < numel; ++i) {
        float gate_val = llaisys::utils::cast<float>(gate[i]);
        float sigmoid = 1.0f / (1.0f + std::exp(-gate_val));
        float up_val = llaisys::utils::cast<float>(up[i]);
        out[i] = llaisys::utils::cast<T>(up_val * sigmoid);
    }
}

template <>
void swiglu_impl<float>(float* out, const float* gate, const float* up, size_t numel) {
    for (size_t i = 0; i < numel; ++i) {
        float sigmoid = 1.0f / (1.0f + std::exp(-gate[i]));
        out[i] = up[i] * sigmoid;
    }
}

void swiglu(void* out, const void* gate, const void* up, 
           llaisysDataType_t dtype, size_t numel) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        swiglu_impl(static_cast<float*>(out), 
                   static_cast<const float*>(gate), 
                   static_cast<const float*>(up), numel);
        break;
    case LLAISYS_DTYPE_F16:
        swiglu_impl(static_cast<llaisys::fp16_t*>(out), 
                   static_cast<const llaisys::fp16_t*>(gate), 
                   static_cast<const llaisys::fp16_t*>(up), numel);
        break;
    case LLAISYS_DTYPE_BF16:
        swiglu_impl(static_cast<llaisys::bf16_t*>(out), 
                   static_cast<const llaisys::bf16_t*>(gate), 
                   static_cast<const llaisys::bf16_t*>(up), numel);
        break;
    default:
        throw std::invalid_argument("Unsupported dtype for swiglu");
    }
}

} // namespace llaisys::ops::cpu