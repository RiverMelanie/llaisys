// src/ops/rms_norm/cpu/rms_norm_cpu.cpp
#include "rms_norm_cpu.hpp"
#include "../../../utils/types.hpp"
#include "../../../utils/utils.hpp"

#include <cmath>
#include <vector>

namespace llaisys::ops::cpu {

template <typename T>
void rms_norm_impl(T* out, const T* in, const T* weight, 
                  int64_t rows, int64_t cols, float eps) {
    for (int64_t i = 0; i < rows; ++i) {
        const T* row_in = in + i * cols;
        T* row_out = out + i * cols;
        
        double sum_sq = 0.0;
        for (int64_t j = 0; j < cols; ++j) {
            float val = llaisys::utils::cast<float, T>(row_in[j]);
            sum_sq += val * val;
        }
        
        float mean_sq = sum_sq / cols;
        float rms = std::sqrt(mean_sq + eps);
        float scale = 1.0f / rms;
        
        for (int64_t j = 0; j < cols; ++j) {
            float normalized = llaisys::utils::cast<float, T>(row_in[j]) * scale;
            float weighted = normalized * llaisys::utils::cast<float, T>(weight[j]);
            row_out[j] = llaisys::utils::cast<T, float>(weighted);
        }
    }
}

void rms_norm(void* out, void* in, void* weight, llaisysDataType_t dtype, 
              const std::vector<int64_t>& shape, float eps) {
    int64_t rows = 1;
    for (size_t i = 0; i < shape.size() - 1; ++i) {
        rows *= shape[i];
    }
    int64_t cols = shape.back();
    
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        rms_norm_impl(static_cast<float*>(out), static_cast<const float*>(in), 
                     static_cast<const float*>(weight), rows, cols, eps);
        break;
    case LLAISYS_DTYPE_F16:
        rms_norm_impl(static_cast<llaisys::fp16_t*>(out), 
                     static_cast<const llaisys::fp16_t*>(in), 
                     static_cast<const llaisys::fp16_t*>(weight), 
                     rows, cols, eps);
        break;
    case LLAISYS_DTYPE_BF16:
        rms_norm_impl(static_cast<llaisys::bf16_t*>(out), 
                     static_cast<const llaisys::bf16_t*>(in), 
                     static_cast<const llaisys::bf16_t*>(weight), 
                     rows, cols, eps);
        break;
    default:
        throw std::runtime_error("RMSNorm: unsupported data type");
    }
}

} // namespace llaisys::ops::cpu