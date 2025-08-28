#include "rope_cpu.hpp"
#include "../../../utils/types.hpp"
#include <cmath>
#include <cstdint>

namespace llaisys::ops::cpu {

template <typename T>
void rope_impl(void* out_ptr, const void* in_ptr, const int64_t* pos_ids, float theta,
               int64_t seq_len, int64_t n_heads, int64_t head_dim) {
    T* out = static_cast<T*>(out_ptr);
    const T* in = static_cast<const T*>(in_ptr);
    
    const int64_t half_dim = head_dim / 2;
    const int64_t elements_per_head = head_dim;
    
    for (int64_t seq_idx = 0; seq_idx < seq_len; ++seq_idx) {
        const int64_t pos = pos_ids[seq_idx];
        
        for (int64_t head_idx = 0; head_idx < n_heads; ++head_idx) {
            const T* head_in = in + (seq_idx * n_heads + head_idx) * head_dim;
            T* head_out = out + (seq_idx * n_heads + head_idx) * head_dim;
            
            for (int64_t j = 0; j < half_dim; ++j) {
                // Calculate the rotation angle
                float angle = pos / std::pow(theta, 2.0f * j / head_dim);
                float cos_val = std::cos(angle);
                float sin_val = std::sin(angle);
                
                // Convert input values to float for calculation
                float a_j = llaisys::utils::cast<float>(head_in[j]);
                float b_j = llaisys::utils::cast<float>(head_in[j + half_dim]);
                
                // Apply rotation
                float a_j_prime = a_j * cos_val - b_j * sin_val;
                float b_j_prime = b_j * cos_val + a_j * sin_val;
                
                // Convert back to target type and store
                head_out[j] = llaisys::utils::cast<T>(a_j_prime);
                head_out[j + half_dim] = llaisys::utils::cast<T>(b_j_prime);
            }
        }
    }
}

void rope(void* out, const void* in, const int64_t* pos_ids, float theta,
          llaisysDataType_t dtype, int64_t seq_len, int64_t n_heads, int64_t head_dim) {
    switch (dtype) {
        case LLAISYS_DTYPE_F32:
            rope_impl<float>(out, in, pos_ids, theta, seq_len, n_heads, head_dim);
            break;
        case LLAISYS_DTYPE_F16:
            rope_impl<llaisys::fp16_t>(out, in, pos_ids, theta, seq_len, n_heads, head_dim);
            break;
        case LLAISYS_DTYPE_BF16:
            rope_impl<llaisys::bf16_t>(out, in, pos_ids, theta, seq_len, n_heads, head_dim);
            break;
        default:
            throw std::runtime_error("Unsupported dtype for RoPE");
    }
}

} // namespace llaisys::ops::cpu