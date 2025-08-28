#include "self_attention_cpu.hpp"
#include "../../../utils/types.hpp"
#include <cmath>
#include <algorithm>
#include <vector>

namespace llaisys::ops::cpu {

template <typename T>
void self_attention_impl(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    int q_seqlen = q->shape()[0];
    int q_nhead = q->shape()[1];
    int q_d = q->shape()[2];
    
    int k_seqlen = k->shape()[0];
    int k_nhead = k->shape()[1];
    int k_d = k->shape()[2];
    
    int v_dv = v->shape()[2];
    int head_ratio = q_nhead / k_nhead;
    
    T* q_data = static_cast<T*>(q->data());
    T* k_data = static_cast<T*>(k->data());
    T* v_data = static_cast<T*>(v->data());
    T* out_data = static_cast<T*>(attn_val->data());
    
    for (int pos = 0; pos < q_seqlen; ++pos) {
        for (int h = 0; h < q_nhead; ++h) {
            int kv_head = h / head_ratio;
            
            std::vector<float> attention_scores(k_seqlen, 0.0f);
            
            for (int kv_pos = 0; kv_pos < k_seqlen; ++kv_pos) {
                float score = 0.0f;
                for (int d = 0; d < q_d; ++d) {
                    float q_val = llaisys::utils::cast<float>(q_data[pos * q_nhead * q_d + h * q_d + d]);
                    float k_val = llaisys::utils::cast<float>(k_data[kv_pos * k_nhead * k_d + kv_head * k_d + d]);
                    score += q_val * k_val;
                }
                attention_scores[kv_pos] = score * scale;
            }
            
            for (int kv_pos = pos + 1; kv_pos < k_seqlen; ++kv_pos) {
                attention_scores[kv_pos] = -std::numeric_limits<float>::infinity();
            }
            
            float max_score = -std::numeric_limits<float>::infinity();
            for (int kv_pos = 0; kv_pos < k_seqlen; ++kv_pos) {
                if (attention_scores[kv_pos] > max_score) {
                    max_score = attention_scores[kv_pos];
                }
            }
            
            float sum_exp = 0.0f;
            for (int kv_pos = 0; kv_pos < k_seqlen; ++kv_pos) {
                attention_scores[kv_pos] = std::exp(attention_scores[kv_pos] - max_score);
                sum_exp += attention_scores[kv_pos];
            }
            
            for (int kv_pos = 0; kv_pos < k_seqlen; ++kv_pos) {
                attention_scores[kv_pos] /= sum_exp;
            }
            
            for (int dv = 0; dv < v_dv; ++dv) {
                float weighted_sum = 0.0f;
                for (int kv_pos = 0; kv_pos < k_seqlen; ++kv_pos) {
                    float v_val = llaisys::utils::cast<float>(v_data[kv_pos * k_nhead * v_dv + kv_head * v_dv + dv]);
                    weighted_sum += attention_scores[kv_pos] * v_val;
                }
                out_data[pos * q_nhead * v_dv + h * v_dv + dv] = llaisys::utils::cast<T>(weighted_sum);
            }
        }
    }
}

void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    switch (q->dtype()) {
    case LLAISYS_DTYPE_F32:
        self_attention_impl<float>(attn_val, q, k, v, scale);
        break;
    case LLAISYS_DTYPE_F16:
        self_attention_impl<llaisys::fp16_t>(attn_val, q, k, v, scale);
        break;
    case LLAISYS_DTYPE_BF16:
        self_attention_impl<llaisys::bf16_t>(attn_val, q, k, v, scale);
        break;
    default:
        throw std::runtime_error("Unsupported data type for self_attention");
    }
}

} // namespace llaisys::ops::cpu