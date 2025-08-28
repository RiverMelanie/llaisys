#pragma once

#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::cpu {
void rope(void* out, const void* in, const int64_t* pos_ids, float theta, 
          llaisysDataType_t dtype, int64_t seq_len, int64_t n_heads, int64_t head_dim);
} // namespace llaisys::ops::cpu