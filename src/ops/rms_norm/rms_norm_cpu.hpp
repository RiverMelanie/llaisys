// src/ops/rms_norm/cpu/rms_norm_cpu.hpp
#pragma once

#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::cpu {
void rms_norm(void* out, void* in, void* weight, llaisysDataType_t dtype, 
              const std::vector<int64_t>& shape, float eps);
} // namespace llaisys::ops::cpu