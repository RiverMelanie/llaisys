#pragma once

#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::cpu {
void swiglu(void* out, const void* gate, const void* up, 
           llaisysDataType_t dtype, size_t numel);
}