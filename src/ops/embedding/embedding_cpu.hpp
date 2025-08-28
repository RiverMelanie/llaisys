#pragma once

#include "../../../tensor/tensor.hpp"
#include "../../../utils/types.hpp"

namespace llaisys::ops::cpu {

template <typename T>
void embedding_impl(void* out_data, const void* index_data, const void* weight_data, 
                   int64_t index_size, int64_t embedding_dim) {
    const int64_t* indices = static_cast<const int64_t*>(index_data);
    const T* weight = static_cast<const T*>(weight_data);
    T* out = static_cast<T*>(out_data);
    
    for (int64_t i = 0; i < index_size; ++i) {
        int64_t idx = indices[i];
        const T* src = weight + idx * embedding_dim;
        T* dest = out + i * embedding_dim;
        
        for (int64_t j = 0; j < embedding_dim; ++j) {
            dest[j] = src[j];
        }
    }
}

void embedding(void* out_data, const void* index_data, const void* weight_data,
              llaisysDataType_t dtype, int64_t index_size, int64_t embedding_dim) {
    switch (dtype) {
        case LLAISYS_DTYPE_F32:
            embedding_impl<float>(out_data, index_data, weight_data, index_size, embedding_dim);
            break;
        case LLAISYS_DTYPE_F16:
            embedding_impl<llaisys::fp16_t>(out_data, index_data, weight_data, index_size, embedding_dim);
            break;
        case LLAISYS_DTYPE_BF16:
            embedding_impl<llaisys::bf16_t>(out_data, index_data, weight_data, index_size, embedding_dim);
            break;
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu