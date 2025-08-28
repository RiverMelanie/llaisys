#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils/check.hpp"

#include "cpu/embedding_cpu.hpp"

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, weight);
    CHECK_SAME_DEVICE(out, index);
    
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());
    CHECK_ARGUMENT(index->dtype() == LLAISYS_DTYPE_I64, 
                  "Index tensor must be Int64 type");
    
    CHECK_ARGUMENT(out->ndim() == 2, "Output must be 2D tensor");
    CHECK_ARGUMENT(index->ndim() == 1, "Index must be 1D tensor");
    CHECK_ARGUMENT(weight->ndim() == 2, "Weight must be 2D tensor");
    
    int64_t index_size = index->shape()[0];
    int64_t embedding_dim = weight->shape()[1];
    
    CHECK_ARGUMENT(out->shape()[0] == index_size, 
                  "Output first dimension must match index size");
    CHECK_ARGUMENT(out->shape()[1] == embedding_dim,
                  "Output second dimension must match weight embedding dimension");
    
    CHECK_ARGUMENT(index_size > 0 && embedding_dim > 0,
                  "Dimensions must be positive");
    
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(),
           "Embedding: all tensors must be contiguous");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(out->data(), index->data(), weight->data(),
                            out->dtype(), index_size, embedding_dim);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
        case LLAISYS_DEVICE_CPU:
            return cpu::embedding(out->data(), index->data(), weight->data(),
                                out->dtype(), index_size, embedding_dim);
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
