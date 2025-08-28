#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "../../utils/types.hpp"

namespace llaisys::ops {

void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    CHECK_SAME_DEVICE(max_idx, max_val, vals);
    CHECK_ARGUMENT(vals->ndim() == 1, "vals must be 1D tensor");
    CHECK_ARGUMENT(max_idx->numel() == 1, "max_idx must be scalar tensor");
    CHECK_ARGUMENT(max_val->numel() == 1, "max_val must be scalar tensor");
    CHECK_ARGUMENT(vals->isContiguous(), "vals must be contiguous");
    
    llaisysDataType_t dtype = vals->dtype();
    
    if (vals->deviceType() == LLAISYS_DEVICE_CPU) {
        void* vals_data = vals->data();
        void* max_idx_data = max_idx->data();
        void* max_val_data = max_val->data();
        
        int64_t numel = vals->numel();
        int64_t max_index = 0;
        
        switch (dtype) {
            case LLAISYS_DTYPE_F32: {
                float* vals_ptr = static_cast<float*>(vals_data);
                float max_value = vals_ptr[0];
                
                for (int64_t i = 1; i < numel; i++) {
                    if (vals_ptr[i] > max_value) {
                        max_value = vals_ptr[i];
                        max_index = i;
                    }
                }
                
                *static_cast<int64_t*>(max_idx_data) = max_index;
                *static_cast<float*>(max_val_data) = max_value;
                break;
            }
            
            case LLAISYS_DTYPE_F16: {
                using namespace llaisys::utils;
                fp16_t* vals_ptr = static_cast<fp16_t*>(vals_data);
                float max_value = _f16_to_f32(vals_ptr[0]);
                
                for (int64_t i = 1; i < numel; i++) {
                    float current_val = _f16_to_f32(vals_ptr[i]);
                    if (current_val > max_value) {
                        max_value = current_val;
                        max_index = i;
                    }
                }
                
                *static_cast<int64_t*>(max_idx_data) = max_index;
                *static_cast<fp16_t*>(max_val_data) = _f32_to_f16(max_value);
                break;
            }
            
            case LLAISYS_DTYPE_BF16: {
                using namespace llaisys::utils;
                bf16_t* vals_ptr = static_cast<bf16_t*>(vals_data);
                float max_value = _bf16_to_f32(vals_ptr[0]);
                
                for (int64_t i = 1; i < numel; i++) {
                    float current_val = _bf16_to_f32(vals_ptr[i]);
                    if (current_val > max_value) {
                        max_value = current_val;
                        max_index = i;
                    }
                }
                
                *static_cast<int64_t*>(max_idx_data) = max_index;
                *static_cast<bf16_t*>(max_val_data) = _f32_to_bf16(max_value);
                break;
            }
            
            default:
                EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
        }
        
        return;
    }

    llaisys::core::context().setDevice(vals->deviceType(), vals->deviceId());

    switch (vals->deviceType()) {
        case LLAISYS_DEVICE_CPU:
            break;
#ifdef ENABLE_NVIDIA_API
        case LLAISYS_DEVICE_NVIDIA:
            TO_BE_IMPLEMENTED();
            break;
#endif
        default:
            EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

} // namespace llaisys::ops
