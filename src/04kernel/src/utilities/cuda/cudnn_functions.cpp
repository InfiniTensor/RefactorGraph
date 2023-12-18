#ifdef USE_CUDA

#include "cudnn_functions.h"

namespace refactor::kernel::cudnn {

    cudnnDataType_t cudnnDataTypeConvert(DataType dataType) {
        // clang-format off
        switch (dataType) {
            case DataType::F32 : return CUDNN_DATA_FLOAT;    break;
            case DataType::F64 : return CUDNN_DATA_DOUBLE;   break;
            case DataType::FP16: return CUDNN_DATA_HALF;     break;
            case DataType::I8  : return CUDNN_DATA_INT8;     break;
            case DataType::I32 : return CUDNN_DATA_INT32;    break;
            case DataType::U8  : return CUDNN_DATA_UINT8;    break;
            case DataType::BF16: return CUDNN_DATA_BFLOAT16; break;
            case DataType::I64 : return CUDNN_DATA_INT64;    break;
            case DataType::Bool: return CUDNN_DATA_BOOLEAN;  break;
            default: UNREACHABLE();
        }
        // clang-format on
    }

    void setCudnnTensor(cudnnTensorDescriptor_t t, DataType dt, slice_t<int> d) {
        auto dt_ = cudnnDataTypeConvert(dt);
        if (auto n = d.size(); n == 4) {
            CUDNN_ASSERT(cudnnSetTensor4dDescriptor(t, CUDNN_TENSOR_NCHW, dt_, d[0], d[1], d[2], d[3]));
        } else if (n < 4) {
            int d_[]{1, 1, 1, 1};
            std::copy_n(d.begin(), n, d_ + 4 - n);
            CUDNN_ASSERT(cudnnSetTensor4dDescriptor(t, CUDNN_TENSOR_NCHW, dt_, d_[0], d_[1], d_[2], d_[3]));
        } else {
            CUDNN_ASSERT(cudnnSetTensorNdDescriptorEx(t, CUDNN_TENSOR_NCHW, dt_, d.size(), d.begin()));
        }
    }
}// namespace refactor::kernel::cudnn

#endif
