#ifdef USE_BANG

#include "cnnl_functions.h"

namespace refactor::kernel::cnnl {

    cnnlDataType_t cnnlDataTypeConvert(DataType dataType) {
        // clang-format off
        switch (dataType) {
            case DataType::F32 : return CNNL_DTYPE_FLOAT;    break;
            case DataType::F64 : return CNNL_DTYPE_DOUBLE;   break;
            case DataType::FP16: return CNNL_DTYPE_HALF;     break;
            case DataType::I8  : return CNNL_DTYPE_INT8;     break;
            case DataType::I32 : return CNNL_DTYPE_INT32;    break;
            case DataType::U8  : return CNNL_DTYPE_UINT8;    break;
            case DataType::BF16: return CNNL_DTYPE_BFLOAT16; break;
            case DataType::I64 : return CNNL_DTYPE_INT64;    break;
            case DataType::Bool: return CNNL_DTYPE_BOOL;     break;
            default: UNREACHABLE();
        }
        // clang-format on
    }

    void setCnnlTensor(cnnlTensorDescriptor_t t, DataType dt, slice_t<int> d) {
        auto dt_ = cnnlDataTypeConvert(dt);
        if (auto n = d.size(); n == 4) {
            CNNL_ASSERT(cnnlSetTensorDescriptor(t, CNNL_LAYOUT_NCHW, dt_, d.size(), d.begin()));
        } else if (n < 4) {
            int d_[]{1, 1, 1, 1};
            std::copy_n(d.begin(), n, d_ + 4 - n);
            CNNL_ASSERT(cnnlSetTensorDescriptor(t, CNNL_LAYOUT_NCHW, dt_, 4, std::move(d_)));
        } else {
            CNNL_ASSERT(cnnlSetTensorDescriptor(t, CNNL_LAYOUT_NCHW, dt_, d.size(), d.begin()));
        }
    }
}// namespace refactor::kernel::cnnl

#endif
