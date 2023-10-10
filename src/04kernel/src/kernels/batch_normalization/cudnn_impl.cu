#include "../cudnn_context.hh"
#include "../cudnn_error_handler.h"
#include "cudnn_impl.h"
#include "runtime/resource.h"
#include <cudnn.h>
#include <utility>

namespace refactor::kernel::cudnn {
    using namespace runtime;
    using Ctx = CudnnContext;
    using DT = common::DataType;

    Operation Info::lower() const {
        int strideAx[4]{0, 0, 0, 1},       // to calculate
            dimAp[4]{1, dimAx[1], 1, 1},   // 1xCx1x1
            strideAp[4]{dimAx[1], 1, 1, 1};// Cx1x1x1
        // TODO: calculate real stride based on layout type
        for (auto i = 3; i > 0; --i) {
            strideAx[i - 1] = strideAx[i] * dimAx[i];
        }

        // RAII for closure
        struct Descriptors {
            cudnnTensorDescriptor_t x, param;

            Descriptors() : x(nullptr), param(nullptr) {
                CUDNN_ASSERT(cudnnCreateTensorDescriptor(&x),
                             "cudnnCreateTensorDescriptor failed");
                CUDNN_ASSERT(cudnnCreateTensorDescriptor(&param),
                             "cudnnCreateTensorDescriptor failed");
            }
            ~Descriptors() {
                CUDNN_ASSERT(cudnnDestroyTensorDescriptor(x),
                             "cudnnDestroyTensorDescriptor failed");
                CUDNN_ASSERT(cudnnDestroyTensorDescriptor(param),
                             "cudnnDestroyTensorDescriptor failed");
            }

            Descriptors(const Descriptors &) = delete;
            Descriptors(Descriptors &&) = delete;
        };
        auto d = std::make_shared<Descriptors>();
        // clang-format off
        cudnnDataType_t
            dtX_ = dtX == DT::F32  ? CUDNN_DATA_FLOAT
                 : dtX == DT::F64  ? CUDNN_DATA_DOUBLE
                 : dtX == DT::FP16 ? CUDNN_DATA_HALF
                 : dtX == DT::BF16 ? CUDNN_DATA_BFLOAT16
                 : UNREACHABLEX(cudnnDataType_t, ""),
            dtParam_ = dtParam == DT::F32 ? CUDNN_DATA_FLOAT
                     : dtParam == DT::F64 ? CUDNN_DATA_DOUBLE
                     : UNREACHABLEX(cudnnDataType_t, "");
        // clang-format on

        CUDNN_ASSERT(cudnnSetTensorNdDescriptor(d->x, dtX_, 4, dimAx, strideAx),
                     "cudnnSetTensorNdDescriptor failed");
        CUDNN_ASSERT(cudnnSetTensorNdDescriptor(d->param, dtParam_, 4, dimAp, strideAp),
                     "cudnnSetTensorNdDescriptor failed");

        // nvcc at c++11 doesn't support real move capture
        return [d = std::move(d),
                param64 = dtParam == DT::F64,
                epsilon = this->epsilon](Resources &res, Addresses inputs, Addresses outputs) {
            // fetch cudnn handle from resources
            auto handle = std::any_cast<cudnnHandle_t>(res.fetchOrStore<CudnnContext>()->handle);
            // name inputs and outputs
            auto x = inputs[0],
                 scale = inputs[1],
                 bias = inputs[2],
                 mean = inputs[3],
                 var = inputs[4];
            auto y = outputs[0];
            // build alpha/beta for double
            union {
                float f32[2];
                double f64[2];
            };
            void *alpha, *beta;
            if (param64) {
                f32[0] = 1;
                f32[1] = 0;
                alpha = f32;
                beta = f32 + 1;
            } else {
                f64[0] = 1;
                f64[1] = 0;
                alpha = f64;
                beta = f64 + 1;
            }
            CUDNN_ASSERT(cudnnBatchNormalizationForwardInference(
                             handle, CUDNN_BATCHNORM_SPATIAL, alpha, beta,
                             d->x, x,
                             d->x, y, // desc(x) === desc(y) for onnx
                             d->param,// scale, bias, mean, var
                             scale, bias, mean, var,
                             epsilon),
                         "cudnnBatchNormalizationForwardInference failed");
        };
    }

}// namespace refactor::kernel::cudnn
