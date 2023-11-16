#include "../../utilities/cuda/cudnn_context.hh"
#include "../../utilities/cuda/cudnn_functions.h"
#include "cudnn_kernel.hh"
#include <cudnn.h>

namespace refactor::kernel {
    using namespace cudnn;
    using namespace runtime;
    using DT = DataType;

    auto BatchNormalizationCudnn::lower(Resources &res) const noexcept -> Routine {
        // RAII for closure
        struct Descriptors {
            cudnnTensorDescriptor_t x, param;
            bool f64;

            Descriptors() : x(nullptr), param(nullptr) {
                CUDNN_ASSERT(cudnnCreateTensorDescriptor(&x));
                CUDNN_ASSERT(cudnnCreateTensorDescriptor(&param));
            }
            ~Descriptors() {
                CUDNN_ASSERT(cudnnDestroyTensorDescriptor(x));
                CUDNN_ASSERT(cudnnDestroyTensorDescriptor(param));
            }

            Descriptors(const Descriptors &) = delete;
            Descriptors(Descriptors &&) = delete;
        };
        auto d = std::make_shared<Descriptors>();
        d->f64 = info.dtParam == DT::F64;

        int strideAx[4]{0, 0, 0, 1},            // to calculate
            dimAp[4]{1, info.dimAx[1], 1, 1},   // 1xCx1x1
            strideAp[4]{info.dimAx[1], 1, 1, 1};// Cx1x1x1
        // TODO: calculate real stride based on layout type
        for (auto i = 3; i > 0; --i) {
            strideAx[i - 1] = strideAx[i] * info.dimAx[i];
        }

        CUDNN_ASSERT(cudnnSetTensorNdDescriptor(d->x, cudnnDataTypeConvert(info.dtX), 4, info.dimAx, strideAx));
        CUDNN_ASSERT(cudnnSetTensorNdDescriptor(d->param, cudnnDataTypeConvert(info.dtParam), 4, dimAp, strideAp));

        res.fetchOrStore<CudnnContext>();

        // nvcc at c++11 doesn't support real move capture
        return [d = std::move(d),
                epsilon = info.epsilon](Resources &res, void const **inputs, void **outputs) {
            // fetch cudnn handle from resources
            auto handle = res.fetchOrStore<CudnnContext>()->handle;
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
            if (d->f64) {
                f64[0] = 1;
                f64[1] = 0;
                alpha = f64;
                beta = f64 + 1;
            } else {
                f32[0] = 1;
                f32[1] = 0;
                alpha = f32;
                beta = f32 + 1;
            }
            CUDNN_ASSERT(cudnnBatchNormalizationForwardInference(
                handle, CUDNN_BATCHNORM_SPATIAL, alpha, beta,
                d->x, x,
                d->x, y, // desc(x) === desc(y) for onnx
                d->param,// scale, bias, mean, var
                scale, bias, mean, var,
                epsilon));
        };
    }

}// namespace refactor::kernel
