#include "../../utilities/cuda/cudnn_context.hh"
#include "../../utilities/cuda/cudnn_functions.h"
#include "cudnn_kernel.hh"

namespace refactor::kernel {
    using namespace cudnn;
    using namespace runtime;
    using Ty = PoolType;

    auto PoolCudnn::lower() const noexcept -> Routine {
        // RAII for closure
        struct Descriptors {
            cudnnTensorDescriptor_t x, y;
            cudnnPoolingDescriptor_t pooling;

            Descriptors() {
                CUDNN_ASSERT(cudnnCreateTensorDescriptor(&x));
                CUDNN_ASSERT(cudnnCreateTensorDescriptor(&y));
                CUDNN_ASSERT(cudnnCreatePoolingDescriptor(&pooling));
            }
            ~Descriptors() {
                CUDNN_ASSERT(cudnnDestroyTensorDescriptor(x));
                CUDNN_ASSERT(cudnnDestroyTensorDescriptor(y));
                CUDNN_ASSERT(cudnnDestroyPoolingDescriptor(pooling));
            }

            Descriptors(const Descriptors &) = delete;
            Descriptors(Descriptors &&) = delete;
        };
        auto d = std::make_shared<Descriptors>();

        auto cudnnDataType = cudnnDataTypeConvert(info.dt);
        auto xs = info.xShape;
        auto ys = info.yShape;
        CUDNN_ASSERT(cudnnSetTensor4dDescriptor(d->x, CUDNN_TENSOR_NCHW, cudnnDataType, xs[0], xs[1], xs[2], xs[3]));
        CUDNN_ASSERT(cudnnSetTensor4dDescriptor(d->y, CUDNN_TENSOR_NCHW, cudnnDataType, ys[0], ys[1], ys[2], ys[3]));

        // clang-format off
        auto mode = info.poolType == Ty::Average ? CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING
                  : info.poolType == Ty::Max     ? CUDNN_POOLING_MAX
                                                 : UNREACHABLEX(cudnnPoolingMode_t, "");
        // clang-format on
        auto pp = info.pads;
        auto ss = info.strides;
        auto kk = info.kernelShape;
        CUDNN_ASSERT(cudnnSetPooling2dDescriptor(
            d->pooling,
            mode, CUDNN_PROPAGATE_NAN,
            kk[0], kk[1],
            pp[0], pp[1],
            ss[0], ss[1]));

        // nvcc at c++11 doesn't support real move capture
        return [d_ = std::move(d)](Resources &res, void const **inputs, void **outputs) {
            // fetch cudnn handle from resources
            auto handle = res.fetchOrStore<CudnnContext>()->handle;
            auto const &d = *d_;
            // name inputs and outputs
            auto x = inputs[0];
            auto y = outputs[0];
            // TODO? build alpha/beta for double
            float alpha = 1, beta = 0;
            CUDNN_ASSERT(cudnnPoolingForward(
                handle,
                d.pooling,
                &alpha,
                d.x, x,
                &beta,
                d.y, y));
        };
    }

}// namespace refactor::kernel
