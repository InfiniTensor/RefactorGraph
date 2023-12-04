#include "../../utilities/cuda/cudnn_context.hh"
#include "../../utilities/cuda/cudnn_functions.h"
#include "../extra_padding/extra_padding.cuh"
#include "cudnn_kernel.hh"

namespace refactor::kernel {
    using namespace cudnn;
    using namespace runtime;
    using Ty = PoolType;

    auto PoolCudnn::lower(Resources &res) const -> RoutineWorkspace {
        // RAII for closure
        struct Descriptors {
            cudnnTensorDescriptor_t x, y;
            cudnnPoolingDescriptor_t pooling;
            std::optional<ExtraPadding> extraPadding;

            Descriptors() {
                CUDNN_ASSERT(cudnnCreateTensorDescriptor(&x));
                CUDNN_ASSERT(cudnnCreateTensorDescriptor(&y));
                CUDNN_ASSERT(cudnnCreatePoolingDescriptor(&pooling));
            }
            ~Descriptors() noexcept(false) {
                CUDNN_ASSERT(cudnnDestroyTensorDescriptor(x));
                CUDNN_ASSERT(cudnnDestroyTensorDescriptor(y));
                CUDNN_ASSERT(cudnnDestroyPoolingDescriptor(pooling));
            }

            Descriptors(const Descriptors &) = delete;
            Descriptors(Descriptors &&) = delete;
        };
        auto d = std::make_shared<Descriptors>();
        d->extraPadding = ExtraPadding::build(info.dt, info.xShape, info.pads);
        int const
            xs[]{
                info.xShape[0],
                info.xShape[1],
                info.xShape[2] + std::abs(info.pads[0] - info.pads[2]),
                info.xShape[3] + std::abs(info.pads[1] - info.pads[3]),
            },
            *ys = info.yShape;
        setCudnnTensor(d->x, info.dt, slice(xs, 4));
        setCudnnTensor(d->y, info.dt, slice(ys, 4));

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
            std::min(pp[0], pp[2]), std::min(pp[1], pp[3]),
            ss[0], ss[1]));

        res.fetchOrStore<CudnnContext>();
        // nvcc at c++11 doesn't support real move capture
        auto routine = [d](Resources &res, void *workspace, void const *const *inputs, void *const *outputs) {
            void const *x = inputs[0];
            if (auto f = d->extraPadding; f) {
                x = (*f)(x, workspace);
            }
            // TODO? build alpha/beta for double
            float alpha = 1, beta = 0;
            CUDNN_ASSERT(cudnnPoolingForward(
                res.fetchOrStore<CudnnContext>()->handle,
                d->pooling,
                &alpha,
                d->x, x,
                &beta,
                d->y, outputs[0]));
        };
        return {std::move(routine), d->extraPadding ? d->extraPadding->workspace() : 0};
    }

}// namespace refactor::kernel
