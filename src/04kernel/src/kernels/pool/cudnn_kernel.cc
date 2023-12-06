#include "cudnn_kernel.hh"

#ifdef USE_CUDA
#include "../../utilities/cuda/cudnn_context.hh"
#include "../../utilities/cuda/cudnn_functions.h"
#include "../extra_padding/extra_padding.cuh"
#endif

namespace refactor::kernel {
    using K = PoolCudnn;

    K::PoolCudnn(decltype(info) info_) noexcept
        : Kernel(), info(std::move(info_)) {}

    auto K::build(PoolType poolType,
                  bool ceil,
                  KernelShape const &kernelShape,
                  PoolAttributes const &poolAttributes,
                  Tensor const &x,
                  Tensor const &y) noexcept -> KernelBox {
#ifndef USE_CUDA
        return nullptr;
#endif

        // TODO check data type
        auto p = poolAttributes.pads(),
             d = poolAttributes.dilations(),
             s = poolAttributes.strides();
        if (x.rank() != 4 ||
            poolType == PoolType::Lp ||
            d[0] != 1 || d[1] != 1) {
            return nullptr;
        }
        return std::make_unique<K>(decltype(info){
            poolType,
            x.dataType,
            {
                static_cast<int>(x.shape[0]),
                static_cast<int>(x.shape[1]),
                static_cast<int>(x.shape[2]),
                static_cast<int>(x.shape[3]),
            },
            {
                static_cast<int>(y.shape[0]),
                static_cast<int>(y.shape[1]),
                static_cast<int>(y.shape[2]),
                static_cast<int>(y.shape[3]),
            },
            {
                static_cast<int>(kernelShape[0]),
                static_cast<int>(kernelShape[1]),
            },
            {p[0], p[1], p[2], p[3]},
            {s[0], s[1]},
        });
    }

    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t { return typeId(); }
    auto K::description() const noexcept -> std::string_view {
        return "Performing pool using CUDNN";
    }

#ifdef USE_CUDA

    auto PoolCudnn::lower(Resources &res) const -> RoutineWorkspace {
        using namespace cudnn;
        using namespace runtime;
        using Ty = PoolType;

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
#endif

}// namespace refactor::kernel
