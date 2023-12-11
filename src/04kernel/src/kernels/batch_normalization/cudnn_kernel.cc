#include "cudnn_kernel.hh"

#ifdef USE_CUDA
#include "../../utilities/cuda/cudnn_context.hh"
#include "../../utilities/cuda/cudnn_functions.h"
#include <cudnn.h>
#endif

namespace refactor::kernel {
    using K = BatchNormalizationCudnn;
    using DT = DataType;

    K::BatchNormalizationCudnn(decltype(info) info_) noexcept
        : info(info_) {}

    auto K::build(float epsilon, TensorRefs inputs) noexcept -> KernelBox {
#ifndef USE_CUDA
        return nullptr;
#endif

        auto const &x = inputs[0].get();
        auto const &scale = inputs[1].get();
        auto const &mean = inputs[3].get();

        if (x.rank() != 4) {
            return nullptr;
        }

        // see "Supported Configurations for `cudnnBatchNormalizationForwardInference`"
        if (scale.dataType != mean.dataType) {
            return nullptr;
        }
        if (x.dataType == DT::F64) {
            if (scale.dataType != DT::F64) {
                return nullptr;
            }
        } else {
            if (scale.dataType != DT::F32) {
                return nullptr;
            }
        }
        return std::make_unique<K>(decltype(info){
            epsilon,
            x.dataType,
            scale.dataType,
            x.layout,
            {
                static_cast<int>(x.shape[0]),
                static_cast<int>(x.shape[1]),
                static_cast<int>(x.shape[2]),
                static_cast<int>(x.shape[3]),
            }});
    }
    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t { return typeId(); }
    auto K::description() const noexcept -> std::string_view {
        return "Performing batch normalization for non-training-mode using CUDNN";
    }

#ifdef USE_CUDA

    auto K::lower(Resources &res) const -> RoutineWorkspace {
        using namespace cudnn;
        using namespace runtime;
        using DT = DataType;

        // RAII for closure
        struct Descriptors {
            cudnnTensorDescriptor_t x, p;
            bool f32;

            explicit Descriptors(decltype(f32) f32_)
                : x(nullptr), p(nullptr), f32(f32_) {
                CUDNN_ASSERT(cudnnCreateTensorDescriptor(&x));
                CUDNN_ASSERT(cudnnCreateTensorDescriptor(&p));
            }
            ~Descriptors() noexcept(false) {
                CUDNN_ASSERT(cudnnDestroyTensorDescriptor(x));
                CUDNN_ASSERT(cudnnDestroyTensorDescriptor(p));
            }

            Descriptors(const Descriptors &) = delete;
            Descriptors(Descriptors &&) = delete;
        };
        auto d = std::make_shared<Descriptors>(info.dtX != DT::F64);
        int dimParam[]{1, info.dimAx[1], 1, 1};
        setCudnnTensor(d->x, info.dtX, slice(info.dimAx, 4));
        setCudnnTensor(d->p, info.dtP, slice(dimParam, 4));

        res.fetchOrStore<CudnnContext>();
        // nvcc at c++11 doesn't support real move capture
        return [d = std::move(d),
                epsilon = info.epsilon](Resources &res, void *workspace, void const *const *inputs, void *const *outputs) {
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
            auto a = d->f32 ? factor<fp32_t>(1) : factor<fp64_t>(1),
                 b = d->f32 ? factor<fp32_t>(0) : factor<fp64_t>(0);
            CUDNN_ASSERT(cudnnBatchNormalizationForwardInference(
                handle, CUDNN_BATCHNORM_SPATIAL, &a, &b,
                d->x, x,
                d->x, y,// desc(x) === desc(y) for onnx
                d->p, scale, bias, mean, var,
                epsilon));
        };
    }

#endif

}// namespace refactor::kernel
