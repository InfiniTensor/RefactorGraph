#include "cudnn_kernel.hh"

#ifdef USE_CUDA
#include "../../utilities/cuda/cudnn_context.hh"
#include "../../utilities/cuda/cudnn_functions.h"
#endif

namespace refactor::kernel {
    using K = SoftmaxCudnn;

    K::SoftmaxCudnn(cudnn::SoftmaxAlgo algo_, DataType type_,
                    int pre_, int mid_, int post_) noexcept
        : Kernel(), algo(algo_), dataType(type_),
          pre(pre_), mid(mid_), post(post_) {}

    auto K::build(cudnn::SoftmaxAlgo algo, SoftmaxInfo info) noexcept -> KernelBox {
#ifndef USE_CUDA
        return nullptr;
#endif

        return std::make_unique<K>(algo, info.type, info.pre, info.mid, info.post);
    }
    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t {
        return typeId();
    }
    auto K::description() const noexcept -> std::string_view {
        return "Performing softmax forward with CUDNN";
    }

#ifdef USE_CUDA

    auto SoftmaxCudnn::lower(Resources &res) const -> RoutineWorkspace {
        using namespace cudnn;
        using namespace runtime;

        // RAII for closure
        struct Descriptors {
            cudnnTensorDescriptor_t t;
            cudnnSoftmaxAlgorithm_t algo;
            bool f32;

            Descriptors(decltype(algo) algo_, decltype(f32) f32_)
                : algo(algo_), f32(f32_) {
                CUDNN_ASSERT(cudnnCreateTensorDescriptor(&t));
            }
            ~Descriptors() noexcept(false) {
                CUDNN_ASSERT(cudnnDestroyTensorDescriptor(t));
            }
            Descriptors(const Descriptors &) = delete;
            Descriptors(Descriptors &&) = delete;
        };

        auto d = std::make_shared<Descriptors>(
            static_cast<cudnnSoftmaxAlgorithm_t>(algo),
            dataType != DataType::F64);
        int dims[]{pre, mid, post, 1};
        setCudnnTensor(d->t, dataType, slice(dims, 4));

        res.fetchOrStore<CudnnContext>();
        return [d = std::move(d)](Resources &res, void *workspace, void const *const *inputs, void *const *outputs) {
            // build alpha/beta for double
            auto a = d->f32 ? factor<fp32_t>(1) : factor<fp64_t>(1),
                 b = d->f32 ? factor<fp32_t>(0) : factor<fp64_t>(0);
            CUDNN_ASSERT(cudnnSoftmaxForward(
                res.fetchOrStore<CudnnContext>()->handle,
                d->algo,
                CUDNN_SOFTMAX_MODE_CHANNEL,
                &a, d->t, inputs[0],
                &b, d->t, outputs[0]));
        };
    }

#endif

}// namespace refactor::kernel
