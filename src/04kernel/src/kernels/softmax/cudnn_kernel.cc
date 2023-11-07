#include "cudnn_kernel.hh"

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

}// namespace refactor::kernel
