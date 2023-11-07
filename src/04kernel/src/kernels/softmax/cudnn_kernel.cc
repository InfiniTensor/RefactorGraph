#include "cudnn_kernel.hh"

namespace refactor::kernel {
    using K = SoftmaxCudnn;

    K::SoftmaxCudnn(cudnn::SoftmaxAlgo algo_,
                    cudnn::SoftmaxMode mode_,
                    DataType type_,
                    std::vector<int> shape_) noexcept : Kernel(), algo(algo_), mode(mode_), dataType(type_), dim(shape_) {}
    auto K::build(cudnn::SoftmaxAlgo algo,
                  cudnn::SoftmaxMode mode,
                  SoftmaxInfo info) noexcept -> KernelBox {
#ifndef USE_CUDA
        return nullptr;
#endif
        std::vector<int> shape = {info.pre, info.mid, info.post, 1};
        return std::make_unique<K>(algo, mode, info.type, shape);
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
