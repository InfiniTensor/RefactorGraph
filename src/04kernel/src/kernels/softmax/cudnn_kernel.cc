#include "cudnn_kernel.hh"

namespace refactor::kernel {
    using K = SoftmaxCudnn;

    K::SoftmaxCudnn(cudnn::SoftmaxAlgo algo_,
                    cudnn::SoftmaxMode mode_,
                    DataType type_,
                    std::vector<int> shape_) noexcept : Kernel(), algo(algo_), mode(mode_), dataType(type_), dim(shape_) {}
    auto K::build(cudnn::SoftmaxAlgo algo,
                  cudnn::SoftmaxMode mode,
                  Tensor const &x) noexcept -> KernelBox {
#ifndef USE_CUDA
        return nullptr;
#endif
        // cudnn softmax not support 4D/5D tensor
        if (x.rank() != 4 && x.rank() != 5) {
            return nullptr;
        }
        std::vector<int> shape(x.rank());
        std::transform(x.shape.begin(), x.shape.end(), shape.begin(), [](auto const i) {
            return static_cast<int>(i);
        });
        return std::make_unique<K>(algo, mode, x.dataType, shape);
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