#include "cuda_kernel.hh"

namespace refactor::kernel {
    using K = SoftmaxCuda;

    K::SoftmaxCuda(SoftmaxInfo info_) noexcept
        : Kernel(), info(std::move(info_)) {}

    auto K::build(SoftmaxInfo info) noexcept -> KernelBox {
        static const std::unordered_set<decltype(DataType::internal)>
            TYPES{DataType::F32, DataType::F64, DataType::FP16, DataType::BF16};

#ifndef USE_CUDA
        return nullptr;
#endif

        return TYPES.contains(info.type)
                   ? std::make_unique<K>(std::move(info))
                   : nullptr;
    }

    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t { return typeId(); }
    auto K::description() const noexcept -> std::string_view {
        return "Performing Softmax using CUDA";
    }

}// namespace refactor::kernel
