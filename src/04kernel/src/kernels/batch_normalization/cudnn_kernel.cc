#include "cudnn_kernel.hh"

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

}// namespace refactor::kernel
