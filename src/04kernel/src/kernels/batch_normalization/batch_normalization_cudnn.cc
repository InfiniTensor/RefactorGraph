#include "batch_normalization_cudnn.hh"
#include "common/error_handler.h"
#include "cudnn_impl.h"

namespace refactor::kernel {
    using K = BatchNormalizationCudnn;
    using DT = common::DataType;

    K::BatchNormalizationCudnn(
        float epsilon_,
        DT dataType_,
        Shape shape_,
        uint32_t valueSize) noexcept
        : Kernel(),
          epsilon(epsilon_),
          dataType(dataType_),
          shape(std::move(shape_)),
          valueSize(valueSize) {}

    auto K::build(float epsilon, TensorRefs inputs) noexcept -> KernelBox {
#ifndef USE_CUDA
        return nullptr;
#endif

        auto const &x = inputs[0].get();
        auto const &scale = inputs[1].get();
        auto const &bias = inputs[2].get();
        auto const &mean = inputs[3].get();
        auto const &var = inputs[4].get();

        // see "Supported Configurations for `cudnnBatchNormalizationForwardInference`"
        if (DT::F64 == x.dataType) {
            if (DT::F64 != scale.dataType || DT::F64 != mean.dataType) {
                return nullptr;
            }
        } else {
            if (DT::F32 != scale.dataType || DT::F32 != mean.dataType) {
                return nullptr;
            }
        }

        return std::make_unique<K>(epsilon, x.dataType, x.shape, scale.shape[0]);
    }
    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t { return typeId(); }
    auto K::description() const noexcept -> std::string_view {
        return "Performing batch normalization for non-training-mode using CUDNN";
    }
    auto K::lower() const noexcept -> Operation {
        TODO("");
    }

}// namespace refactor::kernel
