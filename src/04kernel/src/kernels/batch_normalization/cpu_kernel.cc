#include "cpu_kernel.hh"
#include "common/error_handler.h"

namespace refactor::kernel {
    using K = BatchNormalization;
    using DT = common::DataType;

    K::BatchNormalization(
        float epsilon_,
        std::array<DT, 3> dts_,
        Shape shape_,
        uint32_t paramSize_) noexcept
        : Kernel(),
          epsilon(epsilon_),
          dts(dts_),
          shape(std::move(shape_)),
          paramSize(paramSize_) {}

    auto K::build(float epsilon, TensorRefs inputs) noexcept -> KernelBox {
        auto const &x = inputs[0].get();
        auto const &scale = inputs[1].get();
        auto const &mean = inputs[3].get();
        std::array<DT, 3> dts{x.dataType, scale.dataType, mean.dataType};
        return std::make_unique<K>(epsilon, dts, x.shape, scale.shape[0]);
    }
    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t { return typeId(); }
    auto K::description() const noexcept -> std::string_view {
        return "Performing batch normalization for non-training-mode on generic cpu";
    }
    auto K::lower() const noexcept -> Operation {
        using namespace runtime;
        return [](Resources &, Addresses inputs, Addresses outputs) {
            auto x = inputs[0],
                 scale = inputs[1],
                 bias = inputs[2],
                 mean = inputs[3],
                 var = inputs[4];
            auto y = outputs[0];

            TODO("");
        };
    }

}// namespace refactor::kernel
