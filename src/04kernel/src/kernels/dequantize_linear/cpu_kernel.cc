#include "cpu_kernel.hh"
#include <execution>
#include <numeric>

namespace refactor::kernel {
    using K = DequantizeLinearCpu;

    K::DequantizeLinearCpu(
        decltype(from) from_,
        decltype(size) size_,
        decltype(withZeroPoint) withZeroPoint_) noexcept
        : Kernel(),
          from(from_),
          size(size_),
          withZeroPoint(withZeroPoint_) {}

    auto K::build(TensorRefs const &inputs, Tensor const &output) noexcept -> KernelBox {
        if (inputs[1].get().elementsSize() != 1) {
            return nullptr;
        }
        if (output.dataType != DataType::F32) {
            return nullptr;
        }
        return std::make_unique<K>(
            inputs[0].get().dataType,
            inputs[0].get().elementsSize(),
            inputs.size() > 2);
    }

    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t { return typeId(); }
    auto K::description() const noexcept -> std::string_view {
        return "Performing dequantize linear using CPU";
    }

    template<class TI, class TO>
    auto lowerTyped(size_t size, bool withZeroPoint) noexcept -> RoutineWorkspace {

        return [size, withZeroPoint]//
            (Resources &, void *, void const *const *inputs, void *const *outputs) {
                auto x = reinterpret_cast<TI const *>(inputs[0]);
                auto scale = *reinterpret_cast<TO const *>(inputs[1]);
                auto zp = withZeroPoint ? *reinterpret_cast<TI const *>(inputs[2]) : 0;
                auto y = reinterpret_cast<TO *>(outputs[0]);
                std::transform(
                    std::execution::par_unseq,
                    x, x + size,
                    y,
                    [scale, zp](TI x) {
                        return static_cast<TO>(x - zp) * scale;
                    });
            };
    }

    auto K::lower(Resources &) const noexcept -> RoutineWorkspace {
        switch (from) {
            case DataType::U8:
                return lowerTyped<uint8_t, float>(size, withZeroPoint);
            case DataType::U16:
                return lowerTyped<uint16_t, float>(size, withZeroPoint);
            case DataType::I8:
                return lowerTyped<int8_t, float>(size, withZeroPoint);
            case DataType::I16:
                return lowerTyped<int16_t, float>(size, withZeroPoint);
            case DataType::I32:
                return lowerTyped<int32_t, float>(size, withZeroPoint);
            default:
                UNREACHABLE();
        }
    }

}// namespace refactor::kernel
