#include "cpu_kernel.hh"
#include <execution>

namespace refactor::kernel {
    using K = ClipCpu;

    K::ClipCpu(decltype(dataType) dt,
               decltype(size) size_,
               decltype(hasMax) hasMax_) noexcept
        : dataType(dt), size(size_), hasMax(hasMax_) {}

    auto K::build(Tensor const &data, bool hasMax) noexcept -> KernelBox {
        return data.dataType.isCpuNumberic()
                   ? std::make_unique<K>(data.dataType, data.elementsSize(), hasMax)
                   : nullptr;
    }
    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t {
        return typeId();
    }
    auto K::description() const noexcept -> std::string_view {
        return "Performing clip operation on generic cpu";
    }

    template<class T>
    auto lowerTyped(size_t size, bool hasMax) noexcept -> RoutineWorkspace {
        using namespace runtime;
        return [=](Resources &, void *workspace, void const *const *inputs, void *const *outputs) {
            auto data = reinterpret_cast<T const *>(inputs[0]);
            auto min = *reinterpret_cast<T const *>(inputs[1]),
                 max = hasMax
                           ? *reinterpret_cast<T const *>(inputs[2])
                           : std::numeric_limits<T>::max();
            auto output = reinterpret_cast<T *>(outputs[0]);

            std::transform(std::execution::par_unseq,
                           data, data + size,
                           output,
                           [=](auto x) { return std::clamp(x, min, max); });
        };
    }

    auto K::lower(Resources &) const noexcept -> RoutineWorkspace {
#define CASE(DT)       \
    case DataType::DT: \
        return lowerTyped<primitive<DataType::DT>::type>(size, hasMax)

        switch (dataType) {
            CASE(F32);
            CASE(U8);
            CASE(I8);
            CASE(U16);
            CASE(I16);
            CASE(I32);
            CASE(I64);
            CASE(F64);
            CASE(U32);
            CASE(U64);
            default:
                UNREACHABLE();
        }
    }

}// namespace refactor::kernel
