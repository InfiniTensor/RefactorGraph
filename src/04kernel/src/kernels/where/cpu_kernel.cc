#include "cpu_kernel.hh"
#include <execution>

namespace refactor::kernel {
    using K = WhereCpu;
    using DT = DataType;

    K::WhereCpu(DT dataType_, Broadcaster b) noexcept
        : Kernel(),
          dataType(dataType_),
          broadcaster(std::move(b)) {}

    auto K::build(TensorRefs const &inputs) noexcept -> KernelBox {
        return std::make_unique<K>(inputs[1].get().dataType, Broadcaster(inputs));
    }
    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t {
        return typeId();
    }
    auto K::description() const noexcept -> std::string_view {
        return "Performing where operation on generic cpu";
    }

    auto K::lower(Resources &) const noexcept -> RoutineWorkspace {
        using namespace runtime;

        return [broadcaster = this->broadcaster,
                eleSize = dataType.size()](runtime::Resources &, void *workspace, void const *const *inputs, void *const *outputs) {
            auto c = reinterpret_cast<bool const *>(inputs[0]);
            auto x = reinterpret_cast<uint8_t const *>(inputs[1]);
            auto y = reinterpret_cast<uint8_t const *>(inputs[2]);
            auto output = reinterpret_cast<uint8_t *>(outputs[0]);
            dim_t ii[3];
            for (auto i : range0_(broadcaster.outputsCount)) {
                broadcaster.locate(i, ii);
                std::memcpy(output + i * eleSize,
                            c[ii[0]]
                                ? x + ii[1] * eleSize
                                : y + ii[2] * eleSize,
                            eleSize);
            }
        };
    }

}// namespace refactor::kernel
