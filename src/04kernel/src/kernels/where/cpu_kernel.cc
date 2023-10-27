#include "cpu_kernel.hh"
#include <execution>

namespace refactor::kernel {
    using K = WhereCpu;
    using DT = DataType;

    K::WhereCpu(DT dataType_, WhereBroadcast b) noexcept
        : Kernel(),
          dataType(dataType_),
          broadcast(std::move(b)) {}

    auto K::build(Tensor const &c, Tensor const &x, Tensor const &y, Tensor const &output) noexcept -> KernelBox {
        return std::make_unique<K>(x.dataType, WhereBroadcast(c.shape, x.shape, y.shape, output.shape));
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

    auto K::lower() const noexcept -> Routine {
        using namespace runtime;

        return [eleSize = dataType.size(),
                broadcast = this->broadcast](runtime::Resources &, void const **inputs, void **outputs) {
            auto c = reinterpret_cast<bool const *>(inputs[0]);
            auto x = reinterpret_cast<uint8_t const *>(inputs[1]);
            auto y = reinterpret_cast<uint8_t const *>(inputs[2]);
            auto output = reinterpret_cast<uint8_t *>(outputs[0]);
            for (auto i : range0_(broadcast.size())) {
                auto [ic, ix, iy] = broadcast.locate(i);
                std::memcpy(output + i * eleSize,
                            c[ic]
                                ? x + ix * eleSize
                                : y + iy * eleSize,
                            eleSize);
            }
        };
    }

}// namespace refactor::kernel
