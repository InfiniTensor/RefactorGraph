#include "cpu_kernel.hh"
#include <execution>

namespace refactor::kernel {
    using K = TransposeCpu;
    using Info = TransposeInfo;

    K::TransposeCpu(Info info_) noexcept
        : Kernel(), info(std::move(info_)) {}

    auto K::build(Info info) noexcept -> KernelBox {
        return std::make_unique<K>(std::move(info));
    }
    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t {
        return typeId();
    }
    auto K::description() const noexcept -> std::string_view {
        return "Performing transpose operation on generic cpu";
    }

    auto K::lower(Resources &) const noexcept -> RoutineWorkspace {
        using namespace runtime;
        return [info = this->info](Resources &, void *workspace, void const *const *inputs, void *const *outputs) {
            auto data = reinterpret_cast<uint8_t const *>(inputs[0]);
            auto transposed = reinterpret_cast<uint8_t *>(outputs[0]);
            auto eleSize = info.blockSize;
            std::for_each_n(std::execution::par_unseq,
                            natural_t(0), info.blockCount,
                            [=, &info](auto i) { std::memcpy(transposed + i * eleSize, data + info.locate(i) * eleSize, eleSize); });
        };
    }

}// namespace refactor::kernel
