#include "cpu_kernel.hh"
#include <execution>
#include <unordered_set>

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

    auto K::lower() const noexcept -> Routine {
        using namespace runtime;

        return [info = this->info](Resources &, void const **inputs, void **outputs) {
            fmt::println("size = {}", info.size);
            for (auto i : range0_(info.dims.size())) {
                fmt::println("i = {}, stride = {}, perm = {}", i, info.dims[i].stride, info.dims[i].permutation);
            }
        };
    }

}// namespace refactor::kernel
