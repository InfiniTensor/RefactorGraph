#include "cpu_kernel.hh"
#include <execution>

namespace refactor::kernel {
    using K = SliceCpu;

    K::SliceCpu(SliceInfo info_) noexcept
        : Kernel(), info(std::move(info_)) {}

    auto K::build(SliceInfo info) noexcept -> KernelBox {
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
        return "Performing slice operation on generic cpu";
    }

    auto K::lower(Resources &) const noexcept -> RoutineWorkspace {
        using namespace runtime;
        return [info = this->info](Resources &, void *workspace, void const *const *inputs, void *const *outputs) {
            auto src = reinterpret_cast<uint8_t const *>(inputs[0]);
            auto dst = reinterpret_cast<uint8_t *>(outputs[0]);
            std::for_each_n(std::execution::par_unseq,
                            natural_t(0), info.blockCount,
                            [=, &info](auto i) {
                                long rem = i, j = 0;
                                for (auto const &dim : info.dims) {
                                    auto d = std::div(rem, dim.strideO);
                                    j += d.quot * dim.strideI + dim.skip;
                                    rem = d.rem;
                                }
                                std::memcpy(dst + i * info.blockSize, src + j * info.blockSize, info.blockSize);
                            });
        };
    }

}// namespace refactor::kernel
