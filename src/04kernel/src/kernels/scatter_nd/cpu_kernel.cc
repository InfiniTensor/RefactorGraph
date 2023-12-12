#include "cpu_kernel.hh"
#include <execution>

namespace refactor::kernel {
    using K = ScatterNDCpu;

    K::ScatterNDCpu(decltype(info) info_)
        : Kernel(), info(std::move(info_)) {}

    auto K::build(decltype(info) info) noexcept -> KernelBox {
        return std::make_unique<ScatterNDCpu>(std::move(info));
    }
    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t {
        return typeId();
    }
    auto K::description() const noexcept -> std::string_view {
        return "Performing scatterNd operation on generic cpu";
    }

    auto K::lower(Resources &) const noexcept -> RoutineWorkspace {
        return [info = this->info](Resources &, void *, void const *const *inputs, void *const *outputs) {
            if (outputs[0] != inputs[0]) {
                std::memcpy(outputs[0], inputs[0], info.blockCount * info.blockSize);
            }

            auto out = static_cast<uint8_t *>(outputs[0]);
            auto in = static_cast<uint8_t const *>(inputs[2]);
            auto indices = static_cast<int64_t const *>(inputs[1]);
            std::for_each_n(
                std::execution::par_unseq,
                natural_t(0), info.prefix,
                [=, &info](auto i) {
                    auto j = 0;
                    for (auto i_ = indices + i * info.strides.size();
                         auto k : range0_(info.strides.size())) {
                        j += i_[k] * info.strides[k];
                    }

                    std::memcpy(out + j * info.blockSize,
                                in + i * info.blockSize,
                                info.blockSize);
                });
        };
    }

}// namespace refactor::kernel
