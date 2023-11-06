#include "cpu_kernel.hh"
#include <execution>

namespace refactor::kernel {
    using K = GatherCpu;

    K::GatherCpu(GatherInfo info_) noexcept
        : Kernel(), info(std::move(info_)) {}

    auto K::build(GatherInfo info) noexcept -> KernelBox {
        return std::make_unique<K>(std::move(info));
    }

    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t { return typeId(); }
    auto K::description() const noexcept -> std::string_view {
        return "Performing gather using CPU";
    }

    Routine K::lower(Resources &) const noexcept {
        using namespace runtime;

        return [info = this->info](Resources &, void const **inputs, void **outputs) {
            auto data = reinterpret_cast<uint8_t const *>(inputs[0]);
            auto output = reinterpret_cast<uint8_t *>(outputs[0]);
            auto policy = std::execution::par_unseq;
            std::for_each_n(policy, natural_t(0), info.prefix * info.midSizeO, [&](auto i) {
                auto d = std::div(i, info.midSizeO);
                int64_t k = info.idxType == DataType::I64
                                ? reinterpret_cast<int64_t const *>(inputs[1])[d.rem]
                                : reinterpret_cast<int32_t const *>(inputs[1])[d.rem];
                std::memcpy(info.postfix * i + output,
                            info.postfix * (d.quot * info.midSizeI + k) + data,
                            info.postfix);
            });
        };
    }

}// namespace refactor::kernel
