#include "cpu_kernel.hh"
#include <execution>
#include <numeric>

namespace refactor::kernel {
    using K = DynamicQuantizeLinearCpu;

    K::DynamicQuantizeLinearCpu(decltype(size) size_) noexcept
        : Kernel(), size(size_) {}

    auto K::build(decltype(size) size) noexcept -> KernelBox {
        return std::make_unique<K>(size);
    }

    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t { return typeId(); }
    auto K::description() const noexcept -> std::string_view {
        return "Performing dynamic quantize linear using CPU";
    }

    auto K::lower(Resources &) const noexcept -> RoutineWorkspace {
        using namespace runtime;
        return [size = size](Resources &, void *, void const *const *inputs, void *const *outputs) {
            using TI = float;
            using TO = uint8_t;

            constexpr static auto
                ZERO = static_cast<TI>(0),
                _MIN = std::numeric_limits<TI>::min(),
                _MAX = std::numeric_limits<TI>::max(),
                QMIN = static_cast<TI>(std::numeric_limits<TO>::min()),
                QMAX = static_cast<TI>(std::numeric_limits<TO>::max()),
                QLEN = QMAX - QMIN;

            auto x = reinterpret_cast<TI const *>(inputs[0]);
            auto [min, max] = std::accumulate(
                x, x + size,
                std::pair{_MAX, _MIN},
                [](auto acc, auto it) {
                    auto [min, max] = acc;
                    return std::pair{
                        std::min(min, it),
                        std::max(max, it),
                    };
                });
            auto len = std::max(ZERO, max) - std::min(ZERO, min);
            auto scale = len / QLEN;
            auto zp = static_cast<TO>(std::round(QMIN - min * QLEN / len));

            std::transform(
                std::execution::par_unseq,
                x, x + size,
                reinterpret_cast<TO *>(outputs[0]),
                [=](auto it) { return static_cast<TO>(std::round(it / scale) + zp); });
            *reinterpret_cast<TI *>(outputs[1]) = scale;
            *reinterpret_cast<TO *>(outputs[2]) = zp;
        };
    }

}// namespace refactor::kernel
