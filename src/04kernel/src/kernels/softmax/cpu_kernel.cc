#include "cpu_kernel.hh"
#include <execution>

namespace refactor::kernel {
    using K = SoftmaxCpu;

    K::SoftmaxCpu(SoftmaxInfo info_) noexcept
        : Kernel(), info(std::move(info_)) {}

    auto K::build(SoftmaxInfo info) noexcept -> KernelBox {
        if (!info.type.isCpuNumberic()) {
            return nullptr;
        }
        return std::make_unique<K>(std::move(info));
    }

    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t { return typeId(); }
    auto K::description() const noexcept -> std::string_view {
        return "Performing Softmax using CPU";
    }

    template<class T>
    static Routine lowerTyped(SoftmaxInfo info) {
        using namespace runtime;

        return [info](Resources &, void *workspace, void const *const *inputs, void *const *outputs) {
            std::for_each_n(std::execution::par_unseq,
                            natural_t(0), info.pre * info.post,
                            [x = reinterpret_cast<T const *>(inputs[0]),
                             y = reinterpret_cast<T *>(outputs[0]),
                             mid = info.mid,
                             stride = info.post](auto const i) {
                                auto id = (i - i % stride) * mid + i % stride;
                                auto range = range0_(mid);
                                auto maxi = *std::max_element(
                                    std::execution::unseq,
                                    range.begin(), range.end(),
                                    [&](auto const m, auto const n) {
                                        return x[id + m * stride] < x[id + n * stride];
                                    });
                                auto sum = std::accumulate(
                                    range.begin(), range.end(), 0,
                                    [&, max = x[id + maxi * stride]](auto const acc, auto const j) {
                                        auto k = id + j * stride;
                                        return acc + (y[k] = std::exp(x[k] - max));
                                    });
                                std::for_each(
                                    std::execution::par_unseq,
                                    range.begin(), range.end(),
                                    [&](auto const j) {
                                        y[id + j * stride] /= sum;
                                    });
                            });
        };
    }
    auto K::lower(Resources &) const noexcept -> RoutineWorkspace {
        switch (info.type) {
            case DataType::F32:
                return lowerTyped<float>(info);
            case DataType::F64:
                return lowerTyped<double>(info);
            default:
                UNREACHABLE();
        }
    }

}// namespace refactor::kernel
