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

    template<decltype(DataType::internal) T>
    Routine lowerTyped(SoftmaxInfo info) {
        using namespace runtime;
        using dt = typename primitive<T>::type;

        return [info](Resources &, void *workspace, void const *const *inputs, void *const *outputs) {
            auto x = reinterpret_cast<dt const *>(inputs[0]);
            auto y = reinterpret_cast<dt *>(outputs[0]);

            std::for_each_n(std::execution::par_unseq,
                            natural_t(0), info.post * info.pre,
                            [&x, &y, &info](auto const i) {
                                auto post = info.post;
                                auto calcIdx =
                                    [base = i / post * post * info.mid + i % post,
                                     post](auto const j) {
                                        return base + j * post;
                                    };
                                auto range = range0_(info.mid);
                                auto maxi = *std::max_element(
                                    std::execution::unseq,
                                    range.begin(), range.end(),
                                    [&](auto const m, auto const n) {
                                        return x[calcIdx(m)] < x[calcIdx(n)];
                                    });
                                auto sum = std::accumulate(
                                    range.begin(), range.end(), 0,
                                    [&, max = x[calcIdx(maxi)]](auto const acc, auto const j) {
                                        auto idx = calcIdx(j);
                                        return acc + (y[idx] = std::exp(x[idx] - max));
                                    });
                                std::for_each(
                                    std::execution::par_unseq,
                                    range.begin(), range.end(),
                                    [&](auto const j) {
                                        y[calcIdx(j)] /= sum;
                                    });
                            });
        };
    }
    auto K::lower(Resources &) const noexcept -> RoutineWorkspace {
#define CASE(T) \
    case T:     \
        return lowerTyped<DataType::T>(info);

        switch (info.type) {
            CASE(DataType::F32);
            CASE(DataType::F64);
            // CASE(DataType::FP16);
            // CASE(DataType::BF16);
            default:
                UNREACHABLE();
        }
    }

}// namespace refactor::kernel
