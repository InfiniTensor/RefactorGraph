#include "cpu_kernel.hh"
#include <algorithm>
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

        return [info](Resources &, void const **inputs, void **outputs) {
            auto x = reinterpret_cast<dt const *>(inputs[0]);
            auto y = reinterpret_cast<dt *>(outputs[0]);

            std::for_each_n(std::execution::par_unseq, natural_t(0), info.post * info.pre,
                            [&x, &y, info](auto const i) {
                                auto d = std::div((long) i, info.post);
                                auto indexPartial = d.quot * info.mid * info.post + d.rem;
                                auto maxii = std::max_element(natural_t(0u), natural_t(info.mid), [&](auto const &m, auto const &n) {
                                    return x[m * info.post + indexPartial] < x[n * info.post + indexPartial];
                                });
                                auto max = x[*maxii * info.post + indexPartial];
                                auto sum = std::accumulate(natural_t(0u), natural_t(info.mid), 0, [&](auto acc, auto const k) {
                                    auto index = indexPartial + k * info.post;
                                    y[index] = std::exp(x[index] - max);
                                    return acc + y[index];
                                });
                                std::for_each_n(std::execution::par_unseq, natural_t(0), info.mid,
                                                [&](auto const j) {
                                                    auto index = indexPartial + j * info.post;
                                                    y[index] /= sum;
                                                });
                            });
        };
    }
    Routine K::lower(Resources &) const noexcept {
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
