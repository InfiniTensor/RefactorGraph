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

        return [info](Resources &, void const **inputs, void **outputs) {
            auto x = reinterpret_cast<dt const *>(inputs[0]);
            auto y = reinterpret_cast<dt *>(outputs[0]);
            auto max = x[0];
            for (auto i : range0_(info.size)) {
                if (max < x[i]) {
                    max = x[i];
                }
            }
            auto getSum = [info, max, x](auto const i) {
                dt sum = 0;
                for (auto j : range0_(info.mid)) {
                    auto d = std::div((long) i, info.post);
                    auto index = d.quot * info.mid * info.post + j * info.post + d.rem;
                    sum += std::exp(x[index] - max);
                }
                return sum;
            };
            std::for_each_n(std::execution::par_unseq, natural_t(0), info.post * info.pre,
                            [&x, &y, max, getSum, info](auto const i) {
                                auto sum = getSum(i);
                                std::for_each_n(std::execution::par_unseq, natural_t(0), info.mid,
                                                [&](auto const j) {
                                                    auto d = std::div((long) i, info.post);
                                                    auto index = d.quot * info.mid * info.post + j * info.post + d.rem;
                                                    y[index] = std::exp(x[index] - max) / sum;
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
