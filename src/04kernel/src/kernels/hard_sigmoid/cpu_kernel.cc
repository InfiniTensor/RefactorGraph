#include "cpu_kernel.hh"
#include <execution>

namespace refactor::kernel {
    using K = HardSigmoidCpu;
    using DT = DataType;

    K::HardSigmoidCpu(float alpha_, float beta_, DT dataType_, size_t size_) noexcept
        : Kernel(), alpha(alpha_), beta(beta_), dataType(dataType_), size(size_) {}

    auto K::build(float alpha_, float beta_, Tensor const &a) noexcept -> KernelBox {
        if (!a.dataType.isCpuNumberic()) {
            return nullptr;
        }
        return std::make_unique<K>(alpha_, beta_, a.dataType, a.elementsSize());
    }

    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t { return typeId(); }
    auto K::description() const noexcept -> std::string_view {
        return "Performing HardSigmoid using CPU";
    }

    template<class T>
    static Routine lowerTyped(float alpha_, float beta_, size_t size) {
        using namespace runtime;

        return [=](Resources &, void *workspace, void const *const *inputs, void *const *outputs) {
            auto x = reinterpret_cast<T const *>(inputs[0]);
            auto y = reinterpret_cast<T *>(outputs[0]);
            std::for_each_n(std::execution::par_unseq,
                            natural_t(0), size,
                            [&](auto i) {
                                y[i] = std::clamp(alpha_ * x[i] + beta_, static_cast<T>(0), static_cast<T>(1));
                            });
        };
    }

    auto K::lower(Resources &) const noexcept -> RoutineWorkspace {
        switch (dataType) {
            case DT::F32:
                return lowerTyped<float>(alpha, beta, size);
            case DT::F64:
                return lowerTyped<double>(alpha, beta, size);
            default:
                UNREACHABLE();
        }
    }
}// namespace refactor::kernel

