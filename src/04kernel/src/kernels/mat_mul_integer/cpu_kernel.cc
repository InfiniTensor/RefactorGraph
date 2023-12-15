#include "cpu_kernel.hh"
#include "../mat_mul_common/cpu_template.hpp"

namespace refactor::kernel {
    using K = MatMulIntegerCPU;
    using DT = DataType;

    K::MatMulIntegerCPU(decltype(info) info_) noexcept
        : Kernel(), info(std::move(info_)) {}

    auto K::build(decltype(info) info) noexcept -> KernelBox {
        return std::make_unique<K>(std::move(info));
    }

    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t { return typeId(); }
    auto K::description() const noexcept -> std::string_view {
        return "Performing MatMulInteger using CPU";
    }

    auto K::lower(Resources &res) const -> RoutineWorkspace {
        TODO("");
    };

}// namespace refactor::kernel
