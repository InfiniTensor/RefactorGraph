#include "cublas_kernel.hh"
#include <unordered_set>

namespace refactor::kernel {
    using K = MatMulCublas;
    using DT = DataType;

    K::MatMulCublas(decltype(info) info_) noexcept
        : Kernel(), info(std::move(info_)) {}

    auto K::build(Tensor const &a, Tensor const &b, Tensor const &y, MatMulInfo info) noexcept -> KernelBox {
        static const std::unordered_set<decltype(DT::internal)> TYPE{
            DT::F32, DT::F64, DT::FP16};
#ifndef USE_CUDA
        return nullptr;
#endif
        auto dataType = info.dataType;
        if (dataType != a.dataType ||
            dataType != b.dataType ||
            dataType != y.dataType ||
            TYPE.find(dataType) == TYPE.end()) {
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
        return "Performing MatMul using CUBLAS";
    }

}// namespace refactor::kernel
