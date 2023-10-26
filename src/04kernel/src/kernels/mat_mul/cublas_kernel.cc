#include "cublas_kernel.hh"
#include <unordered_set>

namespace refactor::kernel {
    using K = MatMulCublas;
    using DT = DataType;

    K::MatMulCublas(decltype(info) info_) noexcept
        : Kernel(), info(std::move(info_)) {}

    auto K::build(bool transA, bool transB,
                  float alpha, float beta,
                  Tensor const &a,
                  Tensor const &b,
                  std::optional<std::reference_wrapper<Tensor const>> c) noexcept -> KernelBox {
        static const std::unordered_set<decltype(DT::internal)> TYPE{
            DT::F32, DT::F64, DT::FP16};

#ifndef USE_CUDA
        return nullptr;
#endif

        auto dataType = a.dataType;
        if (a.rank() != 2 ||
            b.rank() != 2 ||
            TYPE.find(dataType) == TYPE.end() ||
            a.dataType != b.dataType) {
            return nullptr;
        }
        return std::make_unique<K>(decltype(info){
            dataType,
            transA,
            transB,
            c.has_value(),
            static_cast<int>(a.shape[0]),
            static_cast<int>(b.shape[1]),
            static_cast<int>(a.shape[1]),
            alpha * beta,
        });
    }

    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t { return typeId(); }
    auto K::description() const noexcept -> std::string_view {
        return "Performing mat mul using CUBLAS";
    }

}// namespace refactor::kernel
