#include "cublas_kernel.hh"

namespace refactor::kernel {
    using K = MatMulCublas;
    using DT = DataType;

    K::MatMulCublas(decltype(info) info_, decltype(biasExpand) biasExpand_) noexcept
        : Kernel(), info(std::move(info_)), biasExpand(std::move(biasExpand_)) {}

    auto K::build(Tensor const &a, Tensor const &b, Tensor const &y, MatMulInfo info) noexcept -> KernelBox {
        static const std::unordered_set<decltype(DT::internal)> TYPE{DT::F32, DT::F64, DT::FP16};
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

        dim_t inputs[2];
        switch (info.biasType) {
            case BiasType::NoBias:
                return std::make_unique<K>(std::move(info), std::nullopt);
            case BiasType::Scalar:
                inputs[0] = 1;
                inputs[1] = 1;
                break;
            case BiasType::RowVector:
                inputs[0] = 1;
                inputs[1] = info.n;
                break;
            case BiasType::ColVector:
                inputs[0] = info.m;
                inputs[1] = 1;
                break;
            case BiasType::Matrix:
                inputs[0] = info.m;
                inputs[1] = info.n;
                break;
            default:
                break;
        }

        std::vector<dim_t> outputShape(std::max(a.rank(), b.rank()));
        for (auto i : range0_(outputShape.size() - 2)) {
            auto a_ = i < a.rank() ? a.shape[i] : 1;
            auto b_ = i < b.rank() ? b.shape[i] : 1;
            outputShape[i] = std::max(a_, b_);
        }
        outputShape.rbegin()[1] = info.m;
        outputShape.rbegin()[0] = info.n;

        return std::make_unique<K>(
            std::move(info),
            std::make_optional(ExpandInfo(
                dataType,
                slice(inputs, 2),
                slice(outputShape.data(), outputShape.size()))));
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
