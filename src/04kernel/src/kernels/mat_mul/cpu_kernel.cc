#include "cpu_kernel.hh"
#include "../expand/cpu_kernel.hh"
#include "../mat_mul_common/cpu_template.hpp"

namespace refactor::kernel {
    using K = MatMulCPU;
    using DT = DataType;

    K::MatMulCPU(decltype(info) info_) noexcept
        : Kernel(), info(std::move(info_)) {}

    auto K::build(decltype(info) info) noexcept -> KernelBox {
        return info.dataType.isCpuNumberic()
                   ? std::make_unique<K>(std::move(info))
                   : nullptr;
    }

    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t { return typeId(); }
    auto K::description() const noexcept -> std::string_view {
        return "Performing MatMul using CPU";
    }

    template<class T>
    static auto lowerTyped(MatMulInfo const &info, Resources &res) noexcept -> RoutineWorkspace {
        MatMulCPUMetaData const md{
            .M = info.m,
            .K = info.k,
            .N = info.n,
            .strideA0 = info.transA ? 1 : info.k,
            .strideA1 = info.transA ? info.m : 1,
            .strideB0 = info.transB ? 1 : info.n,
            .strideB1 = info.transB ? info.k : 1,
            .alpha = static_cast<T>(info.alpha),
            .beta = static_cast<T>(info.biasExpand ? info.beta : 0.0f),
        };

        auto stepY = info.m * info.n,
             stepA = info.m * info.k,
             stepB = info.k * info.n;
        auto biasEx = info.biasExpand
                          ? std::make_optional(ExpandCpu(*info.biasExpand).lower(res).routine)
                          : std::nullopt;

        if (info.broadcaster.needBroadcast()) {
            return [broadcaster = info.broadcaster,
                    stepY, stepA, stepB,
                    md, biasEx]//
                (runtime::Resources & res, void *workspace, void const *const *inputs, void *const *outputs) {
                    if (biasEx) { (*biasEx)(res, workspace, inputs + 2, outputs); }

                    auto a = reinterpret_cast<T const *>(inputs[0]);
                    auto b = reinterpret_cast<T const *>(inputs[1]);
                    auto y = reinterpret_cast<T *>(outputs[0]);
                    dim_t offset[2];
                    for (auto i : range0_(broadcaster.outputsCount)) {
                        broadcaster.locate(i, offset);
                        md.matrixMultiply(a + stepA * offset[0], b + stepB * offset[1], y + stepY * i);
                    }
                };
        } else {
            return [batch = info.broadcaster.outputsCount,
                    stepY, stepA, stepB,
                    md, biasEx]//
                (runtime::Resources & res, void *workspace, void const *const *inputs, void *const *outputs) {
                    if (biasEx) { (*biasEx)(res, workspace, inputs + 2, outputs); }

                    auto a = reinterpret_cast<T const *>(inputs[0]);
                    auto b = reinterpret_cast<T const *>(inputs[1]);
                    auto y = reinterpret_cast<T *>(outputs[0]);
                    for (auto i : range0_(batch)) {
                        md.matrixMultiply(a + stepA * i, b + stepB * i, y + stepY * i);
                    }
                };
        }
    }

    auto K::lower(Resources &res) const noexcept -> RoutineWorkspace {
#define CASE(T)       \
    case DataType::T: \
        return lowerTyped<primitive<DataType::T>::type>(info, res);

        switch (info.dataType) {
            CASE(F32);
            CASE(F64);

            CASE(U8);
            CASE(U16);
            CASE(U32);
            CASE(U64);

            CASE(I8);
            CASE(I16);
            CASE(I32);
            CASE(I64);
            default:
                UNREACHABLE();
        }
    };

}// namespace refactor::kernel
