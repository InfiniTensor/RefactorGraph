#include "cpu_kernel.hh"
#include "../expand/cpu_kernel.hh"

namespace refactor::kernel {
    using K = MatMulCPU;
    using DT = DataType;

    K::MatMulCPU(decltype(info) info_) noexcept
        : Kernel(), info(std::move(info_)) {}

    auto K::build(MatMulInfo info) noexcept -> KernelBox {
        if (!info.dataType.isCpuNumberic()) {
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
        return "Performing MatMul using CPU";
    }

    struct MatMulCPUMetaData {
        size_t M, K, N;
        size_t strideA0, strideA1, strideB0, strideB1;
    };


    /*
    * 2D matrix multiplication: Y = a * A @ B + b * Y
    * Assume bias C has been broadcast to Y already. Beta should be 0 in the absence of bias.
    */
    template<typename T>
    void matrixMultiply(T const *A, T const *B, T *Y,
                        T const alpha, T const beta,
                        const MatMulCPUMetaData md) {
#pragma omp parallel for
        for (size_t i = 0; i < md.M; i++) {
            for (size_t j = 0; j < md.N; j++) {
                T sum = 0;
#pragma omp simd reduction(+ : sum)
                for (size_t k = 0; k < md.K; k++) {
                    sum += A[i * md.strideA0 + k * md.strideA1] * B[k * md.strideB0 + j * md.strideB1];
                }
                Y[i * md.N + j] = beta * Y[i * md.N + j] + alpha * sum;
            }
        }
    }


#define CASE(T)                                                                                                                           \
    case DT::T: {                                                                                                                         \
        using T_ = primitive<DT::T>::type;                                                                                                \
        return [alpha = static_cast<T_>(info.alpha),                                                                                      \
                beta = static_cast<T_>(info.biasExpand ? info.beta : 0.0f),                                                               \
                broadcaster = info.broadcaster,                                                                                           \
                md,                                                                                                                       \
                stepY = info.m * info.n,                                                                                                  \
                stepA = info.m * info.k,                                                                                                  \
                stepB = info.k * info.n,                                                                                                  \
                biasEx = info.biasExpand                                                                                                  \
                             ? std::make_optional(ExpandCpu(*info.biasExpand).lower(res).routine)                                         \
                             : std::nullopt](runtime::Resources &res, void *workspace, void const *const *inputs, void *const *outputs) { \
            if (biasEx) { (*biasEx)(res, workspace, inputs + 2, outputs); }                                                               \
            auto A = reinterpret_cast<T_ const *>(inputs[0]);                                                                             \
            auto B = reinterpret_cast<T_ const *>(inputs[1]);                                                                             \
            auto Y = reinterpret_cast<T_ *>(outputs[0]);                                                                                  \
            dim_t offset[2];                                                                                                              \
            for (size_t i = 0; i < broadcaster.outputsCount; i++) {                                                                       \
                broadcaster.locate(i, offset);                                                                                            \
                matrixMultiply(A + stepA * offset[0], B + stepB * offset[1], Y + stepY * i, alpha, beta, md);                             \
            }                                                                                                                             \
        };                                                                                                                                \
    }


    auto K::lower(Resources &res) const noexcept -> RoutineWorkspace {
        MatMulCPUMetaData md;
        md.M = info.m, md.K = info.k, md.N = info.n;
        md.strideA0 = info.transA ? 1 : info.k;
        md.strideA1 = info.transA ? info.m : 1;
        md.strideB0 = info.transB ? 1 : info.n;
        md.strideB1 = info.transB ? info.k : 1;

        switch (info.dataType) {
            CASE(F32);
            CASE(U8);
            CASE(I8);
            CASE(U16);
            CASE(I16);
            CASE(I32);
            CASE(I64);
            CASE(F64);
            CASE(U32);
            CASE(U64);
        }
    };

}// namespace refactor::kernel
