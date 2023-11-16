#include "basic_cuda.hh"
#include <thrust/device_vector.h>

namespace refactor::kernel {
    using K = BinaryBasicCuda;
    using Op = SimpleBinaryType;
    using DT = DataType;

#define KERNEL(NAME, OP)                                    \
    template<typename T>                                    \
    __global__ void __binary_basic_kernel_##NAME(           \
        T const *a, T const *b, T *c,                       \
        dim_t const *strides, dim_t rank,                   \
        size_t size) {                                      \
                                                            \
        size_t tid = threadIdx.x + blockIdx.x * blockDim.x; \
        if (tid >= size) { return; }                        \
        dim_t ia = 0, ib = 0;                               \
        auto rem = tid;                                     \
        for (auto i = 0; i < rank; ++i) {                   \
            auto dim = strides + 3 * i;                     \
            auto quot = rem / dim[2];                       \
            ia += dim[0] * quot;                            \
            ib += dim[1] * quot;                            \
            rem %= dim[2];                                  \
        }                                                   \
        c[tid] = (OP);                                      \
    }

    // clang-format off
    KERNEL(Add,     a[ia] +  b[ib])
    KERNEL(Sub,     a[ia] -  b[ib])
    KERNEL(Mul,     a[ia] *  b[ib])
    KERNEL(Div,     a[ia] /  b[ib])
    KERNEL(Pow, pow(a[ia] ,  b[ib]))
    KERNEL(And,     a[ia] && b[ib])
    KERNEL(Or ,     a[ia] || b[ib])
    KERNEL(Xor,     a[ia] ^  b[ib])
    // clang-format on

#define CASE_DT(NAME, T)                                                                                        \
    case DT::T:                                                                                                 \
        return [strides = thrust::device_vector<dim_t>(broadcaster.strides.begin(), broadcaster.strides.end()), \
                rank = broadcaster.inputsCount,                                                                 \
                n = broadcaster.outputsCount](runtime::Resources &, void const **inputs, void **outputs) {      \
            using T_ = primitive<DT::T>::type;                                                                  \
            auto a = reinterpret_cast<T_ const *>(inputs[0]);                                                   \
            auto b = reinterpret_cast<T_ const *>(inputs[1]);                                                   \
            auto c = reinterpret_cast<T_ *>(outputs[0]);                                                        \
            size_t blocksize = 1024;                                                                            \
            size_t gridsize = (n + blocksize - 1) / blocksize;                                                  \
            __binary_basic_kernel_##NAME<<<gridsize, blocksize>>>(a, b, c, strides.data().get(), rank, n);      \
        }

#define CASE_OP(NAME)                \
    case Op::NAME:                   \
        switch (dataType.internal) { \
            CASE_DT(NAME, F32);      \
            CASE_DT(NAME, U8);       \
            CASE_DT(NAME, I8);       \
            CASE_DT(NAME, U16);      \
            CASE_DT(NAME, I16);      \
            CASE_DT(NAME, I32);      \
            CASE_DT(NAME, I64);      \
            CASE_DT(NAME, F64);      \
            CASE_DT(NAME, U32);      \
            CASE_DT(NAME, U64);      \
            default:                 \
                UNREACHABLE();       \
        }

    Routine K::lower(Resources &) const noexcept {
        switch (opType) {
            CASE_OP(Add)
            CASE_OP(Sub)
            CASE_OP(Mul)
            CASE_OP(Div)
            case Op::And:
                switch (dataType.internal) {
                    CASE_DT(And, Bool);
                    default:
                        UNREACHABLE();
                }
            case Op::Or:
                switch (dataType.internal) {
                    CASE_DT(Or, Bool);
                    default:
                        UNREACHABLE();
                }
            case Op::Xor:
                switch (dataType.internal) {
                    CASE_DT(Xor, Bool);
                    default:
                        UNREACHABLE();
                }
            case Op::Pow: {
                switch (dataType.internal) {
                    CASE_DT(Pow, F32);
                    CASE_DT(Pow, F64);
                    CASE_DT(Pow, I8);
                    CASE_DT(Pow, I16);
                    CASE_DT(Pow, I32);
                    CASE_DT(Pow, I64);
                    default:
                        UNREACHABLE();
                }
            }
            default:
                UNREACHABLE();
        }
    }
}// namespace refactor::kernel
