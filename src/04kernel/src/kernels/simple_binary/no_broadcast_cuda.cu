#include "no_broadcast_cuda.hh"

namespace refactor::kernel {
    using K = Binary11Cuda;
    using Op = SimpleBinaryType;
    using DT = DataType;

#define KERNEL(NAME, OP)                                                                  \
    template<typename T>                                                                  \
    __global__ void __binary11_kernel_##NAME(T const *a, T const *b, T *c, size_t size) { \
        size_t tid = threadIdx.x + blockIdx.x * blockDim.x;                               \
        if (tid < size) {                                                                 \
            c[tid] = (OP);                                                                \
        }                                                                                 \
    }

    // clang-format off
    KERNEL(Add,     a[tid] +  b[tid])
    KERNEL(Sub,     a[tid] -  b[tid])
    KERNEL(Mul,     a[tid] *  b[tid])
    KERNEL(Div,     a[tid] /  b[tid])
    KERNEL(Pow, pow(a[tid] ,  b[tid]))
    KERNEL(And,     a[tid] && b[tid])
    KERNEL(Or ,     a[tid] || b[tid])
    KERNEL(Xor,     a[tid] ^  b[tid])
    // clang-format on

#define CASE_DT(NAME, T)                                                                     \
    case DT::T:                                                                              \
        return [n = this->size](runtime::Resources &, void *workspace, void const *const *inputs, void *const *outputs) { \
            using T_ = primitive<DT::T>::type;                                               \
            auto a = reinterpret_cast<T_ const *>(inputs[0]);                                \
            auto b = reinterpret_cast<T_ const *>(inputs[1]);                                \
            auto c = reinterpret_cast<T_ *>(outputs[0]);                                     \
            size_t blocksize = 1024;                                                         \
            size_t gridsize = (n + blocksize - 1) / blocksize;                               \
            __binary11_kernel_##NAME<<<gridsize, blocksize>>>(a, b, c, n);                   \
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
