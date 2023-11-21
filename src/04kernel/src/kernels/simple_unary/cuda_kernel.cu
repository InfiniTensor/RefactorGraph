#include "cuda_kernel.hh"
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>

namespace refactor::kernel {
    using K = SimpleUnaryCuda;
    using Op = SimpleUnaryType;
    using DT = DataType;

    template<class T> struct AbsFunctor {
        __device__ T operator()(T x) const { return abs(x); }
    };
    template<class T> struct NegFunctor {
        __device__ T operator()(T x) const { return -x; }
    };
    template<class T> struct ReluFunctor {
        __device__ T operator()(T x) const { return x > 0 ? x : 0; }
    };
    template<class T> struct SqrtFunctor {
        __device__ T operator()(T x) const {
            using M = std::conditional_t<sizeof(T) <= 4, float, double>;
            return static_cast<T>(sqrt(static_cast<M>(x)));
        }
    };
    template<class T> struct SigmoidFunctor {
        __device__ T operator()(T x) const {
            using M = std::conditional_t<sizeof(T) <= 4, float, double>;
            return static_cast<T>(1 / (1 + std::exp(-static_cast<M>(x))));
        }
    };
    template<class T> struct TanhFunctor {
        __device__ T operator()(T x) const {
            using M = std::conditional_t<sizeof(T) <= 4, float, double>;
            return static_cast<T>(tanh(static_cast<M>(x)));
        }
    };

    template<class T, class UnaryFuntor>
    auto lowerTyped(size_t n) noexcept {
        using namespace runtime;

        return [n](Resources &, void *workspace, void const *const *inputs, void *const *outputs) {
            auto const *x = static_cast<T const *>(inputs[0]);
            auto *y = static_cast<T *>(outputs[0]);
            thrust::transform(thrust::device, x, x + n, y, UnaryFuntor{});
        };
    }

#define CASE(FUNC, TYPE) \
    case DT::TYPE:       \
        return lowerTyped<primitive<DT::TYPE>::type, FUNC##Functor<primitive<DT::TYPE>::type>>(size)
#define COPY                                                                                   \
    return [size = size * dataType.size()](Resources &, void *workspace, void const *const *inputs, void *const *outputs) { \
        cudaMemcpyAsync(outputs[0], inputs[0], size, cudaMemcpyDeviceToDevice);                \
    }
#define GROUP_F(FUNC) \
    CASE(FUNC, F32);  \
    CASE(FUNC, F64)
#define GROUP_I(FUNC) \
    CASE(FUNC, I8);   \
    CASE(FUNC, I16);  \
    CASE(FUNC, I32);  \
    CASE(FUNC, I64)
#define GROUP_U(FUNC) \
    CASE(FUNC, U8);   \
    CASE(FUNC, U16);  \
    CASE(FUNC, U32);  \
    CASE(FUNC, U64)

    auto K::lower(Resources &) const noexcept -> RoutineWorkspace {
        switch (opType) {
            case Op::Abs:
                switch (dataType) {
                    GROUP_F(Abs);
                    GROUP_I(Abs);
                    case DT::U8:
                    case DT::U16:
                    case DT::U32:
                    case DT::U64:
                        COPY;
                    default:
                        UNREACHABLE();
                }
            case Op::Relu:
                switch (dataType) {
                    GROUP_F(Relu);
                    GROUP_I(Relu);
                    case DT::U8:
                    case DT::U16:
                    case DT::U32:
                    case DT::U64:
                        COPY;
                    default:
                        UNREACHABLE();
                }
            case Op::Sqrt:
                switch (dataType) {
                    GROUP_F(Sqrt);
                    GROUP_I(Sqrt);
                    GROUP_U(Sqrt);
                    default:
                        UNREACHABLE();
                }
            case Op::Sigmoid:
                switch (dataType) {
                    GROUP_F(Sigmoid);
                    GROUP_I(Sigmoid);
                    GROUP_U(Sigmoid);
                    default:
                        UNREACHABLE();
                }
            case Op::Tanh:
                switch (dataType) {
                    GROUP_F(Tanh);
                    GROUP_I(Tanh);
                    GROUP_U(Tanh);
                    default:
                        UNREACHABLE();
                }
            case Op::Neg:
                switch (dataType) {
                    GROUP_F(Neg);
                    GROUP_I(Neg);
                    default:
                        UNREACHABLE();
                }
            default:
                UNREACHABLE();
        }
    }

}// namespace refactor::kernel
