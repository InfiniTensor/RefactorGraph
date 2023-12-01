#include "cuda_kernel.hh"
#include <thrust/execution_policy.h>
#include <thrust/transform.h>

namespace refactor::kernel {
    using K = ClipCuda;

#define MAX(T, V)                     \
    template<> struct Max<T> {        \
        constexpr static T value = V; \
    }

    template<class T> struct Max {};
    MAX(float, FLT_MAX);
    MAX(uint8_t, UCHAR_MAX);
    MAX(int8_t, CHAR_MAX);
    MAX(uint16_t, USHRT_MAX);
    MAX(int16_t, SHRT_MAX);
    MAX(int32_t, INT_MAX);
    MAX(int64_t, LLONG_MAX);
    // see <https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0192r0.pdf>
    // how to define a constexpr half?
    // MAX(half, static_cast<half>(65504));
    MAX(double, DBL_MAX);
    MAX(uint32_t, UINT_MAX);
    MAX(uint64_t, ULLONG_MAX);
#undef MAX

    template<class T>
    struct ClipFunctor {
        T const *min, *max;

        __device__ T operator()(T x) const noexcept {
            T min_ = *min, max_ = max ? *max : Max<T>::value;
            return x < min_ ? min_ : x > max_ ? max_
                                              : x;
        }
    };

    template<class T>
    static auto lowerTyped(size_t size, bool hasMax) noexcept -> RoutineWorkspace {
        using namespace runtime;
        return [=](Resources &, void *workspace, void const *const *inputs, void *const *outputs) {
            auto data = reinterpret_cast<T const *>(inputs[0]);
            auto min = reinterpret_cast<T const *>(inputs[1]),
                 max = hasMax
                           ? reinterpret_cast<T const *>(inputs[2])
                           : nullptr;
            auto output = reinterpret_cast<T *>(outputs[0]);

            thrust::transform(thrust::device,
                              data, data + size,
                              output,
                              ClipFunctor<T>{min, max});
        };
    }

    auto K::lower(Resources &) const noexcept -> RoutineWorkspace {
#define CASE(DT)       \
    case DataType::DT: \
        return lowerTyped<primitive<DataType::DT>::type>(size, hasMax)

        switch (dataType) {
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
            default:
                UNREACHABLE();
        }
    }

}// namespace refactor::kernel
