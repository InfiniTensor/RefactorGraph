#include "cpu_kernel.hh"
#include <execution>

namespace refactor::kernel {
    using K = CastCpu;

    K::CastCpu(decltype(from) from_,
               decltype(to) to_,
               decltype(size) size_) noexcept
        : from(from_), to(to_), size(size_) {}

    auto K::build(Tensor const &from, Tensor const &to) noexcept -> KernelBox {
        return std::make_unique<K>(from.dataType, to.dataType, from.elementsSize());
    }
    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t {
        return typeId();
    }
    auto K::description() const noexcept -> std::string_view {
        return "Performing cast operation on generic cpu";
    }

    template<class T, class U>
    static auto lowerTyped(size_t size) noexcept -> RoutineWorkspace {
        using namespace runtime;
        return [=](Resources &, void *, void const *const *inputs, void *const *outputs) {
            auto x = reinterpret_cast<T const *>(inputs[0]);
            auto y = reinterpret_cast<U *>(outputs[0]);
            std::transform(std::execution::par_unseq,
                           x, x + size,
                           y,
                           [=](auto a) { return static_cast<U>(a); });
        };
    }

    auto K::lower(Resources &) const noexcept -> RoutineWorkspace {
#define CASE_U(T, U)  \
    case DataType::U: \
        return lowerTyped<T, primitive<DataType::U>::type>(size)

#define CASE_T(T)                                       \
    case DataType::T:                                   \
        switch (to) {                                   \
            CASE_U(primitive<DataType::T>::type, U8);   \
            CASE_U(primitive<DataType::T>::type, U16);  \
            CASE_U(primitive<DataType::T>::type, U32);  \
            CASE_U(primitive<DataType::T>::type, U64);  \
            CASE_U(primitive<DataType::T>::type, I8);   \
            CASE_U(primitive<DataType::T>::type, I16);  \
            CASE_U(primitive<DataType::T>::type, I32);  \
            CASE_U(primitive<DataType::T>::type, I64);  \
            CASE_U(primitive<DataType::T>::type, F32);  \
            CASE_U(primitive<DataType::T>::type, F64);  \
            CASE_U(primitive<DataType::T>::type, Bool); \
            default:                                    \
                UNREACHABLE();                          \
        }
        switch (from) {
            CASE_T(U8);
            CASE_T(U16);
            CASE_T(U32);
            CASE_T(U64);
            CASE_T(I8);
            CASE_T(I16);
            CASE_T(I32);
            CASE_T(I64);
            CASE_T(F32);
            CASE_T(F64);
            CASE_T(Bool);
            default:
                UNREACHABLE();
        }
    }// namespace refactor::kernel

}// namespace refactor::kernel
