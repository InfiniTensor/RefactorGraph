#ifndef REFACTOR_DATA_TYPE
#define REFACTOR_DATA_TYPE

#include "bf16_t.h"
#include "fp16_t.h"
#include <complex>
#include <optional>

namespace refactor {

    using fp32_t = float;
    using fp64_t = double;

    struct DataType {
        /// @brief 数据类型。
        /// @see <https://onnx.ai/onnx/api/mapping.html#l-onnx-types-mapping>
        enum : uint8_t {
            // clang-format off
            F32        =  1,
            U8         =  2,
            I8         =  3,
            U16        =  4,
            I16        =  5,
            I32        =  6,
            I64        =  7,
            String     =  8,
            Bool       =  9,
            FP16       = 10,
            F64        = 11,
            U32        = 12,
            U64        = 13,
            Complex64  = 14,
            Complex128 = 15,
            BF16       = 16,
            // clang-format on
        } internal;

        constexpr DataType(decltype(internal) i) noexcept : internal(i) {}
        constexpr operator decltype(internal)() const noexcept { return internal; }

        static std::optional<DataType> parse(uint8_t) noexcept;

        std::string_view name() const noexcept;
        bool isIeee754() const noexcept;
        bool isFloat() const noexcept;
        bool isSignedLarge() const noexcept;
        bool isSigned() const noexcept;
        bool isUnsigned() const noexcept;
        bool isNumberic() const noexcept;
        bool isCpuNumberic() const noexcept;
        bool isBool() const noexcept;
        size_t size() const noexcept;
    };

    constexpr bool operator==(DataType const &lhs, DataType const &rhs) noexcept {
        return lhs.internal == rhs.internal;
    }
    constexpr bool operator!=(DataType const &lhs, DataType const &rhs) noexcept {
        return !operator==(lhs, rhs);
    }
    constexpr bool operator==(DataType const &lhs, decltype(DataType::internal) const &rhs) noexcept {
        return lhs.internal == rhs;
    }
    constexpr bool operator!=(DataType const &lhs, decltype(DataType::internal) const &rhs) noexcept {
        return !operator==(lhs, rhs);
    }
    constexpr bool operator==(decltype(DataType::internal) const &lhs, DataType const &rhs) noexcept {
        return lhs == rhs.internal;
    }
    constexpr bool operator!=(decltype(DataType::internal) const &lhs, DataType const &rhs) noexcept {
        return !operator==(lhs, rhs);
    }

    template<class T>
    DataType dataType();

    template<decltype(DataType::internal) t>
    struct primitive;

#define PRIMITIVE(T, U)                                              \
    template<> inline DataType dataType<U>() { return DataType::T; } \
    template<> struct primitive<DataType::T> {                       \
        using type = U;                                              \
    }
    // clang-format off
    PRIMITIVE(F32        , fp32_t              );
    PRIMITIVE(U8         , uint8_t             );
    PRIMITIVE(I8         , int8_t              );
    PRIMITIVE(U16        , uint16_t            );
    PRIMITIVE(I16        , int16_t             );
    PRIMITIVE(I32        , int32_t             );
    PRIMITIVE(I64        , int64_t             );
    PRIMITIVE(String     , std::string         );
    PRIMITIVE(Bool       , bool                );
    PRIMITIVE(FP16       , fp16_t              );
    PRIMITIVE(F64        , fp64_t              );
    PRIMITIVE(U32        , uint32_t            );
    PRIMITIVE(U64        , uint64_t            );
    PRIMITIVE(Complex64  , std::complex<fp32_t>);
    PRIMITIVE(Complex128 , std::complex<fp64_t>);
    PRIMITIVE(BF16       , bf16_t              );
    // clang-format on
}// namespace refactor

#endif// REFACTOR_DATA_TYPE
