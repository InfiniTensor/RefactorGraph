#ifndef REFACTOR_DATA_TYPE
#define REFACTOR_DATA_TYPE

#include "bf16_t.h"
#include "fp16_t.h"
#include <complex>
#include <cstdint>
#include <optional>

namespace refactor::common {

    struct DataType {
        /// @brief 数据类型。
        /// @see <https://onnx.ai/onnx/api/mapping.html#l-onnx-types-mapping>
        enum : uint8_t {
            F32 = 1,// float·
            U8 = 2, // uint8_t
            I8 = 3, // int8_t
            U16 = 4,// uint16_t
            I16 = 5,// int16_t
            I32 = 6,// int32_t
            I64 = 7,// int64_t
            // String = 8,
            Bool = 9,       // bool
            FP16 = 10,      // fp16_t
            F64 = 11,       // double
            U32 = 12,       // uint32_t
            U64 = 13,       // uint64_t
            Complex64 = 14, // std::complex<float>
            Complex128 = 15,// std::complex<double>
            BF16 = 16,      // bf16_t
        } internal;

        constexpr DataType(decltype(internal) i) : internal(i) {}

        static std::optional<DataType> parse(uint8_t);

        bool operator==(DataType const &) const;
        bool operator!=(DataType const &) const;
        bool operator<(DataType const &) const;
        bool operator>(DataType const &) const;
        bool operator<=(DataType const &) const;
        bool operator>=(DataType const &) const;

        std::string_view name() const;
        bool isIeee754() const;
        bool isFloat() const;
        bool isSigned() const;
        bool isNumberic() const;
        bool isBool() const;
        size_t size() const;
    };

    template<class T>
    DataType dataType();

    template<> inline DataType dataType<float>() { return DataType::F32; }
    template<> inline DataType dataType<uint8_t>() { return DataType::U8; }
    template<> inline DataType dataType<int8_t>() { return DataType::I8; }
    template<> inline DataType dataType<uint16_t>() { return DataType::U16; }
    template<> inline DataType dataType<int16_t>() { return DataType::I16; }
    template<> inline DataType dataType<int32_t>() { return DataType::I32; }
    template<> inline DataType dataType<int64_t>() { return DataType::I64; }
    template<> inline DataType dataType<bool>() { return DataType::Bool; }
    template<> inline DataType dataType<fp16_t>() { return DataType::FP16; }
    template<> inline DataType dataType<double>() { return DataType::F64; }
    template<> inline DataType dataType<uint32_t>() { return DataType::U32; }
    template<> inline DataType dataType<uint64_t>() { return DataType::U64; }
    template<> inline DataType dataType<bf16_t>() { return DataType::BF16; }

    template<decltype(DataType::internal) t>
    struct primitive_t;

    template<>
    struct primitive_t<DataType::F32> {
        using type = float;
    };
    template<>
    struct primitive_t<DataType::U8> {
        using type = uint8_t;
    };
    template<>
    struct primitive_t<DataType::I8> {
        using type = int8_t;
    };
    template<>
    struct primitive_t<DataType::U16> {
        using type = uint16_t;
    };
    template<>
    struct primitive_t<DataType::I16> {
        using type = int16_t;
    };
    template<>
    struct primitive_t<DataType::I32> {
        using type = int32_t;
    };
    template<>
    struct primitive_t<DataType::I64> {
        using type = int64_t;
    };
    template<>
    struct primitive_t<DataType::Bool> {
        using type = bool;
    };
    template<>
    struct primitive_t<DataType::FP16> {
        using type = fp16_t;
    };
    template<>
    struct primitive_t<DataType::F64> {
        using type = double;
    };
    template<>
    struct primitive_t<DataType::U32> {
        using type = uint32_t;
    };
    template<>
    struct primitive_t<DataType::U64> {
        using type = uint64_t;
    };
    template<>
    struct primitive_t<DataType::Complex64> {
        using type = std::complex<float>;
    };
    template<>
    struct primitive_t<DataType::Complex128> {
        using type = std::complex<double>;
    };
    template<>
    struct primitive_t<DataType::BF16> {
        using type = bf16_t;
    };

}// namespace refactor::common

#endif// REFACTOR_DATA_TYPE
