#ifndef CORE_H
#define CORE_H

#include "bf16_t.h"
#include "fp16_t.h"
#include <cstdint>
#include <optional>

namespace refactor::common {

    /// @brief 数据类型。
    enum class DataType : uint8_t {
        F32 = 1,// float
        U8 = 2, // uint8_t
        I8 = 3, // int8_t
        U16 = 4,// uint16_t
        I16 = 5,// int16_t
        I32 = 6,// int32_t
        I64 = 7,// int64_t
        // String = 8,
        Bool = 9, // bool
        FP16 = 10,// fp16_t
        F64 = 11, // double
        U32 = 12, // uint32_t
        U64 = 13, // uint64_t
        BF16 = 14,// bf16_t
    };

    /// @brief 从数值解析数据类型。
    /// @param param1 数据类型数值。
    /// @return 当 `param1` 是合法的数据类型值，`std::optional` 非空。
    std::optional<DataType> parseDataType(uint8_t);

    /// @brief 判断是否符合 IEE754 的浮点数数据类型。
    bool isIeee754DataType(DataType);

    /// @brief 判断是否浮点数数据类型。
    bool isFloatDataType(DataType);

    /// @brief 判断是否数字数据类型。
    bool isNumbericDataType(DataType);

    template<DataType t>
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
    struct primitive_t<DataType::BF16> {
        using type = bf16_t;
    };

}// namespace refactor::common

#endif// CORE_H
