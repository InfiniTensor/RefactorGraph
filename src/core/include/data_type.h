#ifndef CORE_H
#define CORE_H

#include <cstdint>
#include <optional>

namespace proj_namespace::core {

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
        FP16 = 10,// fp16
        F64 = 11, // f64
        U32 = 12, // u32
        U64 = 13, // u64
    };

    /// @brief 从数值解析数据类型。
    /// @param param1 数据类型数值。
    /// @return 当 `param1` 是合法的数据类型值，`std::optional` 非空。
    std::optional<DataType> parse_data_type(uint8_t);

    void try_sub_project();

}// namespace proj_namespace::core

#endif// CORE_H
