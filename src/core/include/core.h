#ifndef CORE_H
#define CORE_H

#include <cstdint>
#include <optional>

namespace proj_namespace::core {

    enum class DataType : uint8_t {
        F32 = 1,
        U8 = 2,
        I8 = 3,
        U16 = 4,
        I16 = 5,
        I32 = 6,
        I64 = 7,
        // String = 8,
        Bool = 9,
        F16 = 10,
        F64 = 11,
        U32 = 12,
        U64 = 13,
    };

    std::optional<DataType> parse_data_type(uint8_t);

    void try_sub_project();

}// namespace proj_namespace::core

#endif// CORE_H
