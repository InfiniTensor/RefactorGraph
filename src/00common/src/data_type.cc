#include "data_type.h"
#include <fmt/core.h>
#include <unordered_set>

namespace refactor::common {
    std::optional<DataType> parseDataType(uint8_t value) {
        switch (value) {
            case 1:
            case 2:
            case 3:
            case 4:
            case 5:
            case 6:
            case 7:
            case 9:
            case 10:
            case 11:
            case 12:
            case 13:
                return {(DataType) value};
            default:
                return {};
        }
    }

    bool isIeee754DataType(DataType dt) {
        static const std::unordered_set<DataType> set{DataType::F32, DataType::FP16, DataType::F64};
        return set.find(dt) != set.end();
    }

    bool isFloatDataType(DataType dt) {
        static const std::unordered_set<DataType> set{DataType::F32, DataType::FP16, DataType::F64, DataType::BF16};
        return set.find(dt) != set.end();
    }

    bool isNumbericDataType(DataType dt) {
        static const std::unordered_set<DataType> set{
            DataType::F32, DataType::U8, DataType::I8, DataType::U16, DataType::I16, DataType::I32,
            DataType::I64, DataType::FP16, DataType::F64, DataType::U32, DataType::U64, DataType::BF16};
        return set.find(dt) != set.end();
    }
}// namespace refactor::common
