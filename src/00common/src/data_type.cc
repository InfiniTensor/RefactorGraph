#include "common/data_type.h"
#include "common/error_handler.h"
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

    bool isSignedDataType(DataType dt) {
        static const std::unordered_set<DataType> set{
            DataType::F32, DataType::I32, DataType::I64, DataType::FP16, DataType::F64, DataType::BF16};
        return set.find(dt) != set.end();
    }

    bool isNumbericDataType(DataType dt) {
        static const std::unordered_set<DataType> set{
            DataType::F32, DataType::U8, DataType::I8, DataType::U16, DataType::I16, DataType::I32,
            DataType::I64, DataType::FP16, DataType::F64, DataType::U32, DataType::U64, DataType::BF16};
        return set.find(dt) != set.end();
    }

    bool isBool(DataType dt) {
        return dt == DataType::Bool;
    }

    size_t dataTypeSize(DataType dt) {
#define RETURN_SIZE(TYPE) \
    case DataType::TYPE:  \
        return sizeof(primitive_t<DataType::TYPE>::type)

        switch (dt) {
            RETURN_SIZE(F32);
            RETURN_SIZE(U8);
            RETURN_SIZE(I8);
            RETURN_SIZE(U16);
            RETURN_SIZE(I16);
            RETURN_SIZE(I32);
            RETURN_SIZE(I64);
            RETURN_SIZE(Bool);
            RETURN_SIZE(FP16);
            RETURN_SIZE(F64);
            RETURN_SIZE(U32);
            RETURN_SIZE(U64);
            RETURN_SIZE(Complex64);
            RETURN_SIZE(Complex128);
            RETURN_SIZE(BF16);
            default:
                RUNTIME_ERROR("Unreachable");
        }
    }

}// namespace refactor::common
