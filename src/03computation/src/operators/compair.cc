#include "computation/operators/compair.h"

namespace refactor::computation {

    size_t Compair::typeId(CompairType type) {
        switch (type) {
            case CompairType::EQ: {
                static uint8_t ID = 1;
                return reinterpret_cast<size_t>(&ID);
            }
            case CompairType::NE: {
                static uint8_t ID = 2;
                return reinterpret_cast<size_t>(&ID);
            }
            case CompairType::LT: {
                static uint8_t ID = 3;
                return reinterpret_cast<size_t>(&ID);
            }
            case CompairType::LE: {
                static uint8_t ID = 4;
                return reinterpret_cast<size_t>(&ID);
            }
            case CompairType::GT: {
                static uint8_t ID = 5;
                return reinterpret_cast<size_t>(&ID);
            }
            case CompairType::GE: {
                static uint8_t ID = 6;
                return reinterpret_cast<size_t>(&ID);
            }
            default:
                UNREACHABLE();
        }
    }
    size_t Compair::opTypeId() const {
        return typeId(type);
    }
    std::string_view Compair::name() const {
        switch (type) {
            case CompairType::EQ:
                return "Equal";
            case CompairType::NE:
                return "NotEqual";
            case CompairType::LT:
                return "Less";
            case CompairType::LE:
                return "LessEqual";
            case CompairType::GT:
                return "Greater";
            case CompairType::GE:
                return "GreaterEqual";
            default:
                UNREACHABLE();
        }
    }

}// namespace refactor::computation
