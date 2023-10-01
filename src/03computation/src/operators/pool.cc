#include "computation/operators/pool.h"
#include "common/error_handler.h"

namespace refactor::computation {

    size_t Pool::typeId(PoolType type) {
        switch (type) {
            case PoolType::Average: {
                static uint8_t ID = 1;
                return reinterpret_cast<size_t>(&ID);
            }
            case PoolType::Lp: {
                static uint8_t ID = 1;
                return reinterpret_cast<size_t>(&ID);
            }
            case PoolType::Max: {
                static uint8_t ID = 1;
                return reinterpret_cast<size_t>(&ID);
            }
            default:
                UNREACHABLE();
        }
    }
    size_t Pool::opTypeId() const {
        return typeId(type);
    }
    std::string_view Pool::name() const {
        switch (type) {
            case PoolType::Average:
                return "AveragePool";
            case PoolType::Lp:
                return "LpPool";
            case PoolType::Max:
                return "MaxPool";
            default:
                UNREACHABLE();
        }
    }

}// namespace refactor::computation
