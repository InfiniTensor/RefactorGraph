#include "computation/operators/pool.h"
#include "refactor/common.h"

namespace refactor::computation {

    size_t Pool::typeId(PoolType type) noexcept {
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
    size_t Pool::opTypeId() const noexcept {
        return typeId(type);
    }
    std::string_view Pool::name() const noexcept {
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
