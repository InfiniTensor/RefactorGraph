#include "computation/operators/global_pool.h"
#include "common.h"

namespace refactor::computation {
    using Op = GlobalPool;

    auto Op::typeId(PoolType type) noexcept -> size_t {
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
    auto Op::opTypeId() const noexcept -> size_t {
        return typeId(type);
    }
    auto Op::name() const noexcept -> std::string_view {
        switch (type) {
            case PoolType::Average:
                return "GlobalAveragePool";
            case PoolType::Lp:
                return "GlobalLpPool";
            case PoolType::Max:
                return "GlobalMaxPool";
            default:
                UNREACHABLE();
        }
    }
    auto Op::candidateKernels(Target target) const noexcept -> kernel::CollectorBox {
        using Collector_ = kernel::GlobalPoolCollector;
        return std::make_unique<Collector_>(target, type);
    }

}// namespace refactor::computation
