#include "computation/operators/pool.h"
#include "common.h"

namespace refactor::computation {
    using Op = Pool;

    Op::Pool(PoolType type_,
             bool ceil_,
             decltype(kernelShape) kernelShape_,
             PoolAttributes attrs) noexcept
        : Operator(),
          type(type_),
          ceil(ceil_),
          kernelShape(std::move(kernelShape_)),
          attributes(std::move(attrs)) {}

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
                return "AveragePool";
            case PoolType::Lp:
                return "LpPool";
            case PoolType::Max:
                return "MaxPool";
            default:
                UNREACHABLE();
        }
    }
    auto Op::candidateKernels(Target target) const noexcept -> kernel::CollectorBox {
        using Collector_ = kernel::PoolCollector;
        return std::make_unique<Collector_>(target, type, ceil, kernelShape, attributes);
    }

}// namespace refactor::computation
