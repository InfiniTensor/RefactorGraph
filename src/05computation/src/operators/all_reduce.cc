#include "computation/operators/all_reduce.h"
#include "kernel/collectors/all_reduce.h"

namespace refactor::computation {
    using Op = AllReduce;
    using Ty = kernel::AllReduceType;

    auto Op::typeId(Ty type_) noexcept -> size_t {
        switch (type_) {
            case Ty::Sum: {
                static uint8_t ID = 1;
                return reinterpret_cast<size_t>(&ID);
            }
            case Ty::Avg: {
                static uint8_t ID = 1;
                return reinterpret_cast<size_t>(&ID);
            }
            case Ty::Min: {
                static uint8_t ID = 1;
                return reinterpret_cast<size_t>(&ID);
            }
            case Ty::Max: {
                static uint8_t ID = 1;
                return reinterpret_cast<size_t>(&ID);
            }
            case Ty::Prod: {
                static uint8_t ID = 1;
                return reinterpret_cast<size_t>(&ID);
            }
            default:
                UNREACHABLE();
        }
    }

    auto Op::opTypeId() const noexcept -> size_t { return typeId(type); }

    auto Op::name() const noexcept -> std::string_view {
        switch (type) {
            case Ty::Sum:
                return "AllReduceSum";
            case Ty::Avg:
                return "AllReduceAvg";
            case Ty::Min:
                return "AllReduceMin";
            case Ty::Max:
                return "AllReduceMax";
            case Ty::Prod:
                return "AllReduceProd";
            default:
                UNREACHABLE();
        }
    }

    auto Op::candidateKernels(Target target) const -> kernel::CollectorBox {
        using Collector_ = kernel::AllReduceCollector;
        return std::make_unique<Collector_>(target, type);
    }

}// namespace refactor::computation
