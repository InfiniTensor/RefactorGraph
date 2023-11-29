#include "computation/operators/reduce.h"
#include <numeric>

namespace refactor::computation {
    using Op = Reduce;

    Op::Reduce(ReduceType type_,
               kernel::Axes axes_,
               uint32_t rank_,
               bool keepDims_) noexcept
        : Operator(),
          type(type_),
          axes(std::move(axes_)),
          rank(rank_),
          keepDims(keepDims_) {
        if (axes.empty()) {
            axes.resize(rank);
            std::iota(axes.begin(), axes.end(), 0);
        }
    }

    auto Op::typeId(ReduceType type) noexcept -> size_t {
        switch (type) {
            case ReduceType::Mean: {
                static uint8_t ID = 1;
                return reinterpret_cast<size_t>(&ID);
            }
            case ReduceType::L1: {
                static uint8_t ID = 2;
                return reinterpret_cast<size_t>(&ID);
            }
            case ReduceType::L2: {
                static uint8_t ID = 3;
                return reinterpret_cast<size_t>(&ID);
            }
            case ReduceType::LogSum: {
                static uint8_t ID = 4;
                return reinterpret_cast<size_t>(&ID);
            }
            case ReduceType::LogSumExp: {
                static uint8_t ID = 5;
                return reinterpret_cast<size_t>(&ID);
            }
            case ReduceType::Max: {
                static uint8_t ID = 6;
                return reinterpret_cast<size_t>(&ID);
            }
            case ReduceType::Min: {
                static uint8_t ID = 7;
                return reinterpret_cast<size_t>(&ID);
            }
            case ReduceType::Prod: {
                static uint8_t ID = 8;
                return reinterpret_cast<size_t>(&ID);
            }
            case ReduceType::Sum: {
                static uint8_t ID = 9;
                return reinterpret_cast<size_t>(&ID);
            }
            case ReduceType::SumSquare: {
                static uint8_t ID = 10;
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
            case ReduceType::Mean:
                return "ReduceMean";
            case ReduceType::L1:
                return "ReduceL1";
            case ReduceType::L2:
                return "ReduceL2";
            case ReduceType::LogSum:
                return "ReduceLogSum";
            case ReduceType::LogSumExp:
                return "ReduceLogSumExp";
            case ReduceType::Max:
                return "ReduceMax";
            case ReduceType::Min:
                return "ReduceMin";
            case ReduceType::Prod:
                return "ReduceProd";
            case ReduceType::Sum:
                return "ReduceSum";
            case ReduceType::SumSquare:
                return "ReduceSumSquare";
            default:
                UNREACHABLE();
        }
    }
    auto Op::isLayoutDependent() const noexcept -> bool { return rank != 4; }
    auto Op::transposeTo(LayoutType target) noexcept -> void {
        Operator::transposeTo(target);
        switch (target) {
            case LayoutType::NCHW:
                for (auto &axis : axes) {
                    axis = PERMUTATION(NHWC, NCHW)[axis];
                }
                break;
            case LayoutType::NHWC:
                for (auto &axis : axes) {
                    axis = PERMUTATION(NCHW, NHWC)[axis];
                }
                break;
            default:
                UNREACHABLE();
        }
    }
    auto Op::candidateKernels(Target target) const noexcept -> kernel::CollectorBox {
        return std::make_unique<kernel::ReduceCollector>(target, type, axes);
    }

}// namespace refactor::computation
