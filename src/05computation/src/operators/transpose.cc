#include "computation/operators/transpose.h"

namespace refactor::computation {
    using Op = Transpose;

    auto Op::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }
    auto Op::opTypeId() const noexcept -> size_t { return typeId(); }
    auto Op::name() const noexcept -> std::string_view { return "Transpose"; }
    auto Op::isLayoutDependent() const noexcept -> bool { return perm.size() != 4; }
    auto Op::transposeTo(LayoutType target) noexcept -> void {
        Operator::transposeTo(target);
        switch (target) {
            case LayoutType::NCHW:
                for (auto &axis : perm) {
                    axis = PERMUTATION(NHWC, NCHW)[axis];
                }
                break;
            case LayoutType::NHWC:
                for (auto &axis : perm) {
                    axis = PERMUTATION(NCHW, NHWC)[axis];
                }
                break;
            default:
                UNREACHABLE();
        }
    }
    auto Op::candidateKernels(Target target) const noexcept -> kernel::CollectorBox {
        using Collector_ = kernel::TransposeCollector;
        return std::make_unique<Collector_>(target, perm);
    }
    auto Op::serialize() const noexcept -> std::string {
        return fmt::format("{}({})", name(), vec2str(perm));
    }

}// namespace refactor::computation
