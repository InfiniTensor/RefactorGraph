#include "computation/operator.h"

namespace refactor::computation {

    bool Operator::isLayoutDependent() const { return false; }
    bool Operator::isIdentity() const { return false; }
    void Operator::transposeTo(LayoutType) { isLayoutDependent() && UNREACHABLEX(bool, ""); }
    kernel::CollectorBox Operator::candidateKernels(Target) const {
        TODO(fmt::format("Operator \"{}\" lower to kernel not implemented", name()));
    }
    std::string Operator::serialize() const {
        return fmt::format("{}(...)", name());
    }

    bool LayoutDependentOperator::isLayoutDependent() const { return true; }
    void LayoutDependentOperator::transposeTo(LayoutType) { UNREACHABLE(); }

    bool AxisRankOperator::isLayoutDependent() const { return rank != 4; }
    void AxisRankOperator::transposeTo(LayoutType target) {
        Operator::transposeTo(target);
        switch (target) {
            case LayoutType::NCHW:
                axis = PERMUTATION(NHWC, NCHW)[axis];
                break;
            case LayoutType::NHWC:
                axis = PERMUTATION(NCHW, NHWC)[axis];
                break;
            default:
                UNREACHABLE();
        }
    }
    std::string AxisRankOperator::serialize() const {
        return fmt::format("{}({}/{})", name(), axis, rank);
    }

}// namespace refactor::computation
