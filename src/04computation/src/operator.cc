#include "computation/operator.h"

namespace refactor::computation {

    bool Operator::isLayoutDependent() const { return false; }
    void Operator::transposeTo(LayoutType) { isLayoutDependent() && UNREACHABLEX(bool, ""); }
    std::vector<kernel::CandidateBox> Operator::candidateKernels() const { return {}; }

    bool LayoutDependentOperator::isLayoutDependent() const { return true; }
    void LayoutDependentOperator::transposeTo(LayoutType) { UNREACHABLE(); }

    bool AxisRankOperator::isLayoutDependent() const { return rank != 4; }
    void AxisRankOperator::transposeTo(LayoutType target) {
        Operator::transposeTo(target);
        switch (target) {
            case LayoutType::NCHW:
                axis = (int[]){0, 2, 3, 1}[axis];
                break;
            case LayoutType::NHWC:
                axis = (int[]){0, 3, 1, 2}[axis];
                break;
            default:
                UNREACHABLE();
        }
    }

}// namespace refactor::computation
