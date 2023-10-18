#include "computation/operators/transpose.h"

namespace refactor::computation {

    size_t Transpose::typeId() noexcept {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }
    size_t Transpose::opTypeId() const noexcept { return typeId(); }
    std::string_view Transpose::name() const noexcept { return "Transpose"; }
    bool Transpose::isLayoutDependent() const noexcept { return perm.size() != 4; }
    void Transpose::transposeTo(LayoutType target) noexcept {
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

}// namespace refactor::computation
