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

    Shape TransposeBox::verify(Tensor const &a) const noexcept {
        Shape ans = {};
        if (a.rank() != 4) {
            return ans;
        }
        if (perm == Shape{0, 2, 3, 1}) {
            ans = {a.shape[0], a.shape[2], a.shape[3], a.shape[1]};
        } else if (perm == Shape{1, 0, 2, 3}) {
            ans = {a.shape[1], a.shape[0], a.shape[2], a.shape[3]};
        } else if (perm == Shape{0, 3, 1, 2}) {
            ans = {a.shape[0], a.shape[3], a.shape[1], a.shape[2]};
        }
        return ans;
    }

    bool TransposeBox::compute(Tensor const &a, Tensor &out) const noexcept {
        if (a.data == nullptr) {
            return false;
        }
        // compute
        auto kernels = this->base->candidateKernels(Target::Cpu)->filter({a}, {out});
        ASSERT(kernels.size() != 0, "do not supposrt this kernel");
        runtime::Resources res;
        auto rou = kernels[0]->lower(res);
        void const *inputs[]{*a.data};
        void *outputs[]{*out.data};
        rou(res, inputs, outputs);
        return true;
    }

}// namespace refactor::computation
