#include "computation/operators/slice.h"
#include "kernel/collectors/slice.h"

namespace refactor::computation {
    using Op = Slice;

    Op::Slice(Dimensions dims_) noexcept
        : LayoutDependentOperator(),
          dims(std::move(dims_)) { assert(!dims.empty()); }

    auto Op::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }
    auto Op::opTypeId() const noexcept -> size_t { return typeId(); }
    auto Op::name() const noexcept -> std::string_view { return "Slice"; }
    auto Op::candidateKernels(Target target) const noexcept -> kernel::CollectorBox {
        using Collector_ = kernel::SliceCollector;
        return std::make_unique<Collector_>(target, dims);
    }
    auto Op::serialize() const noexcept -> std::string {
        std::stringstream ss;
        ss << name() << "([";
        for (auto const &d : dims) {
            ss << ' ' << d.start << '+' << d.length << 'x' << d.step;
        }
        ss << " ])";
        return ss.str();
    }

}// namespace refactor::computation
