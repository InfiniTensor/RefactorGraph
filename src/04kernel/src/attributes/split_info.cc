#include "kernel/attributes/split_info.h"
#include <numeric>

namespace refactor::kernel {

    SplitInfo::SplitInfo(dim_t axis, TensorRefs const &outputs)
        : blockCount(0),
          sum(std::accumulate(
              outputs.begin(), outputs.end(), 0,
              [=](auto acc, auto const &ref) { return acc + ref.get().shape[axis]; })),
          segments(outputs.size()) {
        ASSERT(!outputs.empty(), "");
        auto eleSize = outputs[0].get().dataType.size();
        auto const &shape = outputs[0].get().shape;
        auto axisIt = shape.begin() + axis;
        blockCount = std::accumulate(shape.begin(), axisIt, 1, std::multiplies());
        auto postfix = std::accumulate(++axisIt, shape.end(), eleSize, std::multiplies());
        sum *= postfix;
        std::transform(outputs.begin(), outputs.end(),
                       segments.begin(),
                       [=](auto const &ref) { return ref.get().shape[axis] * postfix; });
    }

    dim_t SplitInfo::unit(dim_t maxBlockSize) const noexcept {
        auto or_ = std::accumulate(segments.begin(), segments.end(), 0u,
                                   [&](auto acc, auto seg) { return acc | seg; });
        return std::gcd(or_, maxBlockSize);
    }

}// namespace refactor::kernel
