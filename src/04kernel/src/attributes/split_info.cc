#include "kernel/attributes/split_info.h"
#include <numeric>

namespace refactor::kernel {

    SplitInfo::SplitInfo(uint_lv2 axis, TensorRefs const &outputs) noexcept
        : segments(outputs.size()),
          blockCount(0),
          sum(std::accumulate(
              outputs.begin(), outputs.end(), 0,
              [=](auto acc, auto const &ref) { return acc + ref.get().shape[axis]; })) {
        ASSERT(!outputs.empty(), "");
        auto eleSize = outputs[0].get().dataType.size();
        auto const &shape = outputs[0].get().shape;
        auto axisIt = shape.begin() + axis;
        blockCount = std::accumulate(shape.begin(), axisIt++, 1, std::multiplies<>());
        auto postfix = eleSize * std::accumulate(axisIt, shape.end(), 1, std::multiplies<>());
        sum *= postfix;
        std::transform(outputs.begin(), outputs.end(),
                       segments.begin(),
                       [=](auto const &ref) { return ref.get().shape[axis] * postfix; });
    }

}// namespace refactor::kernel
