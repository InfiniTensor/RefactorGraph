#include "kernel/attributes/split_info.h"
#include <numeric>

namespace refactor::kernel {

    SplitInfo::SplitInfo(uint_lv2 axis, TensorRefs const &outputs) noexcept
        : segments(outputs.size()),
          blockCount(0),
          sum(std::accumulate(outputs.begin(), outputs.end(), 0,
                              [=](auto acc, auto const &ref) { return acc + ref.get().shape[axis]; })) {
        ASSERT(!outputs.empty(), "");
        auto const &shape = outputs[0].get().shape;
        blockCount = std::accumulate(shape.begin(), shape.begin() + axis, 1, std::multiplies<>());
        auto postfix = std::accumulate(shape.begin() + axis + 1, shape.end(), 1, std::multiplies<>());
        sum *= postfix;
        std::transform(outputs.begin(), outputs.end(),
                       segments.begin(),
                       [=](auto const &ref) { return ref.get().shape[axis] * postfix; });
    }

}// namespace refactor::kernel
