#include "kernel/attributes/split_info.h"
#include <numeric>

namespace refactor::kernel {

    SplitInfo::SplitInfo(uint_lv2 axis, TensorRefs const &outputs) noexcept
        : _values(axis + outputs.size()),
          _prefixLen(0) {
        ASSERT(!outputs.empty(), "");
        auto const &shape = outputs[0].get().shape;
        auto postfix = std::accumulate(
            shape.begin() + axis + 1, shape.end(), 1,
            std::multiplies<>());
        auto next = std::copy_if(shape.begin(), shape.begin() + axis,
                                 _values.begin(),
                                 [](auto x) { return x != 1; });
        _prefixLen = std::distance(_values.begin(), next);
        auto end = std::transform(outputs.begin(), outputs.end(),
                                  next,
                                  [=](auto const &ref) { return ref.get().shape[axis] * postfix; });
        _values.erase(end, _values.end());
        _values.shrink_to_fit();
    }

    auto SplitInfo::prefix() const noexcept -> slice_t<uint_lv2> {
        return slice(_values.data(), _prefixLen);
    }
    auto SplitInfo::postfix() const noexcept -> slice_t<uint_lv2> {
        return slice(_values.data() + _prefixLen, _values.size() - _prefixLen);
    }

}// namespace refactor::kernel
