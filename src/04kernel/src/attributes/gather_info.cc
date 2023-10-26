#include "kernel/attributes/gather_info.h"
#include <numeric>

namespace refactor::kernel {

    GatherInfo::GatherInfo(uint_lv2 axis, Tensor const &data, Tensor const &indices) noexcept
        : strides(indices.rank(), 1),
          prefix(0),
          postfix(0),
          midSize(0),
          idxType(indices.dataType) {
        auto eleSize = data.dataType.size();
        auto const &shape = data.shape;
        auto axisIt = shape.begin() + axis;
        prefix = std::accumulate(shape.begin(), axisIt++, 1, std::multiplies<>());
        postfix = std::accumulate(axisIt, shape.end(), eleSize, std::multiplies<>());
        if (!strides.empty()) {
            for (auto i : range(1ul, strides.size()).rev()) {
                strides[i - 1] = strides[i] * indices.shape[i];
            }
            midSize = strides[0] * indices.shape[0];
        } else {
            midSize = 1;
        }
    }

}// namespace refactor::kernel
