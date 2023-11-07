#include "kernel/attributes/softmax_info.h"
#include <numeric>

namespace refactor::kernel {
    SoftmaxInfo::SoftmaxInfo(Tensor const &data, uint_lv2 axis) noexcept : pre(1),
                                                                           mid(data.shape[axis]),
                                                                           post(1),
                                                                           size(1),
                                                                           type(data.dataType) {
        auto eleSize = data.dataType.size();
        auto axisIt = data.shape.begin() + axis;
        pre = std::accumulate(data.shape.begin(), axisIt, 1, std::multiplies<>());
        post = std::accumulate(++axisIt, data.shape.end(), 1, std::multiplies<>());
        size = std::accumulate(data.shape.begin(), data.shape.end(), 1, std::multiplies<>());
    };

}// namespace refactor::kernel
