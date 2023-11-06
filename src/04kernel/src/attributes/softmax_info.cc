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

    void SoftmaxInfo::locate(uint_lv2 k, uint_lv2 ans[]) const noexcept {
        std::fill_n(ans, 2, 0);
        long rem = k;
        auto d = std::div(rem, mid * post);
        ans[0] = d.quot;
        ans[1] = d.rem % post;
    };

}// namespace refactor::kernel