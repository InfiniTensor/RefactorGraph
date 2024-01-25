#include "common.h"
#include <numeric>

namespace refactor::onnx {

    ShapeResult pool(SmallInts<4> const &input,
                     Ints const &kernel,
                     OptionalIntsRef const &dilations,
                     OptionalIntsRef const &pads,
                     OptionalIntsRef const &strides) {
        auto dim = input.size();
        if (dim != kernel.size()) {
            return Err(ERROR_MSG("Input shape not support"));
        }
        int64_t const *dilations_ = nullptr,
                      *pads_ = nullptr,
                      *strides_ = nullptr;
        if (dilations) {
            if (dilations->get().size() != dim) {
                return Err(ERROR_MSG("Input shape not support"));
            } else {
                dilations_ = dilations->get().data();
            }
        }
        if (pads) {
            if (pads->get().size() != dim * 2) {
                return Err(ERROR_MSG("Input shape not support"));
            } else {
                pads_ = pads->get().data();
            }
        }
        if (strides) {
            if (strides->get().size() != dim) {
                return Err(ERROR_MSG("Input shape not support"));
            } else {
                strides_ = strides->get().data();
            }
        }
        Shape ans(dim, DimExpr(1));
        auto r = range0_(dim);
        std::transform(r.begin(), r.end(), ans.begin(),
                       [&input, &kernel, dim, dilations_, pads_, strides_](auto i) {
                           auto d = input[i] + (pads_ ? (pads_[i] + pads_[i + dim]) : 0);
                           auto k = (kernel[i] - 1) * (dilations_ ? dilations_[i] : 1) + 1;
                           return DimExpr((d - k) / (strides_ ? strides_[i] : 1) + 1);
                       });
        return Ok(std::move(ans));
    }

}// namespace refactor::onnx
