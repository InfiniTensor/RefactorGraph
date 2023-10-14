#include "common.h"
#include "refactor/common.h"
#include <fmtlog.h>
#include <numeric>
#include <vector>

namespace refactor::onnx {

    ShapeResult multidirBroadcast(ShapeRefs const &inputs) {
        using Iter = std::reverse_iterator<Shape::const_iterator>;
        std::vector<std::pair<Iter, Iter>> iters;
        iters.reserve(inputs.size());
        for (auto const &input : inputs) {
            iters.emplace_back(input.get().rbegin(), input.get().rend());
        }
        Shape ans;
        while (true) {
            std::optional<DimExpr> dim = std::nullopt;
            for (size_t i = 0; i < iters.size();) {
                if (iters[i].first != iters[i].second) {
                    auto new_ = *iters[i].first++;
                    if (!dim || *dim == DimExpr(1)) {
                        dim = std::move(new_);
                    } else if (new_ != DimExpr(1) && new_ != *dim) {
                        loge("shape broadcast failed");
                        for (auto input : inputs) {
                            loge("{}", shapeFormat(input.get()));
                        }
                        return Err(ERROR_MSG("Shape broadcast failed"));
                    }
                    ++i;
                } else {
                    std::swap(iters[i], iters.back());
                    iters.pop_back();
                }
            }
            if (dim) {
                ans.emplace_back(std::move(*dim));
            } else {
                break;
            }
        }
        std ::reverse(ans.begin(), ans.end());
        return Ok(ans);
    }

    bool unidirBroadcast(Shape const &target, Shape const &test) {
        if (target.size() < test.size()) {
            return false;
        } else {
            for (auto i = target.rbegin(), j = test.rbegin(); j != test.rend(); ++i, ++j) {
                if (*j != *i && *j != DimExpr(1)) {
                    return false;
                }
            }
            return true;
        }
    }

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

    Attribute defaultOr(Attributes &attrs, std::string const &name, Attribute defaultValue) {
        auto iter = attrs.find(name);
        return iter == attrs.end() ? defaultValue : std::move(iter->second);
    }

}// namespace refactor::onnx
