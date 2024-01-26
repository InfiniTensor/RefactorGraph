#include "frontend/operator.h"
#include <fmtlog.h>

namespace refactor::frontend {

    void Attributes::insert(std::string key, Attribute value) {
        map.insert({std::move(key), std::move(value)});
    }
    bool Attributes::empty() const {
        return map.empty();
    }
    auto Attributes::operator[](const char *key) -> Attribute & {
        return map.at(key);
    }
    auto Attributes::operator[](const char *key) const -> Attribute const & {
        return map.at(key);
    }
    auto Attributes::get(const char *key) -> std::optional<std::reference_wrapper<Attribute>> {
        auto it = map.find(key);
        return it != map.end() ? std::make_optional(std::ref(it->second)) : std::nullopt;
    }
    auto Attributes::get(const char *key) const -> std::optional<std::reference_wrapper<Attribute const>> {
        auto it = map.find(key);
        return it != map.end() ? std::make_optional(std::cref(it->second)) : std::nullopt;
    }
    auto Attributes::getOrInsert(const char *key, Attribute otherwise) -> Attribute & {
        auto [it, ok] = map.try_emplace(key, std::move(otherwise));
        return it->second;
    }

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

}// namespace refactor::frontend
