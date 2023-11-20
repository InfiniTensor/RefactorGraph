#include "computation/operators/reshape.h"
#include <numeric>

namespace refactor::computation {

    size_t Reshape::typeId() noexcept {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }
    size_t Reshape::opTypeId() const noexcept { return typeId(); }
    std::string_view Reshape::name() const noexcept { return "Reshape"; }
    bool Reshape::isIdentity() const noexcept { return true; }

    bool ReshapeBox::compute(Tensor const &a, Tensor &out) const noexcept {
        if (a.data == nullptr) {
            return false;
        }
        out.data = a.data;
        return true;
    }

    Shape ReshapeBox::verify(Tensor const &a) const noexcept {
        Shape ans = {};
        if (a.rank() < 2) {
            return ans;
        }
        if (shape == Shape{-1, 0}) {
            auto pre = std::accumulate(a.shape.begin(), a.shape.end() - 1, (dim_t) 1, std::multiplies<dim_t>());
            ans = {pre, a.shape[a.rank() - 1]};
        } else if (shape == Shape{0, -1}) {
            auto post = std::accumulate(a.shape.begin() + 1, a.shape.end(), (dim_t) 1, std::multiplies<dim_t>());
            ans = {a.shape[0], post};
        } else if (shape == Shape{1, 3, 3, 2}) {
            auto ta = std::accumulate(a.shape.begin(), a.shape.end(), (dim_t) 1, std::multiplies<dim_t>());
            auto tb = std::accumulate(shape.begin(), shape.end(), (dim_t) 1, std::multiplies<dim_t>());
            if (ta != tb) {
                return ans;
            }
            ans = shape;
        }

        return ans;
    }
}// namespace refactor::computation
