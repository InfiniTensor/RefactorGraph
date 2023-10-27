#include "kernel/attributes/where_info.h"

namespace refactor::kernel {
    WhereBroadcast::WhereBroadcast(Shape const &c_, Shape const &x_, Shape const &y_, Shape const &output) noexcept
        : _strides(), _size(1) {
        auto rank = output.size();
        _strides.resize(rank * 4, 1);
        auto getStride = [](auto const i, Shape const &s) {
            return s.size() > i ? s[s.size() - i] : 0;
        };

        uint_lv2 cMul = 1,
                 xMul = 1, yMul = 1;
        for (auto i : range0_(rank).rev()) {
            _strides[4 * i + 0] = _size;
            _strides[4 * i + 1] = cMul;
            _strides[4 * i + 2] = xMul;
            _strides[4 * i + 3] = yMul;
            _size *= output[i];
            cMul *= getStride(rank - i, c_);
            xMul *= getStride(rank - i, x_);
            yMul *= getStride(rank - i, y_);
        }
    }

    auto WhereBroadcast::locate(uint32_t k) const noexcept -> WhereBroadcast::Triplet {
        uint32_t a = 0, b = 0, c = 0;
        long rem = k;
        for (auto i : range0_(_strides.size() / 4)) {
            auto d = std::div(rem, _strides[4 * i]);
            a += _strides[4 * i + 1] * d.quot;
            b += _strides[4 * i + 2] * d.quot;
            c += _strides[4 * i + 3] * d.quot;
            rem = d.rem;
        }
        return {a, b, c};
    }

    auto WhereBroadcast::size() const noexcept -> uint32_t {
        return _size;
    }

}// namespace refactor::kernel
